"""Crash-durable, content-addressed Phase 8 checkpoints."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import subprocess
from collections import OrderedDict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal

import torch

from sebgpt.model.mini_gpt import (
    ATTENTION_DROPOUT_SEED,
    BLOCK_PARAMETER_INITIALIZATION_SEEDS,
    DROPOUT_PROBABILITY,
    EMBEDDING_INITIALIZATION_SEED,
    FEED_FORWARD_DROPOUT_SEED,
    HEAD_WIDTH,
    HIDDEN_WIDTH,
    LAYER_NORM_EPSILON,
    MAX_SEQUENCE_LENGTH,
    MODEL_WIDTH,
    NUMBER_OF_BLOCKS,
    NUMBER_OF_HEADS,
    OUTPUT_HEAD_INITIALIZATION_SEED,
    PARAMETER_DEVICE,
    PARAMETER_DTYPE,
    MiniGPT,
)
from sebgpt.tokenization.vocabulary_artifact import (
    EXPECTED_ARTIFACT_SHA256,
    VOCABULARY_RELATIVE_PATH,
    load_accepted_vocabulary_binding,
)
from sebgpt.training.phase8_optimization import (
    _validate_optimizer,
    create_phase8_optimizer,
)
from sebgpt.training.phase8_types import (
    Phase8CheckpointError,
    Phase8CheckpointReference,
    Phase8Configuration,
    Phase8ContractError,
    Phase8Evaluation,
    Phase8GovernanceError,
    Phase8MetricState,
    Phase8NumericalError,
    Phase8Progress,
    Phase8PublicationError,
    Phase8RuntimeIdentity,
    Phase8TrainingState,
    Phase8TypeError,
    _fixed_configuration,
    _make_configuration,
    _make_metric_state,
    _make_training_state,
    _validate_runtime_identity,
    _validate_runtime,
)


PHASE7_CLOSURE_COMMIT = "33d4510421107848c4aa8a6014f4a7b1e391065a"
PHASE7_CONTRACT_COMMIT = "60b2a9cce55da79ccc9fbd03fad014cb2a939290"
PHASE7_IMPLEMENTATION_COMMIT = "3139b1736f005fe903e2ea111d91934478b5a683"
PHASE8_ORIGINAL_CONTRACT_COMMIT = "09b2c2e0487a0b7a766655951422471085d943a8"
PHASE8_CONTRACT_COMMIT = "c50d77ac935bdf924b9b429a5776c419982a5d11"
PHASE8_IMPLEMENTATION_COMMIT = "809834323d53407cb4a54ae539585bb3d78856eb"
MINI_GPT_SPEC_SHA256 = "3e1987658d4c9ece59bebbea7061f939c1138aaa73751a523f659f8b3217c022"
MINI_GPT_EXPORT_SHA256 = "af056658e6d7ffe6de89b3ac0486929305655ea76029240d26869b705a2a5d86"
MINI_GPT_SOURCE_SHA256 = "6b8db96577a0ad1f5daa4649d7ca9375e59873d4b635c3f76e75a6751f74f12d"
MINI_GPT_TEST_SHA256 = "95ec62d1164c48525e59e33f3f90fada6d482c84a31ca16edba38178b16b2cab"
PHASE8_SPEC_SHA256 = "1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd"
TOKENIZER_IMPLEMENTATION_COMMIT = "de7a7f096fbbd8607c944412eaef30be9b686b56"
PHASE1_MANIFEST_SHA256 = "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"
PROCESSING_MANIFEST_SHA256 = "bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc"
RUN_ID_PATTERN = re.compile(r"EXP-[0-9]{8}-[0-9]{2}\Z")
LOGICAL_ID_PATTERN = re.compile(r"epoch-00(?:0[1-9]|10)\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
MAXIMUM_CATALOG_BYTES = 16_384
MAXIMUM_OBJECT_BYTES = 67_108_864
_PHASE8_SOURCE_TEST_PATHS = (
    "src/sebgpt/training/__init__.py",
    "src/sebgpt/training/phase8_types.py",
    "src/sebgpt/training/phase8_data.py",
    "src/sebgpt/training/phase8_optimization.py",
    "src/sebgpt/training/phase8_checkpoint.py",
    "src/sebgpt/training/phase8_experiment.py",
    "tests/test_phase8_types.py",
    "tests/test_phase8_data.py",
    "tests/test_phase8_optimization.py",
    "tests/test_phase8_checkpoint.py",
    "tests/test_phase8_experiment.py",
)
_POST_IMPLEMENTATION_ALLOWED_PATHS = frozenset(
    (
        "DECISIONS.md",
        "PROJECT_STATE.md",
        "README.md",
        "ROADMAP.md",
        "docs/TRAINING_CHECKPOINTING_SPEC.md",
        "src/sebgpt/training/phase8_checkpoint.py",
        "src/sebgpt/training/phase8_experiment.py",
        "tests/test_phase8_types.py",
        "tests/test_phase8_checkpoint.py",
        "tests/test_phase8_experiment.py",
    )
)
_DROPOUT_NAMES = (
    "blocks.0.attention_dropout",
    "blocks.0.feed_forward_dropout",
    "blocks.1.attention_dropout",
    "blocks.1.feed_forward_dropout",
    "blocks.2.attention_dropout",
    "blocks.2.feed_forward_dropout",
    "blocks.3.attention_dropout",
    "blocks.3.feed_forward_dropout",
)
_ACCEPTED_MODE_STATE = (
    ("", True),
    ("representation", True),
    ("blocks", True),
    ("blocks.0", True),
    ("blocks.0.norm1", True),
    ("blocks.0.attention", True),
    ("blocks.0.attention.heads", True),
    ("blocks.0.attention.heads.0", True),
    ("blocks.0.attention.heads.1", True),
    ("blocks.0.attention.heads.2", True),
    ("blocks.0.attention.heads.3", True),
    ("blocks.0.attention_dropout", True),
    ("blocks.0.norm2", True),
    ("blocks.0.feed_forward", True),
    ("blocks.0.feed_forward_dropout", True),
    ("blocks.1", True),
    ("blocks.1.norm1", True),
    ("blocks.1.attention", True),
    ("blocks.1.attention.heads", True),
    ("blocks.1.attention.heads.0", True),
    ("blocks.1.attention.heads.1", True),
    ("blocks.1.attention.heads.2", True),
    ("blocks.1.attention.heads.3", True),
    ("blocks.1.attention_dropout", True),
    ("blocks.1.norm2", True),
    ("blocks.1.feed_forward", True),
    ("blocks.1.feed_forward_dropout", True),
    ("blocks.2", True),
    ("blocks.2.norm1", True),
    ("blocks.2.attention", True),
    ("blocks.2.attention.heads", True),
    ("blocks.2.attention.heads.0", True),
    ("blocks.2.attention.heads.1", True),
    ("blocks.2.attention.heads.2", True),
    ("blocks.2.attention.heads.3", True),
    ("blocks.2.attention_dropout", True),
    ("blocks.2.norm2", True),
    ("blocks.2.feed_forward", True),
    ("blocks.2.feed_forward_dropout", True),
    ("blocks.3", True),
    ("blocks.3.norm1", True),
    ("blocks.3.attention", True),
    ("blocks.3.attention.heads", True),
    ("blocks.3.attention.heads.0", True),
    ("blocks.3.attention.heads.1", True),
    ("blocks.3.attention.heads.2", True),
    ("blocks.3.attention.heads.3", True),
    ("blocks.3.attention_dropout", True),
    ("blocks.3.norm2", True),
    ("blocks.3.feed_forward", True),
    ("blocks.3.feed_forward_dropout", True),
    ("final_norm", True),
    ("head", True),
)
_TOP_LEVEL_KEYS = (
    "schema_version",
    "run",
    "authority",
    "configuration",
    "model_state",
    "optimizer_state",
    "progress",
    "random_state",
    "mode_state",
    "metric_state",
    "resume_lineage",
)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise Phase8GovernanceError(  # type: ignore[name-defined]
            "phase8.governance.repository",
            field="file",
        ) from None


def _git(repository_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="git",
        ) from error
    return result.stdout.strip()


def _git_blob_bytes(repository_root: Path, revision: str, relative_path: str) -> bytes:
    try:
        result = subprocess.run(
            ("git", "show", f"{revision}:{relative_path}"),
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        ) from error
    return result.stdout


def _require_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
    *,
    field: str,
) -> None:
    try:
        _git(repository_root, "merge-base", "--is-ancestor", ancestor, descendant)
    except Phase8GovernanceError as error:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field=field,
        ) from error


def _git_authority_fact(
    repository_root: Path,
    field: str,
    *arguments: str,
) -> str:
    try:
        return _git(repository_root, *arguments)
    except Phase8GovernanceError as error:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field=field,
        ) from error


def _run_authority_evidence_bytes(
    run_id: str,
    code_commit: str,
    pre_registration_commit: str,
    planned_record_sha256: str,
) -> bytes:
    value = {
        "schema_version": 1,
        "run_id": run_id,
        "code_commit": code_commit,
        "pre_registration_commit": pre_registration_commit,
        "planned_record_sha256": planned_record_sha256,
    }
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


_RUN_AUTHORITY_EVIDENCE_NAME = "phase8-run-authority.json"


def _run_authority_failure() -> Phase8GovernanceError:
    return Phase8GovernanceError(
        "phase8.governance.repository",
        field="pre_registration_commit",
    )


def _run_authority_directory_identity(fd: int) -> tuple[int, int]:
    try:
        value = os.fstat(fd)
    except OSError:
        raise _run_authority_failure() from None
    if not stat.S_ISDIR(value.st_mode):
        raise _run_authority_failure()
    return value.st_dev, value.st_ino


def _require_run_authority_entry_identity(
    parent_fd: int,
    name: str,
    expected: tuple[int, int],
    *,
    directory: bool,
) -> None:
    try:
        value = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise _run_authority_failure() from None
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(value.st_mode) or (value.st_dev, value.st_ino) != expected:
        raise _run_authority_failure()


def _open_run_authority_root(repository_root: Path) -> int:
    try:
        fd = os.open(
            repository_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError:
        raise _run_authority_failure() from None
    try:
        _run_authority_directory_identity(fd)
    except Phase8GovernanceError:
        os.close(fd)
        raise
    return fd


def _open_run_authority_directory(parent_fd: int, name: str) -> int:
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError:
        raise _run_authority_failure() from None
    try:
        identity = _run_authority_directory_identity(fd)
        _require_run_authority_entry_identity(
            parent_fd,
            name,
            identity,
            directory=True,
        )
    except Phase8GovernanceError:
        os.close(fd)
        raise
    return fd


def _ensure_run_authority_directory(parent_fd: int, name: str) -> int:
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    except OSError:
        raise _run_authority_failure() from None
    child_fd = _open_run_authority_directory(parent_fd, name)
    if created:
        try:
            os.fsync(child_fd)
            _require_run_authority_entry_identity(
                parent_fd,
                name,
                _run_authority_directory_identity(child_fd),
                directory=True,
            )
            os.fsync(parent_fd)
        except OSError:
            os.close(child_fd)
            raise _run_authority_failure() from None
        except Phase8GovernanceError:
            os.close(child_fd)
            raise
    return child_fd


def _open_run_authority_directories(
    repository_root: Path,
    run_id: str,
    *,
    create: bool,
) -> tuple[int, int, int, int]:
    root_fd = _open_run_authority_root(repository_root)
    opened = [root_fd]
    try:
        experiments_fd = _open_run_authority_directory(root_fd, "experiments")
        opened.append(experiments_fd)
        operation = _ensure_run_authority_directory if create else _open_run_authority_directory
        run_fd = operation(experiments_fd, run_id)
        opened.append(run_fd)
        artifacts_fd = operation(run_fd, "artifacts")
        opened.append(artifacts_fd)
        return root_fd, experiments_fd, run_fd, artifacts_fd
    except BaseException:
        for fd in reversed(opened):
            os.close(fd)
        raise


def _require_run_authority_directory_chain(
    directories: tuple[int, int, int, int],
    run_id: str,
) -> None:
    root_fd, experiments_fd, run_fd, artifacts_fd = directories
    relationships = (
        (root_fd, "experiments", experiments_fd),
        (experiments_fd, run_id, run_fd),
        (run_fd, "artifacts", artifacts_fd),
    )
    for parent_fd, name, child_fd in relationships:
        _require_run_authority_entry_identity(
            parent_fd,
            name,
            _run_authority_directory_identity(child_fd),
            directory=True,
        )


def _read_run_authority_evidence_fd(artifacts_fd: int) -> bytes:
    try:
        fd = os.open(
            _RUN_AUTHORITY_EVIDENCE_NAME,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=artifacts_fd,
        )
    except OSError:
        raise _run_authority_failure() from None
    try:
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 4_096:
                raise _run_authority_failure()
            identity = metadata.st_dev, metadata.st_ino
            _require_run_authority_entry_identity(
                artifacts_fd,
                _RUN_AUTHORITY_EVIDENCE_NAME,
                identity,
                directory=False,
            )
            content = os.read(fd, metadata.st_size + 1)
            if len(content) != metadata.st_size or os.read(fd, 1) != b"":
                raise _run_authority_failure()
            current = os.fstat(fd)
            if (
                not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
                or current.st_size != metadata.st_size
            ):
                raise _run_authority_failure()
            _require_run_authority_entry_identity(
                artifacts_fd,
                _RUN_AUTHORITY_EVIDENCE_NAME,
                identity,
                directory=False,
            )
            return content
        except OSError:
            raise _run_authority_failure() from None
    finally:
        os.close(fd)


def _unlink_run_authority_evidence_if_identity(
    artifacts_fd: int,
    expected: tuple[int, int] | None,
) -> None:
    if expected is None:
        return
    try:
        value = os.stat(
            _RUN_AUTHORITY_EVIDENCE_NAME,
            dir_fd=artifacts_fd,
            follow_symlinks=False,
        )
        if (value.st_dev, value.st_ino) == expected:
            os.unlink(_RUN_AUTHORITY_EVIDENCE_NAME, dir_fd=artifacts_fd)
    except OSError:
        pass


def _publish_run_authority_evidence(
    repository_root: Path,
    run_id: str,
    code_commit: str,
    pre_registration_commit: str,
    planned_record_sha256: str,
) -> None:
    if (
        not RUN_ID_PATTERN.fullmatch(run_id)
        or code_commit != PHASE8_IMPLEMENTATION_COMMIT
        or not GIT_COMMIT_PATTERN.fullmatch(pre_registration_commit)
        or pre_registration_commit == code_commit
        or not SHA256_PATTERN.fullmatch(planned_record_sha256)
    ):
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
    expected = _run_authority_evidence_bytes(
        run_id,
        code_commit,
        pre_registration_commit,
        planned_record_sha256,
    )
    directories: tuple[int, int, int, int] | None = None
    evidence_identity: tuple[int, int] | None = None
    try:
        directories = _open_run_authority_directories(
            repository_root,
            run_id,
            create=True,
        )
        _require_run_authority_directory_chain(directories, run_id)
        artifacts_fd = directories[-1]
        try:
            fd = os.open(
                _RUN_AUTHORITY_EVIDENCE_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=artifacts_fd,
            )
        except FileExistsError:
            if _read_run_authority_evidence_fd(artifacts_fd) != expected:
                raise _run_authority_failure()
            _require_run_authority_directory_chain(directories, run_id)
            return
        except OSError:
            raise _run_authority_failure() from None
        try:
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise _run_authority_failure()
                evidence_identity = metadata.st_dev, metadata.st_ino
                _require_run_authority_entry_identity(
                    artifacts_fd,
                    _RUN_AUTHORITY_EVIDENCE_NAME,
                    evidence_identity,
                    directory=False,
                )
                if os.write(fd, expected) != len(expected):
                    raise OSError("short run-authority write")
                os.fsync(fd)
                _require_run_authority_entry_identity(
                    artifacts_fd,
                    _RUN_AUTHORITY_EVIDENCE_NAME,
                    evidence_identity,
                    directory=False,
                )
                _require_run_authority_directory_chain(directories, run_id)
                os.fsync(artifacts_fd)
            except OSError:
                raise _run_authority_failure() from None
        finally:
            os.close(fd)
    except BaseException:
        if directories is not None:
            _unlink_run_authority_evidence_if_identity(
                directories[-1],
                evidence_identity,
            )
        raise
    finally:
        if directories is not None:
            for directory_fd in reversed(directories):
                os.close(directory_fd)


def _require_run_authority_evidence(
    repository_root: Path,
    run_id: str,
    code_commit: str,
    pre_registration_commit: str,
    planned_record_sha256: str,
) -> None:
    expected = _run_authority_evidence_bytes(
        run_id,
        code_commit,
        pre_registration_commit,
        planned_record_sha256,
    )
    directories: tuple[int, int, int, int] | None = None
    try:
        directories = _open_run_authority_directories(
            repository_root,
            run_id,
            create=False,
        )
        _require_run_authority_directory_chain(directories, run_id)
        observed = _read_run_authority_evidence_fd(directories[-1])
        _require_run_authority_directory_chain(directories, run_id)
        if observed != expected:
            raise _run_authority_failure()
    finally:
        if directories is not None:
            for directory_fd in reversed(directories):
                os.close(directory_fd)


def _validate_live_repository(
    repository_root: Path,
    code_commit: str,
    *,
    run_id: str,
    configuration: Phase8Configuration,
    require_run_evidence: bool = False,
) -> str:
    if not isinstance(repository_root, Path):
        raise Phase8TypeError("phase8.type.argument", field="repository_root")
    if type(code_commit) is not str:
        raise Phase8TypeError("phase8.type.argument", field="code_commit")
    if type(run_id) is not str:
        raise Phase8TypeError("phase8.type.argument", field="run_id")
    if type(configuration) is not Phase8Configuration:
        raise Phase8TypeError("phase8.type.record", field="configuration")
    if type(require_run_evidence) is not bool:
        raise Phase8TypeError("phase8.type.argument", field="require_run_evidence")
    if (
        not GIT_COMMIT_PATTERN.fullmatch(code_commit)
        or code_commit != PHASE8_IMPLEMENTATION_COMMIT
    ):
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="implementation_authority",
        )
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
    try:
        root_lstat = repository_root.lstat()
        resolved = repository_root.resolve(strict=True)
    except OSError:
        raise Phase8GovernanceError("phase8.governance.repository", field="repository") from None
    if stat.S_ISLNK(root_lstat.st_mode) or resolved != repository_root:
        raise Phase8GovernanceError("phase8.governance.repository", field="repository")
    if _git(repository_root, "rev-parse", "--show-toplevel") != str(repository_root):
        raise Phase8GovernanceError("phase8.governance.repository", field="root")
    if _git(repository_root, "branch", "--show-current") != "main":
        raise Phase8GovernanceError("phase8.governance.repository", field="branch")
    head = _git_authority_fact(
        repository_root,
        "pre_registration_commit",
        "rev-parse",
        "HEAD",
    )
    origin = _git_authority_fact(
        repository_root,
        "pre_registration_commit",
        "rev-parse",
        "origin/main",
    )
    if (
        not GIT_COMMIT_PATTERN.fullmatch(head)
        or not GIT_COMMIT_PATTERN.fullmatch(origin)
        or head != origin
    ):
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
    if _git(
        repository_root,
        "status",
        "--porcelain=v2",
        "--untracked-files=all",
    ) != "":
        raise Phase8GovernanceError("phase8.governance.repository", field="clean")

    _require_ancestor(
        repository_root,
        PHASE8_IMPLEMENTATION_COMMIT,
        head,
        field="implementation_authority",
    )
    for ancestor in (
        PHASE7_CLOSURE_COMMIT,
        PHASE7_CONTRACT_COMMIT,
        PHASE7_IMPLEMENTATION_COMMIT,
        PHASE8_ORIGINAL_CONTRACT_COMMIT,
        PHASE8_CONTRACT_COMMIT,
    ):
        _require_ancestor(repository_root, ancestor, head, field="ancestor")

    parent_record = _git_authority_fact(
        repository_root,
        "pre_registration_commit",
        "rev-list",
        "--parents",
        "-n",
        "1",
        head,
    )
    parent_parts = parent_record.split()
    if (
        len(parent_parts) != 2
        or parent_parts[0] != head
        or not GIT_COMMIT_PATTERN.fullmatch(parent_parts[1])
    ):
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
    implementation_authority = parent_parts[1]
    _require_ancestor(
        repository_root,
        PHASE8_CONTRACT_COMMIT,
        implementation_authority,
        field="implementation_authority",
    )
    if _git_authority_fact(
        repository_root,
        "pre_registration_commit",
        "diff",
        "--name-only",
        implementation_authority,
        head,
    ).splitlines() != ["EXPERIMENT_LOG.md"]:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
    changed_after_anchor = frozenset(
        _git_authority_fact(
            repository_root,
            "implementation_authority",
            "diff",
            "--name-only",
            PHASE8_IMPLEMENTATION_COMMIT,
            implementation_authority,
        ).splitlines()
    )
    if not changed_after_anchor.issubset(_POST_IMPLEMENTATION_ALLOWED_PATHS):
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="implementation_authority",
        )

    try:
        working_log = (repository_root / "EXPERIMENT_LOG.md").read_bytes()
    except OSError:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        ) from None
    committed_log = _git_blob_bytes(repository_root, head, "EXPERIMENT_LOG.md")
    parent_log = _git_blob_bytes(
        repository_root,
        implementation_authority,
        "EXPERIMENT_LOG.md",
    )
    planned_heading = f"### {run_id} — Phase 8".encode("utf-8")
    if working_log != committed_log or planned_heading in parent_log:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
    from sebgpt.training.phase8_experiment import _planned_run_record

    observed_run_id, _ = _planned_run_record(
        repository_root,
        code_commit=PHASE8_IMPLEMENTATION_COMMIT,
        configuration=configuration,
    )
    if observed_run_id != run_id:
        raise Phase8GovernanceError(
            "phase8.governance.repository",
            field="pre_registration_commit",
        )
    if require_run_evidence:
        _require_run_authority_evidence(
            repository_root,
            run_id,
            PHASE8_IMPLEMENTATION_COMMIT,
            head,
            hashlib.sha256(committed_log).hexdigest(),
        )

    identities = (
        ("docs/MINI_GPT_SPEC.md", MINI_GPT_SPEC_SHA256),
        ("src/sebgpt/model/__init__.py", MINI_GPT_EXPORT_SHA256),
        ("src/sebgpt/model/mini_gpt.py", MINI_GPT_SOURCE_SHA256),
        ("tests/test_mini_gpt.py", MINI_GPT_TEST_SHA256),
        ("docs/TRAINING_CHECKPOINTING_SPEC.md", PHASE8_SPEC_SHA256),
        (
            "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
            EXPECTED_ARTIFACT_SHA256,
        ),
        ("docs/data/shakespeare-eight-play-manifest.json", PHASE1_MANIFEST_SHA256),
        (
            "data/processed/shakespeare-eight-play/processing-manifest.json",
            PROCESSING_MANIFEST_SHA256,
        ),
    )
    for relative_path, digest in identities:
        if _sha256_file(repository_root / relative_path) != digest:
            raise Phase8GovernanceError(
                "phase8.governance.repository",
                field=(
                    "implementation_authority"
                    if relative_path.startswith(("docs/MINI_GPT", "src/sebgpt/model", "tests/test_mini_gpt", "docs/TRAINING"))
                    else relative_path
                ),
            )
    for relative_path in _PHASE8_SOURCE_TEST_PATHS:
        try:
            live_bytes = (repository_root / relative_path).read_bytes()
        except OSError:
            raise Phase8GovernanceError(
                "phase8.governance.repository",
                field="implementation_authority",
            ) from None
        try:
            accepted_bytes = _git_blob_bytes(
                repository_root,
                implementation_authority,
                relative_path,
            )
        except Phase8GovernanceError as error:
            raise Phase8GovernanceError(
                "phase8.governance.repository",
                field="implementation_authority",
            ) from error
        if live_bytes != accepted_bytes:
            raise Phase8GovernanceError(
                "phase8.governance.repository",
                field="implementation_authority",
            )
    return head


def _runtime_mapping(value: Phase8RuntimeIdentity) -> dict[str, object]:
    return {item.name: getattr(value, item.name) for item in fields(value)}


def _configuration_mapping(value: Phase8Configuration) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in fields(value):
        observed = getattr(value, item.name)
        result[item.name] = (
            _runtime_mapping(observed)
            if item.name == "runtime" and isinstance(observed, Phase8RuntimeIdentity)
            else observed
        )
    return result


def _evaluation_mapping(value: Phase8Evaluation) -> dict[str, object]:
    return {
        "split": value.split,
        "loss": value.loss,
        "loss_hex": value.loss.hex(),
        "target_count": value.target_count,
        "window_count": value.window_count,
    }


def _metric_mapping(value: Phase8MetricState) -> dict[str, object]:
    return {
        "initialized_training": _evaluation_mapping(value.initialized_training),
        "initialized_validation": _evaluation_mapping(value.initialized_validation),
        "epoch_training": tuple(
            _evaluation_mapping(item) for item in value.epoch_training
        ),
        "epoch_validation": tuple(
            _evaluation_mapping(item) for item in value.epoch_validation
        ),
        "current_training": _evaluation_mapping(value.current_training),
        "current_validation": _evaluation_mapping(value.current_validation),
        "best_validation_loss": value.best_validation_loss,
        "best_validation_loss_hex": value.best_validation_loss.hex(),
        "best_validation_epoch": value.best_validation_epoch,
        "best_logical_id": value.best_logical_id,
        "best_comparison": "strict_lower",
    }


def _progress_mapping(value: Phase8Progress, metrics: Phase8MetricState) -> dict[str, object]:
    example_count = metrics.current_training.window_count
    target_count = metrics.current_training.target_count
    updates = math.ceil(example_count / 8)
    return {
        "completed_epochs": value.completed_epochs,
        "next_epoch": value.next_epoch,
        "next_example_offset": value.next_example_offset,
        "optimizer_updates": value.optimizer_updates,
        "examples_processed": value.examples_processed,
        "targets_processed": value.targets_processed,
        "per_epoch_optimizer_updates": (updates,) * value.completed_epochs,
        "per_epoch_example_counts": (example_count,) * value.completed_epochs,
        "per_epoch_target_counts": (target_count,) * value.completed_epochs,
        "next_ordering_action": "draw_next_epoch_permutation",
    }


def _dropout_state_mapping(model: MiniGPT) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    position = 0
    for block in model.blocks:
        result[_DROPOUT_NAMES[position]] = (
            block.attention_dropout._generator.get_state().clone()
        )
        result[_DROPOUT_NAMES[position + 1]] = (
            block.feed_forward_dropout._generator.get_state().clone()
        )
        position += 2
    return result


def _dataset_authority(repository_root: Path) -> dict[str, object]:
    phase1_path = repository_root / "docs/data/shakespeare-eight-play-manifest.json"
    processing_path = (
        repository_root
        / "data/processed/shakespeare-eight-play/processing-manifest.json"
    )
    try:
        phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
        processing = json.loads(processing_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase8CheckpointError(
            "phase8.checkpoint.authority",
            field="dataset",
        ) from error
    result_records = phase1["processing"]["per_work_results"]
    permitted = [
        value
        for value in result_records
        if value.get("split") in ("train", "validation")
    ]
    permitted.sort(key=lambda value: value["manifest_order"])
    work_values = tuple(
        {
            "manifest_order": value["manifest_order"],
            "work_id": value["work_id"],
            "split": value["split"],
            "processed_sha256": value["processed"]["sha256"],
        }
        for value in permitted
    )
    if len(work_values) != 7:
        raise Phase8CheckpointError(
            "phase8.checkpoint.authority",
            field="dataset_works",
        )
    return {
        "dataset_id": phase1["dataset_id"],
        "manifest_sha256": PHASE1_MANIFEST_SHA256,
        "processing_manifest_sha256": PROCESSING_MANIFEST_SHA256,
        "training_works": work_values[:6],
        "validation_works": work_values[6:],
    }


def _authority_mapping(
    configuration: Phase8Configuration,
    repository_root: Path,
) -> dict[str, object]:
    runtime = _runtime_mapping(configuration.runtime)
    return {
        "phase7_closure_commit": PHASE7_CLOSURE_COMMIT,
        "phase7_contract_commit": PHASE7_CONTRACT_COMMIT,
        "phase7_implementation_commit": PHASE7_IMPLEMENTATION_COMMIT,
        "mini_gpt_spec_sha256": MINI_GPT_SPEC_SHA256,
        "mini_gpt_source_sha256": MINI_GPT_SOURCE_SHA256,
        "mini_gpt_test_sha256": MINI_GPT_TEST_SHA256,
        "phase8_contract_commit": PHASE8_CONTRACT_COMMIT,
        "phase8_spec_sha256": PHASE8_SPEC_SHA256,
        "requirements_lock_sha256": runtime["requirements_lock_sha256"],
        "tokenizer": {
            "implementation_commit": TOKENIZER_IMPLEMENTATION_COMMIT,
            "tokenizer_id": "shakespeare-code-point-v1",
            "schema_version": 1,
            "vocabulary_size": 81,
            "artifact_path": VOCABULARY_RELATIVE_PATH.as_posix(),
            "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
        },
        "dataset": _dataset_authority(repository_root),
        "model": {
            "model_width": MODEL_WIDTH,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "number_of_blocks": NUMBER_OF_BLOCKS,
            "number_of_heads": NUMBER_OF_HEADS,
            "head_width": HEAD_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "layer_norm_epsilon": LAYER_NORM_EPSILON,
            "dropout_probability": DROPOUT_PROBABILITY,
            "embedding_seed": EMBEDDING_INITIALIZATION_SEED,
            "block_parameter_seeds": BLOCK_PARAMETER_INITIALIZATION_SEEDS,
            "output_head_seed": OUTPUT_HEAD_INITIALIZATION_SEED,
            "parameter_device": str(PARAMETER_DEVICE),
            "parameter_dtype": str(PARAMETER_DTYPE),
            "trainable_tensor_count": 90,
            "parameter_count": 63_825,
        },
        "runtime": runtime,
        "sealed_test_access": "none",
    }


def _validate_training_state(
    state: Phase8TrainingState,
    *,
    require_live_global: bool = True,
) -> None:
    if type(state) is not Phase8TrainingState:
        raise Phase8TypeError("phase8.type.record", field="state")
    if not RUN_ID_PATTERN.fullmatch(state.run_id):
        raise Phase8ContractError("phase8.contract.lifecycle", field="run_id")
    if (
        not GIT_COMMIT_PATTERN.fullmatch(state.code_commit)
        or state.code_commit != PHASE8_IMPLEMENTATION_COMMIT
    ):
        raise Phase8ContractError("phase8.contract.lifecycle", field="code_commit")
    if type(state.configuration) is not Phase8Configuration:
        raise Phase8TypeError("phase8.type.record", field="configuration")
    if state.configuration != _fixed_configuration(_validate_runtime()):
        raise Phase8ContractError("phase8.contract.configuration")
    if type(state.progress) is not Phase8Progress:
        raise Phase8TypeError("phase8.type.record", field="progress")
    if type(state.metrics) is not Phase8MetricState:
        raise Phase8TypeError("phase8.type.record", field="metrics")
    progress = state.progress
    metrics = state.metrics
    if (
        not 1 <= progress.completed_epochs <= 10
        or progress.next_epoch != progress.completed_epochs + 1
        or progress.next_example_offset != 0
        or progress.optimizer_updates <= 0
        or progress.examples_processed <= 0
        or progress.targets_processed <= 0
    ):
        raise Phase8ContractError("phase8.contract.progress")
    if (
        len(metrics.epoch_training) != progress.completed_epochs
        or len(metrics.epoch_validation) != progress.completed_epochs
        or metrics.current_training != metrics.epoch_training[-1]
        or metrics.current_validation != metrics.epoch_validation[-1]
        or metrics.initialized_training.split != "train"
        or metrics.initialized_validation.split != "validation"
        or metrics.current_training.split != "train"
        or metrics.current_validation.split != "validation"
    ):
        raise Phase8ContractError("phase8.contract.metric")
    validation_losses = tuple(item.loss for item in metrics.epoch_validation)
    if any(not math.isfinite(value) for value in validation_losses):
        raise Phase8NumericalError("phase8.numerical.metric")
    best_loss = min(validation_losses)
    best_epoch = validation_losses.index(best_loss) + 1
    if (
        metrics.best_validation_loss != best_loss
        or metrics.best_validation_epoch != best_epoch
        or metrics.best_logical_id != f"epoch-{best_epoch:04d}"
    ):
        raise Phase8ContractError("phase8.contract.metric", field="best")
    examples_per_epoch = metrics.current_training.window_count
    targets_per_epoch = metrics.current_training.target_count
    updates_per_epoch = math.ceil(examples_per_epoch / 8)
    if (
        progress.examples_processed != examples_per_epoch * progress.completed_epochs
        or progress.targets_processed != targets_per_epoch * progress.completed_epochs
        or progress.optimizer_updates != updates_per_epoch * progress.completed_epochs
    ):
        raise Phase8ContractError("phase8.contract.progress", field="counts")
    if not model_mode_is_training(state.model):
        raise Phase8ContractError("phase8.contract.mode")
    if any(parameter.grad is not None for parameter in state.model.parameters()):
        raise Phase8ContractError("phase8.contract.gradient")
    _validate_optimizer(
        state.model,
        state.optimizer,
        require_state=True,
        expected_updates=progress.optimizer_updates,
    )
    if type(state.order_generator) is not torch.Generator:
        raise Phase8TypeError("phase8.type.generator")
    if state.order_generator.device != torch.device("cpu"):
        raise Phase8ContractError("phase8.contract.order")
    if state.order_generator.initial_seed() != 8001:
        raise Phase8ContractError("phase8.contract.order", field="seed")
    if not isinstance(state.global_cpu_rng_state, torch.Tensor):
        raise Phase8TypeError("phase8.type.tensor", field="global_cpu_rng_state")
    if (
        state.global_cpu_rng_state.dim() != 1
        or state.global_cpu_rng_state.device != torch.device("cpu")
        or state.global_cpu_rng_state.dtype is not torch.uint8
        or (
            require_live_global
            and not torch.equal(state.global_cpu_rng_state, torch.get_rng_state())
        )
    ):
        raise Phase8ContractError("phase8.contract.lifecycle", field="global_rng")
    lineage = state.resume_checkpoint_sha256s
    if type(lineage) is not tuple or any(
        type(item) is not str or not SHA256_PATTERN.fullmatch(item)
        for item in lineage
    ):
        raise Phase8ContractError("phase8.contract.lifecycle", field="lineage")
    if (not lineage and state.resumed_from_checkpoint_sha256 is not None) or (
        lineage and state.resumed_from_checkpoint_sha256 != lineage[-1]
    ):
        raise Phase8ContractError("phase8.contract.lifecycle", field="lineage")


def model_mode_is_training(model: MiniGPT) -> bool:
    return all(module.training is True for _, module in model.named_modules())


def _payload(
    state: Phase8TrainingState,
    repository_root: Path,
) -> dict[str, object]:
    _validate_training_state(state)
    progress = _progress_mapping(state.progress, state.metrics)
    logical_id = f"epoch-{state.progress.completed_epochs:04d}"
    result = {
        "schema_version": 1,
        "run": {
            "run_id": state.run_id,
            "logical_id": logical_id,
            "completed_epoch": state.progress.completed_epochs,
            "code_commit": state.code_commit,
        },
        "authority": _authority_mapping(state.configuration, repository_root),
        "configuration": _configuration_mapping(state.configuration),
        "model_state": state.model.state_dict(),
        "optimizer_state": state.optimizer.state_dict(),
        "progress": progress,
        "random_state": {
            "order_generator": state.order_generator.get_state().clone(),
            "global_cpu": state.global_cpu_rng_state.clone(),
            "dropout_generators": _dropout_state_mapping(state.model),
        },
        "mode_state": {
            "training": True,
            "named_modules": tuple(
                (name, module.training) for name, module in state.model.named_modules()
            ),
        },
        "metric_state": _metric_mapping(state.metrics),
        "resume_lineage": {
            "root_run_id": state.run_id,
            "resumed_from_checkpoint_sha256": state.resumed_from_checkpoint_sha256,
            "resume_count": len(state.resume_checkpoint_sha256s),
            "checkpoint_sha256s": state.resume_checkpoint_sha256s,
        },
    }
    _validate_payload(result)
    return result


def _canonical_catalog_bytes(value: dict[str, object]) -> bytes:
    content = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(content) > MAXIMUM_CATALOG_BYTES:
        raise Phase8CheckpointError("phase8.checkpoint.size", field="catalog")
    return content


def _reference_mapping(
    *,
    run_id: str,
    role: str,
    logical_id: str,
    epoch: int,
    validation_loss_hex: str,
    digest: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "role": role,
        "logical_id": logical_id,
        "epoch": epoch,
        "validation_loss_hex": validation_loss_hex,
        "sha256": digest,
        "relative_path": f"objects/{digest}.pt",
    }


def _reference_from_mapping(value: dict[str, object]) -> Phase8CheckpointReference:
    return Phase8CheckpointReference(
        run_id=value["run_id"],  # type: ignore[arg-type]
        role=value["role"],  # type: ignore[arg-type]
        logical_id=value["logical_id"],  # type: ignore[arg-type]
        epoch=value["epoch"],  # type: ignore[arg-type]
        validation_loss_hex=value["validation_loss_hex"],  # type: ignore[arg-type]
        sha256=value["sha256"],  # type: ignore[arg-type]
        relative_path=value["relative_path"],  # type: ignore[arg-type]
    )


def _open_directory(path: str | Path, *, dir_fd: int | None = None) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(path, flags, dir_fd=dir_fd)
    except OSError as error:
        raise Phase8CheckpointError("phase8.checkpoint.path") from error


def _ensure_directory(parent_fd: int, name: str) -> int:
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    except OSError as error:
        raise Phase8PublicationError("phase8.publication.bootstrap") from error
    child_fd = _open_directory(name, dir_fd=parent_fd)
    if created:
        try:
            os.fsync(child_fd)
            os.fsync(parent_fd)
        except OSError as error:
            os.close(child_fd)
            raise Phase8PublicationError("phase8.publication.bootstrap") from error
    return child_fd


def _read_bounded_fd(fd: int, maximum: int) -> bytes:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        raise Phase8CheckpointError("phase8.checkpoint.size")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = info.st_size
    while remaining:
        value = os.read(fd, min(remaining, 1_048_576))
        if not value:
            raise Phase8CheckpointError("phase8.checkpoint.size", field="short_read")
        chunks.append(value)
        remaining -= len(value)
    if os.read(fd, 1) != b"" or os.fstat(fd).st_size != info.st_size:
        raise Phase8CheckpointError("phase8.checkpoint.size", field="changing")
    return b"".join(chunks)


def _read_catalog(run_fd: int) -> dict[str, object] | None:
    try:
        fd = os.open(
            "checkpoint-catalog.json",
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=run_fd,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise Phase8CheckpointError("phase8.checkpoint.catalog") from error
    try:
        content = _read_bounded_fd(fd, MAXIMUM_CATALOG_BYTES)
    finally:
        os.close(fd)
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Phase8CheckpointError("phase8.checkpoint.catalog") from error
    if type(value) is not dict or content != _canonical_catalog_bytes(value):
        raise Phase8CheckpointError("phase8.checkpoint.catalog")
    _validate_catalog(value)
    return value


def _validate_reference(value: object, role: str, run_id: str) -> dict[str, object]:
    expected_keys = (
        "run_id",
        "role",
        "logical_id",
        "epoch",
        "validation_loss_hex",
        "sha256",
        "relative_path",
    )
    if (
        type(value) is not dict
        or len(value) != len(expected_keys)
        or set(value) != set(expected_keys)
    ):
        raise Phase8CheckpointError("phase8.checkpoint.reference")
    result = value
    digest = result["sha256"]
    if (
        type(result["run_id"]) is not str
        or result["run_id"] != run_id
        or type(result["role"]) is not str
        or result["role"] != role
        or type(result["epoch"]) is not int
        or not 1 <= result["epoch"] <= 10
        or type(result["logical_id"]) is not str
        or result["logical_id"] != f"epoch-{result['epoch']:04d}"
        or type(digest) is not str
        or not SHA256_PATTERN.fullmatch(digest)
        or type(result["relative_path"]) is not str
        or result["relative_path"] != f"objects/{digest}.pt"
        or type(result["validation_loss_hex"]) is not str
    ):
        raise Phase8CheckpointError("phase8.checkpoint.reference")
    try:
        loss = float.fromhex(result["validation_loss_hex"])
    except ValueError:
        raise Phase8CheckpointError("phase8.checkpoint.reference") from None
    if not math.isfinite(loss) or loss.hex() != result["validation_loss_hex"]:
        raise Phase8CheckpointError("phase8.checkpoint.reference")
    return result


def _validate_catalog(value: dict[str, object]) -> None:
    expected_keys = {"schema_version", "run_id", "latest", "best_validation"}
    if len(value) != len(expected_keys) or set(value) != expected_keys:
        raise Phase8CheckpointError("phase8.checkpoint.catalog")
    run_id = value["run_id"]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(run_id) is not str
        or not RUN_ID_PATTERN.fullmatch(run_id)
    ):
        raise Phase8CheckpointError("phase8.checkpoint.catalog")
    latest = _validate_reference(value["latest"], "latest", run_id)
    best = _validate_reference(value["best_validation"], "best_validation", run_id)
    if best["epoch"] > latest["epoch"]:
        raise Phase8CheckpointError("phase8.checkpoint.catalog")


def _open_checkpoint_directories(
    repository_root: Path,
    run_id: str,
    *,
    create: bool,
    components: tuple[str, ...] | None = None,
) -> tuple[int, ...]:
    if components is None:
        components = ("checkpoints", "phase8", run_id, "objects")
    if (
        type(components) is not tuple
        or len(components) < 2
        or components[-1] != "objects"
        or any(
            type(name) is not str
            or not name
            or name in (".", "..")
            or "/" in name
            for name in components
        )
    ):
        raise Phase8CheckpointError("phase8.checkpoint.path")
    root_fd = _open_directory(repository_root)
    opened = [root_fd]
    try:
        parent_fd = root_fd
        for position, name in enumerate(components):
            child_fd = (
                _ensure_directory(parent_fd, name)
                if create and position > 0
                else _open_directory(name, dir_fd=parent_fd)
            )
            opened.append(child_fd)
            parent_fd = child_fd
        return tuple(opened)
    except BaseException:
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _close_directories(values: tuple[int, ...]) -> None:
    for fd in reversed(values):
        try:
            os.close(fd)
        except OSError:
            pass


def _directory_identities(
    values: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for fd in values:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise Phase8CheckpointError("phase8.checkpoint.path")
        result.append((info.st_dev, info.st_ino))
    return tuple(result)


def _require_directory_identities(
    values: tuple[int, ...],
    expected: tuple[tuple[int, int], ...],
) -> None:
    if _directory_identities(values) != expected:
        raise Phase8CheckpointError("phase8.checkpoint.path", field="identity")


def _entry_identity(parent_fd: int, name: str) -> tuple[int, int]:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise Phase8CheckpointError(
            "phase8.checkpoint.path",
            field="temporary_identity",
        ) from error
    if not stat.S_ISREG(info.st_mode):
        raise Phase8CheckpointError(
            "phase8.checkpoint.path",
            field="temporary_type",
        )
    return info.st_dev, info.st_ino


def _unlink_if_identity(
    parent_fd: int,
    name: str,
    expected: tuple[int, int] | None,
) -> bool:
    if expected is None:
        return False
    try:
        observed = _entry_identity(parent_fd, name)
    except Phase8CheckpointError:
        return False
    if observed != expected:
        return False
    try:
        os.unlink(name, dir_fd=parent_fd)
    except OSError:
        return False
    return True


def _publish_payload(
    payload: dict[str, object],
    repository_root: Path,
    *,
    components: tuple[str, ...] | None = None,
) -> Phase8CheckpointReference:
    run = payload["run"]
    metrics = payload["metric_state"]
    assert isinstance(run, dict) and isinstance(metrics, dict)
    run_id = run["run_id"]
    logical_id = run["logical_id"]
    epoch = run["completed_epoch"]
    validation_loss_hex = metrics["current_validation"]["loss_hex"]
    assert isinstance(run_id, str)
    assert isinstance(logical_id, str)
    assert isinstance(epoch, int)
    assert isinstance(validation_loss_hex, str)
    directories = _open_checkpoint_directories(
        repository_root,
        run_id,
        create=True,
        components=components,
    )
    directory_identities = _directory_identities(directories)
    run_fd, objects_fd = directories[-2], directories[-1]
    temporary_name = f".checkpoint-{logical_id}-{secrets.token_hex(16)}.tmp"
    object_temporary_present = False
    object_temporary_identity: tuple[int, int] | None = None
    catalog_temp: str | None = None
    catalog_temporary_identity: tuple[int, int] | None = None
    catalog_replaced = False
    try:
        previous = _read_catalog(run_fd)
        previous_best_content: bytes | None = None
        previous_best_payload: dict[str, object] | None = None
        if previous is not None:
            previous_latest = _validate_reference(
                previous["latest"],
                "latest",
                run_id,
            )
            previous_best = _validate_reference(
                previous["best_validation"],
                "best_validation",
                run_id,
            )
            previous_latest_content, previous_latest_payload = _load_object(
                objects_fd,
                previous_latest,
            )
            _validate_payload(previous_latest_payload)
            if previous_best["sha256"] == previous_latest["sha256"]:
                previous_best_content = previous_latest_content
                previous_best_payload = previous_latest_payload
            else:
                previous_best_content, previous_best_payload = _load_object(
                    objects_fd,
                    previous_best,
                )
                _validate_payload(previous_best_payload)
            _cross_validate_catalog_graph(
                previous,
                previous_latest,
                previous_latest_payload,
                hashlib.sha256(previous_latest_content).hexdigest(),
                previous_best,
                previous_best_payload,
                hashlib.sha256(previous_best_content).hexdigest(),
            )
        _require_directory_identities(directories, directory_identities)
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=run_fd,
            )
        except OSError as error:
            raise Phase8PublicationError("phase8.publication.object") from error
        object_temporary_present = True
        info = os.fstat(temporary_fd)
        if not stat.S_ISREG(info.st_mode):
            raise Phase8PublicationError("phase8.publication.object")
        object_temporary_identity = (info.st_dev, info.st_ino)
        try:
            with os.fdopen(temporary_fd, "w+b", closefd=False) as stream:
                torch.save(payload, stream)
                stream.flush()
            os.fsync(temporary_fd)
            content = _read_bounded_fd(temporary_fd, MAXIMUM_OBJECT_BYTES)
        finally:
            os.close(temporary_fd)
        digest = hashlib.sha256(content).hexdigest()
        object_name = f"{digest}.pt"
        _require_directory_identities(directories, directory_identities)
        promoted_new = False
        try:
            os.link(
                temporary_name,
                object_name,
                src_dir_fd=run_fd,
                dst_dir_fd=objects_fd,
                follow_symlinks=False,
            )
            promoted_new = True
        except FileExistsError:
            existing_fd = os.open(
                object_name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=objects_fd,
            )
            try:
                if _read_bounded_fd(existing_fd, MAXIMUM_OBJECT_BYTES) != content:
                    raise Phase8PublicationError("phase8.publication.object")
            finally:
                os.close(existing_fd)
        except OSError as error:
            raise Phase8PublicationError("phase8.publication.object") from error
        _require_directory_identities(directories, directory_identities)
        if promoted_new and _entry_identity(objects_fd, object_name) != object_temporary_identity:
            raise Phase8PublicationError("phase8.publication.object")
        if not _unlink_if_identity(
            run_fd,
            temporary_name,
            object_temporary_identity,
        ):
            raise Phase8PublicationError("phase8.publication.object")
        object_temporary_present = False
        _require_directory_identities(directories, directory_identities)
        os.fsync(objects_fd)
        _require_directory_identities(directories, directory_identities)
        latest = _reference_mapping(
            run_id=run_id,
            role="latest",
            logical_id=logical_id,
            epoch=epoch,
            validation_loss_hex=validation_loss_hex,
            digest=digest,
        )
        if previous is None or float.fromhex(validation_loss_hex) < float.fromhex(
            previous["best_validation"]["validation_loss_hex"]  # type: ignore[index]
        ):
            best = dict(latest)
            best["role"] = "best_validation"
            best_payload_for_catalog = payload
            best_content_for_catalog = content
        else:
            best = previous["best_validation"]
            assert previous_best_payload is not None
            assert previous_best_content is not None
            best_payload_for_catalog = previous_best_payload
            best_content_for_catalog = previous_best_content
        catalog = {
            "schema_version": 1,
            "run_id": run_id,
            "latest": latest,
            "best_validation": best,
        }
        _validate_catalog(catalog)
        _cross_validate_catalog_graph(
            catalog,
            latest,
            payload,
            digest,
            best,
            best_payload_for_catalog,
            hashlib.sha256(best_content_for_catalog).hexdigest(),
        )
        catalog_content = _canonical_catalog_bytes(catalog)
        _require_directory_identities(directories, directory_identities)
        catalog_temp = f".catalog-{logical_id}-{secrets.token_hex(16)}.tmp"
        _require_directory_identities(directories, directory_identities)
        catalog_fd = os.open(
            catalog_temp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=run_fd,
        )
        catalog_info = os.fstat(catalog_fd)
        if not stat.S_ISREG(catalog_info.st_mode):
            raise Phase8PublicationError("phase8.publication.catalog")
        catalog_temporary_identity = (catalog_info.st_dev, catalog_info.st_ino)
        try:
            written = os.write(catalog_fd, catalog_content)
            if written != len(catalog_content):
                raise Phase8PublicationError("phase8.publication.catalog")
            os.fsync(catalog_fd)
        finally:
            os.close(catalog_fd)
        _require_directory_identities(directories, directory_identities)
        if _entry_identity(run_fd, catalog_temp) != catalog_temporary_identity:
            raise Phase8PublicationError("phase8.publication.catalog")
        try:
            os.replace(
                catalog_temp,
                "checkpoint-catalog.json",
                src_dir_fd=run_fd,
                dst_dir_fd=run_fd,
            )
        except OSError as error:
            raise Phase8PublicationError("phase8.publication.catalog") from error
        catalog_replaced = True
        catalog_temp = None
        if (
            _entry_identity(run_fd, "checkpoint-catalog.json")
            != catalog_temporary_identity
        ):
            raise Phase8PublicationError("phase8.publication.uncertain")
        try:
            os.fsync(run_fd)
        except OSError as error:
            raise Phase8PublicationError("phase8.publication.uncertain") from error
        reference = _reference_from_mapping(latest)
        try:
            _require_directory_identities(directories, directory_identities)
            observed_catalog = _read_catalog(run_fd)
            if observed_catalog != catalog:
                raise Phase8CheckpointError(
                    "phase8.checkpoint.cross_field",
                    field="published_catalog",
                )
            _validate_catalog_graph_fds(
                run_fd,
                objects_fd,
                observed_catalog,
            )
        except BaseException as error:
            raise Phase8PublicationError(
                "phase8.publication.run_fsync",
                state="committed",
                run_id=reference.run_id,
                role=reference.role,
                logical_id=reference.logical_id,
                epoch=reference.epoch,
                validation_loss_hex=reference.validation_loss_hex,
                sha256=reference.sha256,
                relative_path=reference.relative_path,
            ) from error
        return reference
    finally:
        if object_temporary_present:
            _unlink_if_identity(
                run_fd,
                temporary_name,
                object_temporary_identity,
            )
        if catalog_temp is not None and not catalog_replaced:
            _unlink_if_identity(
                run_fd,
                catalog_temp,
                catalog_temporary_identity,
            )
        _close_directories(directories)


def save_phase8_checkpoint(
    state: Phase8TrainingState,
    repository_root: Path,
) -> Phase8CheckpointReference:
    _validate_training_state(state)
    _validate_live_repository(
        repository_root,
        state.code_commit,
        run_id=state.run_id,
        configuration=state.configuration,
        require_run_evidence=True,
    )
    try:
        return _publish_payload(_payload(state, repository_root), repository_root)
    except (
        Phase8TypeError,
        Phase8ContractError,
        Phase8GovernanceError,
        Phase8NumericalError,
        Phase8CheckpointError,
        Phase8PublicationError,
    ):
        raise
    except BaseException as error:
        raise Phase8PublicationError("phase8.publication.uncertain") from error


def _catalog_references(
    repository_root: Path,
    run_id: str,
) -> tuple[Phase8CheckpointReference, Phase8CheckpointReference]:
    directories = _open_checkpoint_directories(repository_root, run_id, create=False)
    try:
        catalog = _read_catalog(directories[-2])
        if catalog is None:
            raise Phase8CheckpointError("phase8.checkpoint.catalog")
        latest = _validate_reference(catalog["latest"], "latest", run_id)
        best = _validate_reference(
            catalog["best_validation"],
            "best_validation",
            run_id,
        )
        return _reference_from_mapping(latest), _reference_from_mapping(best)
    finally:
        _close_directories(directories)


def _load_object(objects_fd: int, reference: dict[str, object]) -> tuple[bytes, dict[str, object]]:
    digest = reference["sha256"]
    assert isinstance(digest, str)
    fd = os.open(
        f"{digest}.pt",
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=objects_fd,
    )
    try:
        content = _read_bounded_fd(fd, MAXIMUM_OBJECT_BYTES)
    finally:
        os.close(fd)
    if hashlib.sha256(content).hexdigest() != digest:
        raise Phase8CheckpointError("phase8.checkpoint.hash")
    try:
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except BaseException as error:
        raise Phase8CheckpointError("phase8.checkpoint.payload") from error
    if type(payload) is not dict:
        raise Phase8CheckpointError("phase8.checkpoint.payload")
    return content, payload


def _model_state_oracle() -> tuple[tuple[str, tuple[int, ...]], ...]:
    result: list[tuple[str, tuple[int, ...]]] = [
        ("representation.token_embeddings", (81, 32)),
        ("representation.position_embeddings", (256, 32)),
    ]
    for block in range(4):
        prefix = f"blocks.{block}"
        result.extend(
            (
                (f"{prefix}.norm1.gamma", (32,)),
                (f"{prefix}.norm1.beta", (32,)),
                (f"{prefix}.attention.output_weight", (32, 32)),
            )
        )
        for head in range(4):
            for projection in ("query", "key", "value"):
                result.append(
                    (
                        f"{prefix}.attention.heads.{head}.{projection}_weight",
                        (32, 8),
                    )
                )
        result.extend(
            (
                (f"{prefix}.norm2.gamma", (32,)),
                (f"{prefix}.norm2.beta", (32,)),
                (f"{prefix}.feed_forward.first_weight", (32, 128)),
                (f"{prefix}.feed_forward.first_bias", (128,)),
                (f"{prefix}.feed_forward.second_weight", (128, 32)),
                (f"{prefix}.feed_forward.second_bias", (32,)),
            )
        )
    result.extend(
        (
            ("final_norm.gamma", (32,)),
            ("final_norm.beta", (32,)),
            ("head.output_weight", (32, 81)),
            ("head.output_bias", (81,)),
        )
    )
    return tuple(result)


_MODEL_STATE_ORACLE = _model_state_oracle()


def _exact_mapping(
    value: object,
    keys: tuple[str, ...],
    *,
    field: str,
) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != keys:
        raise Phase8CheckpointError("phase8.checkpoint.schema", field=field)
    return value


def _exact_digest(value: object, *, length: int = 64) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_authority_schema(value: object) -> None:
    authority = _exact_mapping(
        value,
        (
            "phase7_closure_commit",
            "phase7_contract_commit",
            "phase7_implementation_commit",
            "mini_gpt_spec_sha256",
            "mini_gpt_source_sha256",
            "mini_gpt_test_sha256",
            "phase8_contract_commit",
            "phase8_spec_sha256",
            "requirements_lock_sha256",
            "tokenizer",
            "dataset",
            "model",
            "runtime",
            "sealed_test_access",
        ),
        field="authority",
    )
    expected_commits = {
        "phase7_closure_commit": PHASE7_CLOSURE_COMMIT,
        "phase7_contract_commit": PHASE7_CONTRACT_COMMIT,
        "phase7_implementation_commit": PHASE7_IMPLEMENTATION_COMMIT,
        "phase8_contract_commit": PHASE8_CONTRACT_COMMIT,
    }
    expected_digests = {
        "mini_gpt_spec_sha256": MINI_GPT_SPEC_SHA256,
        "mini_gpt_source_sha256": MINI_GPT_SOURCE_SHA256,
        "mini_gpt_test_sha256": MINI_GPT_TEST_SHA256,
        "phase8_spec_sha256": PHASE8_SPEC_SHA256,
    }
    if (
        any(
            type(authority[name]) is not str or authority[name] != expected
            for name, expected in expected_commits.items()
        )
        or any(
            type(authority[name]) is not str or authority[name] != expected
            for name, expected in expected_digests.items()
        )
        or not _exact_digest(authority["requirements_lock_sha256"])
        or type(authority["sealed_test_access"]) is not str
        or authority["sealed_test_access"] != "none"
    ):
        raise Phase8CheckpointError("phase8.checkpoint.authority")
    tokenizer = _exact_mapping(
        authority["tokenizer"],
        (
            "implementation_commit",
            "tokenizer_id",
            "schema_version",
            "vocabulary_size",
            "artifact_path",
            "artifact_sha256",
        ),
        field="tokenizer",
    )
    if (
        type(tokenizer["implementation_commit"]) is not str
        or tokenizer["implementation_commit"] != TOKENIZER_IMPLEMENTATION_COMMIT
        or type(tokenizer["tokenizer_id"]) is not str
        or tokenizer["tokenizer_id"] != "shakespeare-code-point-v1"
        or type(tokenizer["schema_version"]) is not int
        or tokenizer["schema_version"] != 1
        or type(tokenizer["vocabulary_size"]) is not int
        or tokenizer["vocabulary_size"] != 81
        or type(tokenizer["artifact_path"]) is not str
        or tokenizer["artifact_path"] != VOCABULARY_RELATIVE_PATH.as_posix()
        or type(tokenizer["artifact_sha256"]) is not str
        or tokenizer["artifact_sha256"] != EXPECTED_ARTIFACT_SHA256
    ):
        raise Phase8CheckpointError("phase8.checkpoint.authority", field="tokenizer")
    dataset = _exact_mapping(
        authority["dataset"],
        (
            "dataset_id",
            "manifest_sha256",
            "processing_manifest_sha256",
            "training_works",
            "validation_works",
        ),
        field="dataset",
    )
    if (
        type(dataset["dataset_id"]) is not str
        or dataset["dataset_id"] != "shakespeare-eight-play"
        or type(dataset["manifest_sha256"]) is not str
        or dataset["manifest_sha256"] != PHASE1_MANIFEST_SHA256
        or type(dataset["processing_manifest_sha256"]) is not str
        or dataset["processing_manifest_sha256"] != PROCESSING_MANIFEST_SHA256
        or type(dataset["training_works"]) is not tuple
        or len(dataset["training_works"]) != 6
        or type(dataset["validation_works"]) is not tuple
        or len(dataset["validation_works"]) != 1
    ):
        raise Phase8CheckpointError("phase8.checkpoint.authority", field="dataset")
    expected_works = (
        (1, "hamlet", "train"),
        (2, "romeo-and-juliet", "train"),
        (3, "macbeth", "train"),
        (4, "a-midsummer-nights-dream", "train"),
        (5, "much-ado-about-nothing", "train"),
        (6, "henry-v", "train"),
        (7, "the-tempest", "validation"),
    )
    observed_works = (*dataset["training_works"], *dataset["validation_works"])
    for position, (work, expected) in enumerate(
        zip(observed_works, expected_works, strict=True)
    ):
        work_mapping = _exact_mapping(
            work,
            ("manifest_order", "work_id", "split", "processed_sha256"),
            field="work",
        )
        if (
            type(work_mapping["manifest_order"]) is not int
            or type(work_mapping["work_id"]) is not str
            or type(work_mapping["split"]) is not str
            or (
                work_mapping["manifest_order"],
                work_mapping["work_id"],
                work_mapping["split"],
            )
            != expected
            or not _exact_digest(work_mapping["processed_sha256"])
        ):
            raise Phase8CheckpointError(
                "phase8.checkpoint.authority",
                field="work",
                position=position,
            )
    model = _exact_mapping(
        authority["model"],
        (
            "model_width",
            "max_sequence_length",
            "number_of_blocks",
            "number_of_heads",
            "head_width",
            "hidden_width",
            "layer_norm_epsilon",
            "dropout_probability",
            "embedding_seed",
            "block_parameter_seeds",
            "output_head_seed",
            "parameter_device",
            "parameter_dtype",
            "trainable_tensor_count",
            "parameter_count",
        ),
        field="model",
    )
    expected_model = {
        "model_width": 32,
        "max_sequence_length": 256,
        "number_of_blocks": 4,
        "number_of_heads": 4,
        "head_width": 8,
        "hidden_width": 128,
        "layer_norm_epsilon": 1e-5,
        "dropout_probability": 0.1,
        "embedding_seed": 1337,
        "block_parameter_seeds": (7001, 7002, 7003, 7004),
        "output_head_seed": 7005,
        "parameter_device": "cpu",
        "parameter_dtype": "torch.float32",
        "trainable_tensor_count": 90,
        "parameter_count": 63_825,
    }
    if any(
        type(model[name]) is not type(expected) or model[name] != expected
        for name, expected in expected_model.items()
    ):
        raise Phase8CheckpointError("phase8.checkpoint.authority", field="model")
    runtime = _runtime_from_mapping(authority["runtime"])
    if authority["requirements_lock_sha256"] != runtime.requirements_lock_sha256:
        raise Phase8CheckpointError("phase8.checkpoint.cross_field", field="runtime")


def _validate_model_state(value: object) -> None:
    if type(value) is not OrderedDict or tuple(value) != tuple(
        name for name, _ in _MODEL_STATE_ORACLE
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="model_state")
    for position, ((_, shape), tensor) in enumerate(
        zip(_MODEL_STATE_ORACLE, value.values(), strict=True)
    ):
        if (
            type(tensor) is not torch.Tensor
            or tuple(tensor.shape) != shape
            or tensor.device != torch.device("cpu")
            or tensor.dtype is not torch.float32
            or not bool(torch.all(torch.isfinite(tensor)).item())
        ):
            raise Phase8CheckpointError(
                "phase8.checkpoint.schema",
                field="model_state",
                position=position,
            )


def _validate_optimizer_state(
    value: object,
    *,
    expected_updates: int,
) -> None:
    optimizer = _exact_mapping(
        value,
        ("state", "param_groups"),
        field="optimizer_state",
    )
    groups = optimizer["param_groups"]
    if type(groups) is not list or len(groups) != 1:
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="optimizer_state")
    group = _exact_mapping(
        groups[0],
        (
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
            "params",
        ),
        field="optimizer_group",
    )
    expected_group = {
        "lr": 3e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
        "params": list(range(90)),
    }
    if any(
        type(group[name]) is not type(expected) or group[name] != expected
        for name, expected in expected_group.items()
    ) or any(type(item) is not int for item in group["params"]):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="optimizer_group")
    state = optimizer["state"]
    if (
        type(state) is not dict
        or tuple(state) != tuple(range(90))
        or any(type(key) is not int for key in state)
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="optimizer_state")
    for position, (_, shape) in enumerate(_MODEL_STATE_ORACLE):
        parameter_state = _exact_mapping(
            state[position],
            ("step", "exp_avg", "exp_avg_sq"),
            field="optimizer_parameter_state",
        )
        step = parameter_state["step"]
        if (
            type(step) is not torch.Tensor
            or step.shape != torch.Size([])
            or step.device != torch.device("cpu")
            or step.dtype is not torch.float32
            or not math.isfinite(float(step.item()))
            or not float(step.item()).is_integer()
            or float(step.item()) != expected_updates
        ):
            raise Phase8CheckpointError(
                "phase8.checkpoint.cross_field",
                field="optimizer_step",
                position=position,
            )
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state[name]
            if (
                type(tensor) is not torch.Tensor
                or tuple(tensor.shape) != shape
                or tensor.device != torch.device("cpu")
                or tensor.dtype is not torch.float32
                or not bool(torch.all(torch.isfinite(tensor)).item())
            ):
                raise Phase8CheckpointError(
                    "phase8.checkpoint.schema",
                    field=name,
                    position=position,
                )


def _validate_payload(payload: dict[str, object]) -> None:
    if (
        tuple(payload) != _TOP_LEVEL_KEYS
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema")
    run = _exact_mapping(payload["run"], (
        "run_id",
        "logical_id",
        "completed_epoch",
        "code_commit",
    ), field="run")
    configuration = payload["configuration"]
    expected_configuration_keys = tuple(item.name for item in fields(Phase8Configuration))
    if (
        type(configuration) is not dict
        or tuple(configuration) != expected_configuration_keys
        or type(configuration.get("schema_version")) is not int
        or configuration["schema_version"] != 1
    ):
        raise Phase8CheckpointError("phase8.checkpoint.configuration")
    parsed_configuration = _configuration_from_mapping(configuration)
    _validate_authority_schema(payload["authority"])
    authority = payload["authority"]
    assert type(authority) is dict
    if authority["runtime"] != configuration["runtime"]:
        raise Phase8CheckpointError("phase8.checkpoint.cross_field", field="runtime")
    run_id = run["run_id"]
    epoch = run["completed_epoch"]
    if (
        type(run_id) is not str
        or not RUN_ID_PATTERN.fullmatch(run_id)
        or type(epoch) is not int
        or not 1 <= epoch <= 10
        or type(run["logical_id"]) is not str
        or run["logical_id"] != f"epoch-{epoch:04d}"
        or type(run["code_commit"]) is not str
        or not GIT_COMMIT_PATTERN.fullmatch(run["code_commit"])
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="run")
    progress = _exact_mapping(payload["progress"], (
        "completed_epochs",
        "next_epoch",
        "next_example_offset",
        "optimizer_updates",
        "examples_processed",
        "targets_processed",
        "per_epoch_optimizer_updates",
        "per_epoch_example_counts",
        "per_epoch_target_counts",
        "next_ordering_action",
    ), field="progress")
    parsed_progress = _progress_from_mapping(progress)
    if parsed_progress.completed_epochs != epoch:
        raise Phase8CheckpointError("phase8.checkpoint.cross_field", field="epoch")
    metric_state = _exact_mapping(payload["metric_state"], (
        "initialized_training",
        "initialized_validation",
        "epoch_training",
        "epoch_validation",
        "current_training",
        "current_validation",
        "best_validation_loss",
        "best_validation_loss_hex",
        "best_validation_epoch",
        "best_logical_id",
        "best_comparison",
    ), field="metric_state")
    parsed_metrics = _metric_from_mapping(
        metric_state,
        completed_epochs=parsed_progress.completed_epochs,
    )
    for index in range(parsed_progress.completed_epochs):
        training_evaluation = parsed_metrics.epoch_training[index]
        if (
            progress["per_epoch_example_counts"][index]
            != training_evaluation.window_count
            or progress["per_epoch_target_counts"][index]
            != training_evaluation.target_count
            or progress["per_epoch_optimizer_updates"][index]
            != math.ceil(training_evaluation.window_count / 8)
        ):
            raise Phase8CheckpointError(
                "phase8.checkpoint.cross_field",
                field="epoch_counts",
                position=index,
            )
    _validate_model_state(payload["model_state"])
    _validate_optimizer_state(
        payload["optimizer_state"],
        expected_updates=parsed_progress.optimizer_updates,
    )
    random_state = _exact_mapping(payload["random_state"], (
        "order_generator",
        "global_cpu",
        "dropout_generators",
    ), field="random_state")
    dropout = random_state["dropout_generators"]
    if type(dropout) is not dict or tuple(dropout) != _DROPOUT_NAMES:
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="dropout")
    for value in (
        random_state["order_generator"],
        random_state["global_cpu"],
        *dropout.values(),
    ):
        if (
            type(value) is not torch.Tensor
            or value.dim() != 1
            or value.device != torch.device("cpu")
            or value.dtype is not torch.uint8
        ):
            raise Phase8CheckpointError("phase8.checkpoint.schema", field="rng")
    expected_seeds = (
        8001,
        -1,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
    )
    for position, (value, expected_seed) in enumerate(
        zip(
            (
                random_state["order_generator"],
                random_state["global_cpu"],
                *dropout.values(),
            ),
            expected_seeds,
            strict=True,
        )
    ):
        generator = torch.Generator(device="cpu")
        try:
            generator.set_state(value)
        except BaseException as error:
            raise Phase8CheckpointError(
                "phase8.checkpoint.schema",
                field="rng_state",
                position=position,
            ) from error
        if expected_seed >= 0 and generator.initial_seed() != expected_seed:
            raise Phase8CheckpointError(
                "phase8.checkpoint.cross_field",
                field="rng_seed",
                position=position,
            )
    mode_state = _exact_mapping(
        payload["mode_state"],
        ("training", "named_modules"),
        field="mode_state",
    )
    if (
        type(mode_state["training"]) is not bool
        or mode_state["training"] is not True
        or type(mode_state["named_modules"]) is not tuple
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not bool
            or item[1] is not True
            for item in mode_state["named_modules"]
        )
        or mode_state["named_modules"] != _ACCEPTED_MODE_STATE
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="mode")
    lineage = _exact_mapping(payload["resume_lineage"], (
        "root_run_id",
        "resumed_from_checkpoint_sha256",
        "resume_count",
        "checkpoint_sha256s",
    ), field="lineage")
    hashes = lineage["checkpoint_sha256s"]
    if (
        type(lineage["root_run_id"]) is not str
        or lineage["root_run_id"] != run_id
        or type(lineage["resume_count"]) is not int
        or type(hashes) is not tuple
        or lineage["resume_count"] != len(hashes)
        or any(
            type(item) is not str or not SHA256_PATTERN.fullmatch(item)
            for item in hashes
        )
        or (
            not hashes and lineage["resumed_from_checkpoint_sha256"] is not None
        )
        or (
            hashes
            and (
                type(lineage["resumed_from_checkpoint_sha256"]) is not str
                or lineage["resumed_from_checkpoint_sha256"] != hashes[-1]
            )
        )
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="lineage")
    if parsed_configuration.runtime != _runtime_from_mapping(authority["runtime"]):
        raise Phase8CheckpointError("phase8.checkpoint.cross_field", field="runtime")


def _cross_validate(
    catalog: dict[str, object],
    catalog_role: str,
    reference: dict[str, object],
    payload: dict[str, object],
    digest: str,
) -> None:
    run = payload["run"]
    progress = payload["progress"]
    metrics = payload["metric_state"]
    lineage = payload["resume_lineage"]
    assert all(isinstance(value, dict) for value in (run, progress, metrics, lineage))
    current_validation = metrics["current_validation"]
    assert type(current_validation) is dict
    loss_hex = current_validation["loss_hex"]
    valid = (
        reference["role"] == catalog_role
        and catalog[catalog_role] is reference
        and catalog["run_id"]
        == reference["run_id"]
        == run["run_id"]
        == lineage["root_run_id"]
        and reference["logical_id"] == run["logical_id"]
        and reference["epoch"] == run["completed_epoch"] == progress["completed_epochs"]
        and reference["validation_loss_hex"] == loss_hex
        and reference["sha256"] == digest
        and reference["relative_path"] == f"objects/{digest}.pt"
    )
    if not valid:
        raise Phase8CheckpointError("phase8.checkpoint.cross_field")


def _cross_validate_catalog_graph(
    catalog: dict[str, object],
    latest: dict[str, object],
    latest_payload: dict[str, object],
    latest_digest: str,
    best: dict[str, object],
    best_payload: dict[str, object],
    best_digest: str,
) -> None:
    _cross_validate(catalog, "latest", latest, latest_payload, latest_digest)
    _cross_validate(
        catalog,
        "best_validation",
        best,
        best_payload,
        best_digest,
    )
    latest_metrics = latest_payload["metric_state"]
    best_metrics = best_payload["metric_state"]
    latest_run = latest_payload["run"]
    best_run = best_payload["run"]
    assert all(
        type(value) is dict
        for value in (latest_metrics, best_metrics, latest_run, best_run)
    )
    best_current = best_metrics["current_validation"]
    assert type(best_current) is dict
    if (
        latest["epoch"] != latest_run["completed_epoch"]
        or best["epoch"] > latest["epoch"]
        or best["epoch"] != latest_metrics["best_validation_epoch"]
        or best["logical_id"] != latest_metrics["best_logical_id"]
        or best["validation_loss_hex"]
        != latest_metrics["best_validation_loss_hex"]
        or best_run["completed_epoch"] != best["epoch"]
        or best_run["logical_id"] != best["logical_id"]
        or best_current["loss_hex"] != best["validation_loss_hex"]
        or best_metrics["best_validation_epoch"] != best["epoch"]
        or best_metrics["best_logical_id"] != best["logical_id"]
        or best_metrics["best_validation_loss_hex"]
        != best["validation_loss_hex"]
        or best["sha256"] != best_digest
        or best["relative_path"] != f"objects/{best_digest}.pt"
        or latest_payload["authority"] != best_payload["authority"]
        or latest_payload["configuration"] != best_payload["configuration"]
        or latest_run["run_id"] != best_run["run_id"]
        or latest_run["code_commit"] != best_run["code_commit"]
    ):
        raise Phase8CheckpointError("phase8.checkpoint.cross_field", field="best")


def _validate_catalog_graph_fds(
    run_fd: int,
    objects_fd: int,
    catalog: dict[str, object] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    if catalog is None:
        catalog = _read_catalog(run_fd)
    if catalog is None:
        raise Phase8CheckpointError("phase8.checkpoint.catalog")
    run_id = catalog["run_id"]
    assert type(run_id) is str
    latest = _validate_reference(catalog["latest"], "latest", run_id)
    best = _validate_reference(
        catalog["best_validation"],
        "best_validation",
        run_id,
    )
    latest_content, latest_payload = _load_object(objects_fd, latest)
    _validate_payload(latest_payload)
    if best["sha256"] == latest["sha256"]:
        best_content = latest_content
        best_payload = latest_payload
    else:
        best_content, best_payload = _load_object(objects_fd, best)
        _validate_payload(best_payload)
    _cross_validate_catalog_graph(
        catalog,
        latest,
        latest_payload,
        hashlib.sha256(latest_content).hexdigest(),
        best,
        best_payload,
        hashlib.sha256(best_content).hexdigest(),
    )
    return latest, latest_payload, best, best_payload


def _runtime_from_mapping(value: object) -> Phase8RuntimeIdentity:
    if type(value) is not dict or tuple(value) != tuple(
        item.name for item in fields(Phase8RuntimeIdentity)
    ):
        raise Phase8CheckpointError("phase8.checkpoint.configuration", field="runtime")
    runtime = Phase8RuntimeIdentity(**value)
    try:
        _validate_runtime_identity(runtime)
    except (Phase8TypeError, Phase8ContractError) as error:
        raise Phase8CheckpointError(
            "phase8.checkpoint.configuration",
            field="runtime",
        ) from error
    return runtime


def _configuration_from_mapping(value: dict[str, object]) -> Phase8Configuration:
    items = dict(value)
    items["runtime"] = _runtime_from_mapping(items["runtime"])
    configuration = _make_configuration(**items)
    expected = _fixed_configuration(configuration.runtime)
    for item in fields(Phase8Configuration):
        observed_value = getattr(configuration, item.name)
        expected_value = getattr(expected, item.name)
        if (
            type(observed_value) is not type(expected_value)
            or observed_value != expected_value
        ):
            raise Phase8CheckpointError(
                "phase8.checkpoint.configuration",
                field=item.name,
            )
    return configuration


def _evaluation_from_mapping(value: object) -> Phase8Evaluation:
    if type(value) is not dict or tuple(value) != (
        "split",
        "loss",
        "loss_hex",
        "target_count",
        "window_count",
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="evaluation")
    loss = value["loss"]
    if (
        type(value["split"]) is not str
        or value["split"] not in ("train", "validation")
        or type(loss) is not float
        or not math.isfinite(loss)
        or type(value["loss_hex"]) is not str
        or loss.hex() != value["loss_hex"]
        or type(value["target_count"]) is not int
        or value["target_count"] <= 0
        or type(value["window_count"]) is not int
        or value["window_count"] <= 0
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="evaluation")
    return Phase8Evaluation(
        split=value["split"],
        loss=loss,
        target_count=value["target_count"],
        window_count=value["window_count"],
    )


def _metric_from_mapping(
    value: dict[str, object],
    *,
    completed_epochs: int | None = None,
) -> Phase8MetricState:
    expected = (
        "initialized_training",
        "initialized_validation",
        "epoch_training",
        "epoch_validation",
        "current_training",
        "current_validation",
        "best_validation_loss",
        "best_validation_loss_hex",
        "best_validation_epoch",
        "best_logical_id",
        "best_comparison",
    )
    if tuple(value) != expected:
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="metrics")
    if (
        type(value["epoch_training"]) is not tuple
        or type(value["epoch_validation"]) is not tuple
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="metrics")
    epoch_training = tuple(
        _evaluation_from_mapping(item) for item in value["epoch_training"]
    )
    epoch_validation = tuple(
        _evaluation_from_mapping(item) for item in value["epoch_validation"]
    )
    initialized_training = _evaluation_from_mapping(value["initialized_training"])
    initialized_validation = _evaluation_from_mapping(value["initialized_validation"])
    current_training = _evaluation_from_mapping(value["current_training"])
    current_validation = _evaluation_from_mapping(value["current_validation"])
    best_loss = value["best_validation_loss"]
    if (
        type(best_loss) is not float
        or not math.isfinite(best_loss)
        or type(value["best_validation_loss_hex"]) is not str
        or best_loss.hex() != value["best_validation_loss_hex"]
        or type(value["best_validation_epoch"]) is not int
        or type(value["best_logical_id"]) is not str
        or type(value["best_comparison"]) is not str
        or value["best_comparison"] != "strict_lower"
        or initialized_training.split != "train"
        or initialized_validation.split != "validation"
        or any(item.split != "train" for item in epoch_training)
        or any(item.split != "validation" for item in epoch_validation)
        or current_training.split != "train"
        or current_validation.split != "validation"
        or not epoch_training
        or len(epoch_training) != len(epoch_validation)
        or current_training != epoch_training[-1]
        or current_validation != epoch_validation[-1]
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="metrics")
    validation_losses = tuple(item.loss for item in epoch_validation)
    expected_best = min(validation_losses)
    expected_best_epoch = validation_losses.index(expected_best) + 1
    if (
        (completed_epochs is not None and len(epoch_training) != completed_epochs)
        or any(
            (item.target_count, item.window_count)
            != (
                initialized_training.target_count,
                initialized_training.window_count,
            )
            for item in epoch_training
        )
        or any(
            (item.target_count, item.window_count)
            != (
                initialized_validation.target_count,
                initialized_validation.window_count,
            )
            for item in epoch_validation
        )
        or best_loss != expected_best
        or value["best_validation_epoch"] != expected_best_epoch
        or value["best_logical_id"] != f"epoch-{expected_best_epoch:04d}"
    ):
        raise Phase8CheckpointError("phase8.checkpoint.cross_field", field="metrics")
    return _make_metric_state(
        initialized_training=initialized_training,
        initialized_validation=initialized_validation,
        epoch_training=epoch_training,
        epoch_validation=epoch_validation,
        current_training=current_training,
        current_validation=current_validation,
        best_validation_loss=best_loss,
        best_validation_epoch=value["best_validation_epoch"],
        best_logical_id=value["best_logical_id"],
    )


def _progress_from_mapping(value: dict[str, object]) -> Phase8Progress:
    expected = (
        "completed_epochs",
        "next_epoch",
        "next_example_offset",
        "optimizer_updates",
        "examples_processed",
        "targets_processed",
        "per_epoch_optimizer_updates",
        "per_epoch_example_counts",
        "per_epoch_target_counts",
        "next_ordering_action",
    )
    if tuple(value) != expected:
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="progress")
    scalar_names = (
        "completed_epochs",
        "next_epoch",
        "next_example_offset",
        "optimizer_updates",
        "examples_processed",
        "targets_processed",
    )
    tuple_names = (
        "per_epoch_optimizer_updates",
        "per_epoch_example_counts",
        "per_epoch_target_counts",
    )
    if (
        any(type(value[name]) is not int for name in scalar_names)
        or any(type(value[name]) is not tuple for name in tuple_names)
        or any(
            type(item) is not int
            for name in tuple_names
            for item in value[name]
        )
        or type(value["next_ordering_action"]) is not str
        or value["next_ordering_action"] != "draw_next_epoch_permutation"
    ):
        raise Phase8CheckpointError("phase8.checkpoint.schema", field="progress")
    completed = value["completed_epochs"]
    if (
        not 1 <= completed <= 10
        or value["next_epoch"] != completed + 1
        or value["next_example_offset"] != 0
        or any(len(value[name]) != completed for name in tuple_names)
        or any(item <= 0 for name in tuple_names for item in value[name])
        or sum(value["per_epoch_optimizer_updates"]) != value["optimizer_updates"]
        or sum(value["per_epoch_example_counts"]) != value["examples_processed"]
        or sum(value["per_epoch_target_counts"]) != value["targets_processed"]
    ):
        raise Phase8CheckpointError("phase8.checkpoint.cross_field", field="progress")
    return Phase8Progress(
        completed_epochs=value["completed_epochs"],
        next_epoch=value["next_epoch"],
        next_example_offset=value["next_example_offset"],
        optimizer_updates=value["optimizer_updates"],
        examples_processed=value["examples_processed"],
        targets_processed=value["targets_processed"],
    )


def _construct_model(vocabulary: object) -> MiniGPT:
    return MiniGPT(
        vocabulary,  # type: ignore[arg-type]
        model_width=MODEL_WIDTH,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        number_of_blocks=NUMBER_OF_BLOCKS,
        number_of_heads=NUMBER_OF_HEADS,
        head_width=HEAD_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        layer_norm_epsilon=LAYER_NORM_EPSILON,
        dropout_probability=DROPOUT_PROBABILITY,
        embedding_seed=EMBEDDING_INITIALIZATION_SEED,
        block_parameter_seeds=BLOCK_PARAMETER_INITIALIZATION_SEEDS,
        output_head_seed=OUTPUT_HEAD_INITIALIZATION_SEED,
        device=PARAMETER_DEVICE,
        dtype=PARAMETER_DTYPE,
    )


def _load_phase8_checkpoint(
    repository_root: Path,
    *,
    run_id: str,
    role: Literal["latest"],
    components: tuple[str, ...] | None,
) -> Phase8TrainingState:
    if not isinstance(repository_root, Path):
        raise Phase8TypeError("phase8.type.argument", field="repository_root")
    if type(run_id) is not str or type(role) is not str:
        raise Phase8TypeError("phase8.type.argument")
    if role != "latest":
        raise Phase8CheckpointError("phase8.checkpoint.role.continuation")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise Phase8CheckpointError("phase8.checkpoint.path", field="run_id")
    runtime = _validate_runtime()
    caller_rng = torch.get_rng_state().clone()
    committed_global = False
    rollback_attempted = False
    directories: tuple[int, ...] | None = None
    try:
        configuration = _fixed_configuration(runtime)
        _validate_live_repository(
            repository_root,
            PHASE8_IMPLEMENTATION_COMMIT,
            run_id=run_id,
            configuration=configuration,
            require_run_evidence=True,
        )
        directories = _open_checkpoint_directories(
            repository_root,
            run_id,
            create=False,
            components=components,
        )
        directory_identities = _directory_identities(directories)
        catalog = _read_catalog(directories[-2])
        if catalog is None:
            raise Phase8CheckpointError("phase8.checkpoint.catalog")
        latest = _validate_reference(catalog["latest"], "latest", run_id)
        best = _validate_reference(catalog["best_validation"], "best_validation", run_id)
        latest_content, payload = _load_object(directories[-1], latest)
        _validate_payload(payload)
        if best["sha256"] == latest["sha256"]:
            best_content = latest_content
            best_payload = payload
        else:
            best_content, best_payload = _load_object(directories[-1], best)
            _validate_payload(best_payload)
        _cross_validate_catalog_graph(
            catalog,
            latest,
            payload,
            hashlib.sha256(latest_content).hexdigest(),
            best,
            best_payload,
            hashlib.sha256(best_content).hexdigest(),
        )
        _require_directory_identities(directories, directory_identities)
        if payload["run"]["code_commit"] != PHASE8_IMPLEMENTATION_COMMIT:  # type: ignore[index]
            raise Phase8CheckpointError(
                "phase8.checkpoint.authority",
                field="code_commit",
            )
        configuration = _configuration_from_mapping(payload["configuration"])
        if configuration != _fixed_configuration(runtime):
            raise Phase8CheckpointError("phase8.checkpoint.configuration")
        authority = payload["authority"]
        assert isinstance(authority, dict)
        if authority != _authority_mapping(configuration, repository_root):
            raise Phase8CheckpointError("phase8.checkpoint.authority")
        vocabulary = load_accepted_vocabulary_binding(repository_root)
        model = _construct_model(vocabulary)
        mode_state = payload["mode_state"]
        assert isinstance(mode_state, dict)
        expected_modes = tuple(
            (name, True) for name, _ in model.named_modules()
        )
        if mode_state["named_modules"] != expected_modes:
            raise Phase8CheckpointError(
                "phase8.checkpoint.schema",
                field="mode_names",
            )
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer = create_phase8_optimizer(model)
        optimizer.load_state_dict(payload["optimizer_state"])
        progress = _progress_from_mapping(payload["progress"])
        _validate_optimizer(
            model,
            optimizer,
            require_state=True,
            expected_updates=progress.optimizer_updates,
        )
        random_state = payload["random_state"]
        assert isinstance(random_state, dict)
        order_generator = torch.Generator(device="cpu")
        order_generator.set_state(random_state["order_generator"])
        dropout_states = random_state["dropout_generators"]
        assert isinstance(dropout_states, dict)
        for position, block in enumerate(model.blocks):
            block.attention_dropout._generator.set_state(
                dropout_states[_DROPOUT_NAMES[position * 2]]
            )
            block.feed_forward_dropout._generator.set_state(
                dropout_states[_DROPOUT_NAMES[position * 2 + 1]]
            )
        model.train()
        metrics = _metric_from_mapping(payload["metric_state"])
        lineage = payload["resume_lineage"]
        assert isinstance(lineage, dict)
        previous_lineage = tuple(lineage["checkpoint_sha256s"])
        selected_sha = latest["sha256"]
        restored = _make_training_state(
            run_id=run_id,
            code_commit=PHASE8_IMPLEMENTATION_COMMIT,
            configuration=configuration,
            model=model,
            optimizer=optimizer,
            order_generator=order_generator,
            progress=progress,
            metrics=metrics,
            global_cpu_rng_state=random_state["global_cpu"].clone(),
            resumed_from_checkpoint_sha256=selected_sha,
            resume_checkpoint_sha256s=(*previous_lineage, selected_sha),
        )
        _validate_training_state(restored, require_live_global=False)
        _require_directory_identities(directories, directory_identities)
        checkpoint_global = random_state["global_cpu"].clone()
        try:
            torch.set_rng_state(checkpoint_global)
            if not torch.equal(torch.get_rng_state(), checkpoint_global):
                raise RuntimeError("global RNG verification")
            committed_global = True
        except BaseException as error:
            rollback_attempted = True
            try:
                torch.set_rng_state(caller_rng)
                if not torch.equal(torch.get_rng_state(), caller_rng):
                    raise RuntimeError("global RNG rollback verification")
            except BaseException:
                raise Phase8CheckpointError(
                    "phase8.checkpoint.global_rng",
                    state_guarantee="unknown",
                ) from error
            raise Phase8CheckpointError("phase8.checkpoint.global_rng") from error
        return restored
    except BaseException as error:
        if not committed_global and not rollback_attempted:
            try:
                if not torch.equal(torch.get_rng_state(), caller_rng):
                    rollback_attempted = True
                    torch.set_rng_state(caller_rng)
                    if not torch.equal(torch.get_rng_state(), caller_rng):
                        raise RuntimeError("global RNG rollback verification")
            except BaseException:
                raise Phase8CheckpointError(
                    "phase8.checkpoint.global_rng",
                    state_guarantee="unknown",
                ) from error
        if isinstance(
            error,
            (
                Phase8TypeError,
                Phase8ContractError,
                Phase8GovernanceError,
                Phase8NumericalError,
                Phase8CheckpointError,
                Phase8PublicationError,
            ),
        ):
            raise
        raise Phase8CheckpointError("phase8.checkpoint.restore") from error
    finally:
        if directories is not None:
            _close_directories(directories)


def load_phase8_checkpoint(
    repository_root: Path,
    *,
    run_id: str,
    role: Literal["latest"],
) -> Phase8TrainingState:
    return _load_phase8_checkpoint(
        repository_root,
        run_id=run_id,
        role=role,
        components=None,
    )


__all__ = [
    "load_phase8_checkpoint",
    "save_phase8_checkpoint",
]
