"""Synthetic lifecycle and provenance tests for Phase 8."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.training.phase8_checkpoint as checkpoint  # noqa: E402
import sebgpt.training.phase8_experiment as experiment  # noqa: E402
from sebgpt.model.phase4_experiment import Phase4PermittedCorpus  # noqa: E402
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)
from sebgpt.training import (  # noqa: E402
    Phase8ContractError,
    Phase8GovernanceError,
    Phase8Evaluation,
    Phase8Window,
    configure_phase8_runtime,
    load_phase8_configuration,
    run_fixed_phase8_experiment,
    validate_phase8_runtime,
)


IMPLEMENTATION_COMMIT = "809834323d53407cb4a54ae539585bb3d78856eb"
PRE_REGISTRATION_COMMIT = "b" * 40
IMPLEMENTATION_AUTHORITY = "c" * 40
PHASE8_SOURCE_TEST_PATHS = (
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


def _copy_provenance_authority(repository_root: Path) -> None:
    paths = (
        "requirements.lock",
        "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
        "docs/data/shakespeare-eight-play-manifest.json",
        "data/processed/shakespeare-eight-play/processing-manifest.json",
        "docs/MINI_GPT_SPEC.md",
        "src/sebgpt/model/__init__.py",
        "src/sebgpt/model/mini_gpt.py",
        "tests/test_mini_gpt.py",
        "docs/TRAINING_CHECKPOINTING_SPEC.md",
        *PHASE8_SOURCE_TEST_PATHS,
    )
    for relative in paths:
        target = repository_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, target)
    (repository_root / "experiments").mkdir(exist_ok=True)


def _write_planned_record(
    repository_root: Path,
    configuration,
    *,
    code_commit: str = IMPLEMENTATION_COMMIT,
) -> bytes:
    authority = checkpoint._authority_mapping(configuration, repository_root)
    values = {name: "recorded" for name in experiment._PLANNED_RECORD_FIELDS}
    values.update(
        {
            "Experiment ID": "EXP-20260911-03",
            "Entry kind": "planned",
            "Recorded at UTC": "2026-09-11T12:00:00Z",
            "Status": "planned",
            "Code commit": code_commit,
            "Phase 7 authority": {
                key: authority[key]
                for key in (
                    "phase7_closure_commit",
                    "phase7_contract_commit",
                    "phase7_implementation_commit",
                    "mini_gpt_spec_sha256",
                    "mini_gpt_source_sha256",
                    "mini_gpt_test_sha256",
                )
            },
            "Phase 8 contract authority": {
                "phase8_contract_commit": "c50d77ac935bdf924b9b429a5776c419982a5d11",
                "phase8_spec_sha256": "1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd",
            },
            "Runtime identity": authority["runtime"],
            "Dataset authority": authority["dataset"],
            "Tokenizer authority": authority["tokenizer"],
            "Model configuration": authority["model"],
            "Training configuration": checkpoint._configuration_mapping(configuration),
            "Checkpoint configuration": {
                "checkpoint_schema_version": 1,
                "catalog_schema_version": 1,
                "maximum_catalog_bytes": 16_384,
                "maximum_checkpoint_object_bytes": 67_108_864,
            },
            "Success predicate": "min(training_loss at completed epochs 1..10) < initialized training_loss",
            "Sealed-test policy": "none",
            "Generation policy": "none",
        }
    )

    def render(value):
        if type(value) is dict:
            return json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        return f"`{value}`"

    content = (
        "### EXP-20260911-03 — Phase 8 fixed training\n\n"
        + "\n".join(
            f"- **{label}:** {render(values[label])}"
            for label in experiment._PLANNED_RECORD_FIELDS
        )
        + "\n"
    ).encode("utf-8")
    (repository_root / "EXPERIMENT_LOG.md").write_bytes(content)
    return content


class Phase8ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_phase8_runtime()

    def test_runtime_and_configuration_are_repeatable(self) -> None:
        first = validate_phase8_runtime()
        second = validate_phase8_runtime()
        self.assertEqual(first, second)
        config = load_phase8_configuration()
        self.assertEqual(config.runtime, first)
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.evaluation_split_order, ("train", "validation"))
        self.assertIsNone(config.scheduler_name)
        self.assertEqual(config.warmup_steps, 0)

    def test_exact_experiment_record_oracles(self) -> None:
        self.assertEqual(
            experiment._PLANNED_RECORD_FIELDS,
            (
                "Experiment ID", "Entry kind", "Recorded at UTC", "Status",
                "Question", "Authorization", "Code commit", "Phase 7 authority",
                "Phase 8 contract authority", "Runtime identity", "Dataset authority",
                "Tokenizer authority", "Model configuration", "Training configuration",
                "Checkpoint configuration", "Feasibility evidence", "Success predicate",
                "Sealed-test policy", "Generation policy", "Next gate",
            ),
        )
        self.assertEqual(
            experiment._RESULT_RECORD_FIELDS,
            (
                "Experiment ID", "Entry kind", "Recorded at UTC",
                "Started at UTC", "Finished at UTC", "Status", "Question",
                "Authorization and predecessor", "Code commit",
                "Pre-registration commit", "Phase 7 authority",
                "Phase 8 contract authority", "Runtime identity",
                "Dataset authority", "Tokenizer authority",
                "Model configuration", "Training configuration",
                "Window counts", "Target counts", "Update counts",
                "Initialized losses", "Epoch training losses",
                "Epoch validation losses", "Gradient-norm evidence",
                "Latest checkpoint", "Best-validation checkpoint",
                "Checkpoint publication evidence", "Resume audit",
                "Resume lineage", "Training-loss improvement predicate",
                "Training-loss improvement result", "Fitting evidence",
                "Sealed-test access", "Generation and samples",
                "Stopping reason", "Observations", "Conclusions",
                "Limitations", "Next gate",
            ),
        )
        self.assertEqual(
            experiment._float_record(1.5),
            {"decimal": "1.5", "hex": "0x1.8000000000000p+0"},
        )

    def test_planned_run_id_requires_one_exact_phase8_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {name: "recorded" for name in experiment._PLANNED_RECORD_FIELDS}
            values.update(
                {
                    "Experiment ID": "EXP-20260911-03",
                    "Entry kind": "planned",
                    "Recorded at UTC": "2026-09-11T12:00:00Z",
                    "Status": "planned",
                    "Code commit": IMPLEMENTATION_COMMIT,
                    "Success predicate": "min(training_loss at completed epochs 1..10) < initialized training_loss",
                }
            )
            (root / "EXPERIMENT_LOG.md").write_text(
                "### EXP-20260911-03 — Phase 8 fixed training\n\n"
                + "\n".join(
                    f"- **{label}:** `{values[label]}`"
                    for label in experiment._PLANNED_RECORD_FIELDS
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(experiment._planned_run_id(root), "EXP-20260911-03")
            (root / "EXPERIMENT_LOG.md").write_text("no planned run\n", encoding="utf-8")
            with self.assertRaises(Exception):
                experiment._planned_run_id(root)

    def test_complete_pre_registration_binds_every_authority_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for relative in (
                "requirements.lock",
                "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
                "docs/data/shakespeare-eight-play-manifest.json",
                "data/processed/shakespeare-eight-play/processing-manifest.json",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPOSITORY_ROOT / relative, target)
            config = load_phase8_configuration()
            authority = checkpoint._authority_mapping(config, root)
            values = {name: "recorded" for name in experiment._PLANNED_RECORD_FIELDS}
            values.update(
                {
                    "Experiment ID": "EXP-20260911-03",
                    "Entry kind": "planned",
                    "Recorded at UTC": "2026-09-11T12:00:00Z",
                    "Status": "planned",
                    "Code commit": IMPLEMENTATION_COMMIT,
                    "Phase 7 authority": {
                        key: authority[key]
                        for key in (
                            "phase7_closure_commit",
                            "phase7_contract_commit",
                            "phase7_implementation_commit",
                            "mini_gpt_spec_sha256",
                            "mini_gpt_source_sha256",
                            "mini_gpt_test_sha256",
                        )
                    },
                    "Phase 8 contract authority": {
                        "phase8_contract_commit": authority["phase8_contract_commit"],
                        "phase8_spec_sha256": authority["phase8_spec_sha256"],
                    },
                    "Runtime identity": authority["runtime"],
                    "Dataset authority": authority["dataset"],
                    "Tokenizer authority": authority["tokenizer"],
                    "Model configuration": authority["model"],
                    "Training configuration": checkpoint._configuration_mapping(config),
                    "Checkpoint configuration": {
                        "checkpoint_schema_version": 1,
                        "catalog_schema_version": 1,
                        "maximum_catalog_bytes": 16_384,
                        "maximum_checkpoint_object_bytes": 67_108_864,
                    },
                    "Sealed-test policy": "none",
                    "Generation policy": "none",
                    "Success predicate": "min(training_loss at completed epochs 1..10) < initialized training_loss",
                }
            )

            def render(value):
                if type(value) is dict:
                    return json.dumps(
                        value,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                return f"`{value}`"

            log_path = root / "EXPERIMENT_LOG.md"
            log_path.write_text(
                "### EXP-20260911-03 — Phase 8 fixed training\n\n"
                + "\n".join(
                    f"- **{label}:** {render(values[label])}"
                    for label in experiment._PLANNED_RECORD_FIELDS
                )
                + "\n",
                encoding="utf-8",
            )
            run_id, record = experiment._planned_run_record(
                root,
                code_commit=IMPLEMENTATION_COMMIT,
                configuration=config,
            )
            self.assertEqual(run_id, "EXP-20260911-03")
            self.assertEqual(tuple(record), experiment._PLANNED_RECORD_FIELDS)
            values["Runtime identity"] = {
                **authority["runtime"],
                "processor": "mismatch",
            }
            log_path.write_text(
                "### EXP-20260911-03 — Phase 8 fixed training\n\n"
                + "\n".join(
                    f"- **{label}:** {render(values[label])}"
                    for label in experiment._PLANNED_RECORD_FIELDS
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                experiment._planned_run_record(
                    root,
                    code_commit=IMPLEMENTATION_COMMIT,
                    configuration=config,
                )

    def test_planned_code_commit_is_literal_implementation_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _copy_provenance_authority(root)
            config = load_phase8_configuration()
            content = _write_planned_record(root, config)
            self.assertNotIn(PRE_REGISTRATION_COMMIT.encode("ascii"), content)
            run_id, record = experiment._planned_run_record(
                root,
                code_commit=IMPLEMENTATION_COMMIT,
                configuration=config,
            )
            self.assertEqual(run_id, "EXP-20260911-03")
            self.assertEqual(record["Code commit"], IMPLEMENTATION_COMMIT)
            _write_planned_record(root, config, code_commit="d" * 40)
            with self.assertRaises(Phase8ContractError) as caught:
                experiment._planned_run_record(
                    root,
                    code_commit=IMPLEMENTATION_COMMIT,
                    configuration=config,
                )
            self.assertEqual(
                caught.exception.details,
                {
                    "invariant": "phase8.contract.lifecycle",
                    "field": "planned_record_code_commit",
                },
            )

    def test_corrected_live_repository_preflight_matrix(self) -> None:
        self.assertEqual(
            checkpoint.PHASE8_IMPLEMENTATION_COMMIT,
            "809834323d53407cb4a54ae539585bb3d78856eb",
        )
        self.assertEqual(
            checkpoint.PHASE8_CONTRACT_COMMIT,
            "c50d77ac935bdf924b9b429a5776c419982a5d11",
        )
        self.assertEqual(
            checkpoint.PHASE8_SPEC_SHA256,
            "1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd",
        )
        self.assertEqual(
            checkpoint._PHASE8_SOURCE_TEST_PATHS,
            PHASE8_SOURCE_TEST_PATHS,
        )

        def exercise(**changes):
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name).resolve()
            _copy_provenance_authority(root)
            config = load_phase8_configuration()
            committed_log = _write_planned_record(root, config)
            facts = {
                "origin": PRE_REGISTRATION_COMMIT,
                "status": "",
                "parent_diff": "EXPERIMENT_LOG.md",
                "anchor_diff": "\n".join(
                    (
                        "DECISIONS.md",
                        "PROJECT_STATE.md",
                        "README.md",
                        "ROADMAP.md",
                        "docs/TRAINING_CHECKPOINTING_SPEC.md",
                        "src/sebgpt/training/phase8_checkpoint.py",
                        "src/sebgpt/training/phase8_experiment.py",
                        "tests/test_phase8_experiment.py",
                    )
                ),
                "committed_log": committed_log,
                "parent_log": b"# Experiment Log\n",
                "bad_ancestor": None,
                "bad_source": None,
                **changes,
            }

            def fake_git(_root, *arguments):
                if arguments == ("rev-parse", "--show-toplevel"):
                    return str(root)
                if arguments == ("branch", "--show-current"):
                    return "main"
                if arguments == ("rev-parse", "HEAD"):
                    return PRE_REGISTRATION_COMMIT
                if arguments == ("rev-parse", "origin/main"):
                    return facts["origin"]
                if arguments[:2] == ("status", "--porcelain=v2"):
                    return facts["status"]
                if arguments[:2] == ("merge-base", "--is-ancestor"):
                    if arguments[2] == facts["bad_ancestor"]:
                        raise Phase8GovernanceError(
                            "phase8.governance.repository",
                            field="git",
                        )
                    return ""
                if arguments == (
                    "rev-list", "--parents", "-n", "1", PRE_REGISTRATION_COMMIT
                ):
                    return f"{PRE_REGISTRATION_COMMIT} {IMPLEMENTATION_AUTHORITY}"
                if arguments == (
                    "diff", "--name-only", IMPLEMENTATION_AUTHORITY,
                    PRE_REGISTRATION_COMMIT,
                ):
                    return facts["parent_diff"]
                if arguments == (
                    "diff", "--name-only", IMPLEMENTATION_COMMIT,
                    IMPLEMENTATION_AUTHORITY,
                ):
                    return facts["anchor_diff"]
                raise AssertionError(arguments)

            def fake_blob(_root, revision, relative_path):
                if revision == PRE_REGISTRATION_COMMIT:
                    self.assertEqual(relative_path, "EXPERIMENT_LOG.md")
                    return facts["committed_log"]
                if revision == IMPLEMENTATION_AUTHORITY:
                    if relative_path == "EXPERIMENT_LOG.md":
                        return facts["parent_log"]
                    if relative_path == facts["bad_source"]:
                        return b"altered"
                    return (root / relative_path).read_bytes()
                raise AssertionError((revision, relative_path))

            return root, config, fake_git, fake_blob

        root, config, fake_git, fake_blob = exercise()
        with (
            patch.object(checkpoint, "_git", side_effect=fake_git),
            patch.object(checkpoint, "_git_blob_bytes", side_effect=fake_blob),
        ):
            self.assertEqual(
                checkpoint._validate_live_repository(
                    root,
                    IMPLEMENTATION_COMMIT,
                    run_id="EXP-20260911-03",
                    configuration=config,
                ),
                PRE_REGISTRATION_COMMIT,
            )
            checkpoint._publish_run_authority_evidence(
                root,
                "EXP-20260911-03",
                IMPLEMENTATION_COMMIT,
                PRE_REGISTRATION_COMMIT,
                hashlib.sha256(
                    (root / "EXPERIMENT_LOG.md").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                checkpoint._validate_live_repository(
                    root,
                    IMPLEMENTATION_COMMIT,
                    run_id="EXP-20260911-03",
                    configuration=config,
                    require_run_evidence=True,
                ),
                PRE_REGISTRATION_COMMIT,
            )
            with self.assertRaises(Phase8GovernanceError) as caught:
                checkpoint._validate_live_repository(
                    root,
                    "d" * 40,
                    run_id="EXP-20260911-03",
                    configuration=config,
                )
            self.assertEqual(
                caught.exception.details["field"],
                "implementation_authority",
            )

        cases = (
            ("ancestry", {"bad_ancestor": IMPLEMENTATION_COMMIT}, "implementation_authority"),
            ("origin", {"origin": "d" * 40}, "pre_registration_commit"),
            ("missing_record_commit", {"parent_diff": ""}, "pre_registration_commit"),
            (
                "record_already_in_parent",
                {"parent_log": b"### EXP-20260911-03 \xe2\x80\x94 Phase 8\n"},
                "pre_registration_commit",
            ),
            ("altered_record", {"committed_log": b"altered\n"}, "pre_registration_commit"),
            (
                "unapproved_descendant",
                {"anchor_diff": "src/sebgpt/model/mini_gpt.py"},
                "implementation_authority",
            ),
            (
                "source_bytes",
                {"bad_source": "src/sebgpt/training/phase8_experiment.py"},
                "implementation_authority",
            ),
            (
                "test_bytes",
                {"bad_source": "tests/test_phase8_experiment.py"},
                "implementation_authority",
            ),
            ("dirty", {"status": "1 .M tracked"}, "clean"),
        )
        for name, changes, expected_field in cases:
            with self.subTest(name=name):
                root, config, fake_git, fake_blob = exercise(**changes)
                with (
                    patch.object(checkpoint, "_git", side_effect=fake_git),
                    patch.object(
                        checkpoint,
                        "_git_blob_bytes",
                        side_effect=fake_blob,
                    ),
                    self.assertRaises(Phase8GovernanceError) as caught,
                ):
                    checkpoint._validate_live_repository(
                        root,
                        IMPLEMENTATION_COMMIT,
                        run_id="EXP-20260911-03",
                        configuration=config,
                    )
                self.assertEqual(caught.exception.details["field"], expected_field)

        root, config, fake_git, fake_blob = exercise()
        real_hash = checkpoint._sha256_file

        def wrong_spec(path):
            if path.name == "TRAINING_CHECKPOINTING_SPEC.md":
                return "0" * 64
            return real_hash(path)

        with (
            patch.object(checkpoint, "_git", side_effect=fake_git),
            patch.object(checkpoint, "_git_blob_bytes", side_effect=fake_blob),
            patch.object(checkpoint, "_sha256_file", side_effect=wrong_spec),
            self.assertRaises(Phase8GovernanceError) as caught,
        ):
            checkpoint._validate_live_repository(
                root,
                IMPLEMENTATION_COMMIT,
                run_id="EXP-20260911-03",
                configuration=config,
            )
        self.assertEqual(caught.exception.details["field"], "implementation_authority")

    def test_result_provenance_representation_keeps_commits_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "experiments").mkdir()
            checkpoint._publish_run_authority_evidence(
                root,
                "EXP-20260911-03",
                IMPLEMENTATION_COMMIT,
                PRE_REGISTRATION_COMMIT,
                "e" * 64,
            )
            checkpoint._require_run_authority_evidence(
                root,
                "EXP-20260911-03",
                IMPLEMENTATION_COMMIT,
                PRE_REGISTRATION_COMMIT,
                "e" * 64,
            )
            with self.assertRaises(Phase8GovernanceError) as caught:
                checkpoint._require_run_authority_evidence(
                    root,
                    "EXP-20260911-03",
                    IMPLEMENTATION_COMMIT,
                    "f" * 40,
                    "e" * 64,
                )
            self.assertEqual(
                caught.exception.details["field"],
                "pre_registration_commit",
            )
            reference = checkpoint.Phase8CheckpointReference(
                "EXP-20260911-03",
                "latest",
                "epoch-0010",
                10,
                float(1.0).hex(),
                "d" * 64,
                f"objects/{'d' * 64}.pt",
            )
            artifact = experiment._write_evidence_artifact(
                root,
                "EXP-20260911-03",
                IMPLEMENTATION_COMMIT,
                PRE_REGISTRATION_COMMIT,
                (),
                reference,
                reference,
            )
            content = json.loads((root / artifact["relative_path"]).read_text())
            self.assertEqual(content["code_commit"], IMPLEMENTATION_COMMIT)
            self.assertEqual(
                content["pre_registration_commit"],
                PRE_REGISTRATION_COMMIT,
            )
            self.assertNotEqual(
                content["code_commit"],
                content["pre_registration_commit"],
            )

    def test_run_authority_path_rejects_symlinks_and_wrong_types_without_outside_write(self) -> None:
        cases = ("run_symlink", "artifacts_symlink", "run_file", "artifacts_file")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_directory:
                root = Path(directory).resolve()
                outside = Path(outside_directory).resolve()
                experiments = root / "experiments"
                experiments.mkdir()
                run_path = experiments / "EXP-20260911-03"
                sentinel = outside / "sentinel"
                sentinel.write_bytes(b"unchanged")
                before = tuple(sorted(path.name for path in outside.iterdir()))
                if case == "run_symlink":
                    run_path.symlink_to(outside, target_is_directory=True)
                elif case == "run_file":
                    run_path.write_bytes(b"wrong type")
                else:
                    run_path.mkdir()
                    artifacts = run_path / "artifacts"
                    if case == "artifacts_symlink":
                        artifacts.symlink_to(outside, target_is_directory=True)
                    else:
                        artifacts.write_bytes(b"wrong type")
                with self.assertRaises(Phase8GovernanceError) as caught:
                    checkpoint._publish_run_authority_evidence(
                        root,
                        "EXP-20260911-03",
                        IMPLEMENTATION_COMMIT,
                        PRE_REGISTRATION_COMMIT,
                        "e" * 64,
                    )
                self.assertEqual(
                    caught.exception.details,
                    {
                        "invariant": "phase8.governance.repository",
                        "field": "pre_registration_commit",
                    },
                )
                self.assertEqual(sentinel.read_bytes(), b"unchanged")
                self.assertEqual(
                    tuple(sorted(path.name for path in outside.iterdir())),
                    before,
                )
                self.assertFalse(
                    (outside / "artifacts" / "phase8-run-authority.json").exists()
                )
                self.assertFalse((outside / "phase8-run-authority.json").exists())

    def test_run_authority_path_existing_tree_and_descriptor_relative_fsync_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            experiments = root / "experiments"
            experiments.mkdir()
            run_path = experiments / "EXP-20260911-03"
            artifacts = run_path / "artifacts"
            artifacts.mkdir(parents=True)
            existing = (run_path.stat().st_ino, artifacts.stat().st_ino)
            checkpoint._publish_run_authority_evidence(
                root,
                "EXP-20260911-03",
                IMPLEMENTATION_COMMIT,
                PRE_REGISTRATION_COMMIT,
                "e" * 64,
            )
            self.assertEqual(
                (run_path.stat().st_ino, artifacts.stat().st_ino),
                existing,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            experiments = root / "experiments"
            experiments.mkdir()
            open_calls = []
            fsync_identities = []
            real_open = os.open
            real_fsync = os.fsync

            def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
                open_calls.append((path, flags, dir_fd))
                return real_open(path, flags, mode, dir_fd=dir_fd)

            def tracked_fsync(fd):
                value = os.fstat(fd)
                fsync_identities.append((value.st_dev, value.st_ino))
                return real_fsync(fd)

            with (
                patch.object(checkpoint.os, "open", side_effect=tracked_open),
                patch.object(checkpoint.os, "fsync", side_effect=tracked_fsync),
            ):
                checkpoint._publish_run_authority_evidence(
                    root,
                    "EXP-20260911-03",
                    IMPLEMENTATION_COMMIT,
                    PRE_REGISTRATION_COMMIT,
                    "e" * 64,
                )
            run_path = experiments / "EXP-20260911-03"
            artifacts = run_path / "artifacts"
            evidence = artifacts / "phase8-run-authority.json"

            def identity(path):
                value = path.stat()
                return value.st_dev, value.st_ino

            self.assertEqual(
                fsync_identities,
                [
                    identity(run_path),
                    identity(experiments),
                    identity(artifacts),
                    identity(run_path),
                    identity(evidence),
                    identity(artifacts),
                ],
            )
            self.assertEqual(open_calls[0][0], root)
            self.assertEqual(
                [call[0] for call in open_calls[1:]],
                [
                    "experiments",
                    "EXP-20260911-03",
                    "artifacts",
                    "phase8-run-authority.json",
                ],
            )
            self.assertTrue(all(call[2] is not None for call in open_calls[1:]))
            self.assertTrue(
                all(
                    call[1] & os.O_NOFOLLOW
                    for call in open_calls
                )
            )
            self.assertTrue(
                all(call[1] & os.O_DIRECTORY for call in open_calls[:4])
            )
            self.assertFalse(open_calls[4][1] & os.O_DIRECTORY)
            open_calls.clear()
            with patch.object(checkpoint.os, "open", side_effect=tracked_open):
                checkpoint._require_run_authority_evidence(
                    root,
                    "EXP-20260911-03",
                    IMPLEMENTATION_COMMIT,
                    PRE_REGISTRATION_COMMIT,
                    "e" * 64,
                )
            self.assertEqual(open_calls[0][0], root)
            self.assertEqual(
                [call[0] for call in open_calls[1:]],
                [
                    "experiments",
                    "EXP-20260911-03",
                    "artifacts",
                    "phase8-run-authority.json",
                ],
            )
            self.assertTrue(all(call[2] is not None for call in open_calls[1:]))
            self.assertTrue(all(call[1] & os.O_NOFOLLOW for call in open_calls))
            self.assertTrue(
                all(call[1] & os.O_DIRECTORY for call in open_calls[:4])
            )
            self.assertFalse(open_calls[4][1] & os.O_DIRECTORY)

    def test_run_authority_directory_failure_maps_to_governance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "experiments").mkdir()
            with (
                patch.object(checkpoint.os, "mkdir", side_effect=OSError("injected")),
                self.assertRaises(Phase8GovernanceError) as caught,
            ):
                checkpoint._publish_run_authority_evidence(
                    root,
                    "EXP-20260911-03",
                    IMPLEMENTATION_COMMIT,
                    PRE_REGISTRATION_COMMIT,
                    "e" * 64,
                )
            self.assertEqual(
                caught.exception.details,
                {
                    "invariant": "phase8.governance.repository",
                    "field": "pre_registration_commit",
                },
            )

    def test_run_authority_cleanup_preserves_substituted_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifacts = (
                root
                / "experiments"
                / "EXP-20260911-03"
                / "artifacts"
            )
            artifacts.mkdir(parents=True)
            evidence = artifacts / "phase8-run-authority.json"

            def substitute_then_fail(_fd, _content):
                evidence.unlink()
                evidence.write_bytes(b"replacement")
                raise OSError("injected")

            with (
                patch.object(checkpoint.os, "write", side_effect=substitute_then_fail),
                self.assertRaises(Phase8GovernanceError) as caught,
            ):
                checkpoint._publish_run_authority_evidence(
                    root,
                    "EXP-20260911-03",
                    IMPLEMENTATION_COMMIT,
                    PRE_REGISTRATION_COMMIT,
                    "e" * 64,
                )
            self.assertEqual(caught.exception.details["field"], "pre_registration_commit")
            self.assertEqual(evidence.read_bytes(), b"replacement")

    def test_sealed_supplier_is_rejected_before_configuration_or_repository(self) -> None:
        corpus = Phase4PermittedCorpus(
            phase_1_manifest={},
            phase_1_manifest_sha256="x",
            training_works=(),
            validation_works=(),
            training_source_tokens=0,
            training_targets=0,
            training_examples=0,
            validation_source_tokens=0,
            validation_targets=0,
            validation_examples=0,
            sealed_test_supplier=object(),
        )
        vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
        with self.assertRaises(Phase8GovernanceError) as caught:
            run_fixed_phase8_experiment(
                corpus,
                vocabulary,
                load_phase8_configuration(),
                code_commit="a" * 40,
            )
        self.assertEqual(
            caught.exception.details["invariant"],
            "phase8.governance.sealed_test",
        )

    def test_runner_has_no_epoch_zero_state_and_executes_durable_dual_branch_audit(self) -> None:
        from tests.test_phase8_checkpoint import _copy_authority
        from sebgpt.training.phase8_types import _make_metric_state as real_make_metric

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _copy_authority(root)
            (root / "experiments").mkdir()
            (root / "EXPERIMENT_LOG.md").write_text(
                "synthetic planned record\n",
                encoding="utf-8",
            )
            corpus = Phase4PermittedCorpus(
                phase_1_manifest={},
                phase_1_manifest_sha256="x",
                training_works=(),
                validation_works=(),
                training_source_tokens=0,
                training_targets=0,
                training_examples=0,
                validation_source_tokens=0,
                validation_targets=0,
                validation_examples=0,
                sealed_test_supplier=None,
            )
            training_windows = (
                Phase8Window("hamlet", 1, "train", 0, (1, 2), (2, 3)),
            )
            validation_windows = (
                Phase8Window(
                    "the-tempest", 7, "validation", 0, (1, 2), (2, 3)
                ),
            )
            metric_calls = []
            audit_branches = []
            audit_publications = []
            audit_reloads = []
            semantic_comparisons = []
            real_audit = experiment._run_exact_resume_audit
            real_publish = experiment._publish_payload
            real_private_load = experiment._load_phase8_checkpoint
            real_public_load = experiment.load_phase8_checkpoint
            real_semantic_equal = experiment._semantic_equal

            def capture_metric(**values):
                metric_calls.append(values)
                return real_make_metric(**values)

            def evaluate(_model, _optimizer, windows, _generator, progress, *, split):
                base = 2.0 if split == "train" else 2.1
                direction = -0.1 if split == "train" else 0.01
                return Phase8Evaluation(
                    split,
                    base + direction * progress.completed_epochs,
                    2,
                    len(windows),
                )

            def audit(*args, **kwargs):
                result = real_audit(*args, **kwargs)
                audit_branches.append(result[:2])
                return result

            def publish(*args, **kwargs):
                result = real_publish(*args, **kwargs)
                audit_publications.append((kwargs.get("components"), result))
                return result

            def private_load(*args, **kwargs):
                result = real_private_load(*args, **kwargs)
                audit_reloads.append((kwargs.get("components"), result))
                return result

            def public_load(*args, **kwargs):
                result = real_public_load(*args, **kwargs)
                audit_reloads.append((None, result))
                return result

            def semantic_equal(left, right):
                result = real_semantic_equal(left, right)
                semantic_comparisons.append(
                    (left.progress.completed_epochs, right.progress.completed_epochs, result)
                )
                return result

            vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
            with (
                patch.object(experiment, "REPOSITORY_ROOT", root),
                patch.object(
                    experiment,
                    "_validate_live_repository",
                    return_value=PRE_REGISTRATION_COMMIT,
                ),
                patch.object(checkpoint, "_validate_live_repository", return_value=None),
                patch.object(checkpoint, "_git", return_value=IMPLEMENTATION_COMMIT),
                patch.object(
                    experiment,
                    "_planned_run_record",
                    return_value=("EXP-20260911-03", {}),
                ),
                patch.object(
                    experiment,
                    "build_phase8_windows",
                    side_effect=lambda *_args, split: (
                        training_windows if split == "train" else validation_windows
                    ),
                ),
                patch.object(experiment, "evaluate_phase8_split", side_effect=evaluate),
                patch.object(experiment, "_make_metric_state", side_effect=capture_metric),
                patch.object(experiment, "_run_exact_resume_audit", side_effect=audit),
                patch.object(experiment, "_publish_payload", side_effect=publish),
                patch.object(experiment, "_load_phase8_checkpoint", side_effect=private_load),
                patch.object(experiment, "load_phase8_checkpoint", side_effect=public_load),
                patch.object(experiment, "_semantic_equal", side_effect=semantic_equal),
            ):
                result = run_fixed_phase8_experiment(
                    corpus,
                    vocabulary,
                    load_phase8_configuration(),
                    code_commit=IMPLEMENTATION_COMMIT,
                )
            self.assertEqual(result.progress.completed_epochs, 10)
            self.assertTrue(result.exact_resume_passed)
            self.assertEqual(len(audit_branches), 1)
            control, resumed = audit_branches[0]
            self.assertIsNot(control, resumed)
            self.assertIsNot(control.model, resumed.model)
            self.assertTrue(
                set(map(id, control.model.parameters())).isdisjoint(
                    set(map(id, resumed.model.parameters()))
                )
            )
            self.assertEqual(
                (control.progress.completed_epochs, resumed.progress.completed_epochs),
                (2, 2),
            )
            control_payload = checkpoint._payload(control, root)
            resumed_payload = checkpoint._payload(resumed, root)
            control_lineage = control_payload.pop("resume_lineage")
            resumed_lineage = resumed_payload.pop("resume_lineage")
            self.assertNotEqual(control_lineage, resumed_lineage)
            self.assertTrue(
                experiment._objects_equal(control_payload, resumed_payload)
            )
            audit_components = (
                "experiments",
                "EXP-20260911-03",
                "artifacts",
                "resume-audit-control",
                "objects",
            )
            self.assertTrue(
                any(components == audit_components for components, _ in audit_publications)
            )
            self.assertTrue(
                any(components == audit_components for components, _ in audit_reloads)
            )
            self.assertTrue(
                any(
                    components is None and state.progress.completed_epochs == 2
                    for components, state in audit_reloads
                )
            )
            self.assertIn((2, 2, True), semantic_comparisons)
            control_root = root / "experiments/EXP-20260911-03/artifacts/resume-audit-control"
            canonical_root = root / "checkpoints/phase8/EXP-20260911-03"
            self.assertTrue((control_root / "checkpoint-catalog.json").is_file())
            self.assertTrue((canonical_root / "checkpoint-catalog.json").is_file())
            self.assertGreaterEqual(len(tuple((control_root / "objects").glob("*.pt"))), 1)
            self.assertGreaterEqual(len(tuple((canonical_root / "objects").glob("*.pt"))), 10)
            evidence = json.loads(
                (
                    root
                    / "experiments/EXP-20260911-03/artifacts/phase8-training-evidence.json"
                ).read_text(encoding="utf-8")
            )
            audit_record = next(
                item
                for item in evidence["batch_and_resume_evidence"]
                if item.get("both_catalog_graphs_verified") is True
            )
            self.assertIs(audit_record["both_public_load_paths_reloaded"], True)
            self.assertIs(
                audit_record["semantic_state_equal_except_lineage_hash_location"],
                True,
            )
            self.assertNotEqual(
                audit_record["control_checkpoint_sha256"],
                audit_record["resumed_checkpoint_sha256"],
            )
            self.assertTrue(metric_calls)
            self.assertTrue(
                all(
                    math.isfinite(values["best_validation_loss"])
                    and values["best_validation_epoch"] >= 1
                    and values["best_logical_id"] != "epoch-0000"
                    for values in metric_calls
                )
            )


if __name__ == "__main__":
    unittest.main()
