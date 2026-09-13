"""Read-only Phase 9 best-validation checkpoint loading."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import re
import stat
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch

from sebgpt.inference.phase9_types import (
    Phase9CheckpointError,
    Phase9CheckpointIdentity,
    Phase9ContractError,
    Phase9GovernanceError,
    Phase9InferenceBundle,
    Phase9TypeError,
    _make_inference_bundle,
    _validate_phase9_minigpt_structure,
)
from sebgpt.model.mini_gpt import (
    ATTENTION_DROPOUT_SEED,
    FEED_FORWARD_DROPOUT_SEED,
    MiniGPT,
)
from sebgpt.tokenization.code_point import CodePointTokenizer
from sebgpt.tokenization.vocabulary_artifact import load_accepted_vocabulary_binding


PHASE8_CLOSURE_COMMIT = "6d086625cb293f61700bd59b551ea310a5b680b9"
PHASE8_RESULT_COMMIT = "334119e63c716e4922cd0b67da4f5fc221faa886"
PHASE8_IMPLEMENTATION_COMMIT = "809834323d53407cb4a54ae539585bb3d78856eb"
PHASE8_CONTRACT_COMMIT = "c50d77ac935bdf924b9b429a5776c419982a5d11"
PHASE8_SPEC_SHA256 = "1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd"
PHASE7_CLOSURE_COMMIT = "33d4510421107848c4aa8a6014f4a7b1e391065a"
PHASE7_CONTRACT_COMMIT = "60b2a9cce55da79ccc9fbd03fad014cb2a939290"
PHASE7_IMPLEMENTATION_COMMIT = "3139b1736f005fe903e2ea111d91934478b5a683"
MINI_GPT_SPEC_SHA256 = "3e1987658d4c9ece59bebbea7061f939c1138aaa73751a523f659f8b3217c022"
MINI_GPT_SOURCE_SHA256 = "6b8db96577a0ad1f5daa4649d7ca9375e59873d4b635c3f76e75a6751f74f12d"
MINI_GPT_TEST_SHA256 = "95ec62d1164c48525e59e33f3f90fada6d482c84a31ca16edba38178b16b2cab"
TOKENIZER_IMPLEMENTATION_COMMIT = "de7a7f096fbbd8607c944412eaef30be9b686b56"
VOCABULARY_SHA256 = "9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e"
PHASE1_MANIFEST_SHA256 = "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"
PROCESSING_MANIFEST_SHA256 = "bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc"
PHASE9_CONTRACT_COMMIT = "48357be6f717e7d4e0445b96ca9886f122673a83"
PHASE9_SPEC_SHA256 = "61489c3a146d6fa6d3ff6a59f6052ff978ae283d42520c599bd1a25b14a71780"
RUN_ID = "EXP-20260912-01"
CATALOG_SHA256 = "6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241"
CHECKPOINT_SHA256 = "6990c89166d48732b0601aeae1edd57a9f0e22eec1c7c1160d5be1b8281c9148"
VALIDATION_LOSS_HEX = "0x1.3f94b678f5807p+1"
PHASE8_RESULT_RECORD_SHA256 = "e7b40c888bc23c4e58d5abad10a766be16698d917ee8d8f54eaa6a1d199706d5"
CATALOG_LIMIT = 16_384
OBJECT_LIMIT = 67_108_864
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"EXP-[0-9]{8}-[0-9]{2}\Z")

PROTECTED_PATHS = (
    "src/sebgpt/__init__.py",
    "src/sebgpt/tokenization/__init__.py",
    "src/sebgpt/tokenization/code_point.py",
    "src/sebgpt/tokenization/vocabulary_artifact.py",
    "src/sebgpt/model/__init__.py",
    "src/sebgpt/model/embeddings.py",
    "src/sebgpt/model/self_attention.py",
    "src/sebgpt/model/transformer_block.py",
    "src/sebgpt/model/simple_language_model.py",
    "src/sebgpt/model/mini_gpt.py",
    "src/sebgpt/data/__init__.py",
    "src/sebgpt/data/shakespeare_extract.py",
    "src/sebgpt/data/shakespeare_preflight.py",
    "src/sebgpt/data/shifted_examples.py",
    "src/sebgpt/data/shakespeare_examples.py",
    "src/sebgpt/model/phase4_corpus.py",
    "src/sebgpt/model/phase4_experiment.py",
    "src/sebgpt/training/phase8_types.py",
    "src/sebgpt/training/phase8_checkpoint.py",
)

TRACKED_IDENTITY_SHA256S = {
    "docs/TRAINING_CHECKPOINTING_SPEC.md": PHASE8_SPEC_SHA256,
    "docs/MINI_GPT_SPEC.md": MINI_GPT_SPEC_SHA256,
    "docs/data/shakespeare-eight-play-manifest.json": PHASE1_MANIFEST_SHA256,
    "requirements.lock": "8ddf7a11bf67d36bce8ba58abffc5416a0fb778cdc20d03c077d318715122cfa",
}

_TOP_LEVEL_KEYS = (
    "schema_version", "run", "authority", "configuration", "model_state",
    "optimizer_state", "progress", "random_state", "mode_state",
    "metric_state", "resume_lineage",
)
_RUN_KEYS = ("run_id", "logical_id", "completed_epoch", "code_commit")
_REFERENCE_KEYS = (
    "run_id", "role", "logical_id", "epoch", "validation_loss_hex", "sha256", "relative_path",
)
_CATALOG_KEYS = {"schema_version", "run_id", "latest", "best_validation"}
_AUTHORITY_KEYS = (
    "phase7_closure_commit", "phase7_contract_commit", "phase7_implementation_commit",
    "mini_gpt_spec_sha256", "mini_gpt_source_sha256", "mini_gpt_test_sha256",
    "phase8_contract_commit", "phase8_spec_sha256", "requirements_lock_sha256",
    "tokenizer", "dataset", "model", "runtime", "sealed_test_access",
)
_TOKENIZER_KEYS = (
    "implementation_commit", "tokenizer_id", "schema_version", "vocabulary_size",
    "artifact_path", "artifact_sha256",
)
_DATASET_KEYS = (
    "dataset_id", "manifest_sha256", "processing_manifest_sha256",
    "training_works", "validation_works",
)
_MODEL_KEYS = (
    "model_width", "max_sequence_length", "number_of_blocks", "number_of_heads",
    "head_width", "hidden_width", "layer_norm_epsilon", "dropout_probability",
    "embedding_seed", "block_parameter_seeds", "output_head_seed",
    "parameter_device", "parameter_dtype", "trainable_tensor_count", "parameter_count",
)
_CONFIG_KEYS = (
    "schema_version", "runtime", "context_length", "stride", "retain_unpadded_tail",
    "logical_batch_capacity", "allow_final_partial_batch", "maximum_epochs", "order_seed",
    "order_algorithm", "optimizer_name", "learning_rate", "betas", "epsilon",
    "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable",
    "fused", "scheduler_name", "warmup_steps", "gradient_clip_norm_type",
    "gradient_clip_max_norm", "gradient_clip_epsilon", "evaluate_initialized_state",
    "evaluation_interval_epochs", "evaluation_split_order", "best_comparison",
    "validation_early_stopping", "parameter_device", "parameter_dtype", "token_dtype",
    "checkpoint_schema_version", "catalog_schema_version", "maximum_catalog_bytes",
    "maximum_checkpoint_object_bytes",
)
_PROGRESS_KEYS = (
    "completed_epochs", "next_epoch", "next_example_offset", "optimizer_updates",
    "examples_processed", "targets_processed", "per_epoch_optimizer_updates",
    "per_epoch_example_counts", "per_epoch_target_counts", "next_ordering_action",
)
_METRIC_KEYS = (
    "initialized_training", "initialized_validation", "epoch_training",
    "epoch_validation", "current_training", "current_validation",
    "best_validation_loss", "best_validation_loss_hex", "best_validation_epoch",
    "best_logical_id", "best_comparison",
)
_EVALUATION_KEYS = ("split", "loss", "loss_hex", "target_count", "window_count")
_LINEAGE_KEYS = (
    "root_run_id", "resumed_from_checkpoint_sha256", "resume_count", "checkpoint_sha256s",
)
_DROPOUT_NAMES = tuple(
    f"blocks.{block}.{branch}_dropout"
    for block in range(4)
    for branch in ("attention", "feed_forward")
)
_RUNTIME_KEYS = (
    "schema_version", "python_version", "python_implementation", "torch_version",
    "operating_system", "operating_system_release", "machine", "processor",
    "device", "parameter_dtype", "token_dtype", "torch_intra_op_threads",
    "torch_inter_op_threads", "deterministic_algorithms", "deterministic_warn_only",
    "float32_matmul_precision", "default_dtype", "default_device",
    "grad_mode_enabled", "cpu_autocast_enabled", "inference_mode_enabled",
    "mkldnn_enabled", "torch_build_config_sha256", "torch_parallel_info_sha256",
    "requirements_lock_sha256",
)
_MODE_NAMES = (
    "", "representation", "blocks",
    *(name for block in range(4) for name in (
        f"blocks.{block}", f"blocks.{block}.norm1", f"blocks.{block}.attention",
        f"blocks.{block}.attention.heads",
        *(f"blocks.{block}.attention.heads.{head}" for head in range(4)),
        f"blocks.{block}.attention_dropout", f"blocks.{block}.norm2",
        f"blocks.{block}.feed_forward", f"blocks.{block}.feed_forward_dropout",
    )),
    "final_norm", "head",
)

ACCEPTED_RUNTIME_IDENTITY = {
    "schema_version": 1,
    "python_version": "3.14.4",
    "python_implementation": "CPython",
    "torch_version": "2.14.0",
    "operating_system": "Darwin",
    "operating_system_release": "25.6.0",
    "machine": "arm64",
    "processor": "arm",
    "device": "cpu",
    "parameter_dtype": "torch.float32",
    "token_dtype": "torch.long",
    "torch_intra_op_threads": 1,
    "torch_inter_op_threads": 1,
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "float32_matmul_precision": "highest",
    "default_dtype": "torch.float32",
    "default_device": "cpu",
    "grad_mode_enabled": True,
    "cpu_autocast_enabled": False,
    "inference_mode_enabled": False,
    "mkldnn_enabled": False,
    "torch_build_config_sha256": "7087d129a9b4d0dad0f49ccb08b46952ceac534033de37a2791e4a7870f75bd2",
    "torch_parallel_info_sha256": "f374b265f279a73db67700c37cada26235861d10ef91328bf12a992806c43b61",
    "requirements_lock_sha256": "8ddf7a11bf67d36bce8ba58abffc5416a0fb778cdc20d03c077d318715122cfa",
}


def _model_state_oracle() -> tuple[tuple[str, tuple[int, ...]], ...]:
    result: list[tuple[str, tuple[int, ...]]] = [
        ("representation.token_embeddings", (81, 32)),
        ("representation.position_embeddings", (256, 32)),
    ]
    for block in range(4):
        prefix = f"blocks.{block}"
        result.extend((
            (f"{prefix}.norm1.gamma", (32,)),
            (f"{prefix}.norm1.beta", (32,)),
            (f"{prefix}.attention.output_weight", (32, 32)),
        ))
        for head in range(4):
            for projection in ("query", "key", "value"):
                result.append((f"{prefix}.attention.heads.{head}.{projection}_weight", (32, 8)))
        result.extend((
            (f"{prefix}.norm2.gamma", (32,)), (f"{prefix}.norm2.beta", (32,)),
            (f"{prefix}.feed_forward.first_weight", (32, 128)),
            (f"{prefix}.feed_forward.first_bias", (128,)),
            (f"{prefix}.feed_forward.second_weight", (128, 32)),
            (f"{prefix}.feed_forward.second_bias", (32,)),
        ))
    result.extend((("final_norm.gamma", (32,)), ("final_norm.beta", (32,)), ("head.output_weight", (32, 81)), ("head.output_bias", (81,))))
    return tuple(result)


_MODEL_STATE_ORACLE = _model_state_oracle()


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    try:
        result = subprocess.run(
            ("git", *args), cwd=root, check=True, capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Phase9GovernanceError("phase9.governance.repository", field="record") from error
    return result.stdout if binary else result.stdout.strip()


def _sha_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise Phase9GovernanceError("phase9.governance.repository", field="protected_path") from None


def _validate_phase8_result_record(log: bytes) -> None:
    heading = "### EXP-20260912-01 — Completed Phase 8 fixed training\n".encode("utf-8")
    if log.count(heading) != 1:
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    start = log.index(heading)
    next_record = log.find(b"\n### EXP-", start + len(heading))
    end = len(log) if next_record < 0 else next_record
    if hashlib.sha256(log[start:end]).hexdigest() != PHASE8_RESULT_RECORD_SHA256:
        raise Phase9GovernanceError("phase9.governance.repository", field="record")


def _validate_repository(root: object) -> Path:
    if not isinstance(root, Path):
        raise Phase9TypeError("phase9.type.argument", field="repository_root")
    if not root.is_absolute():
        raise Phase9GovernanceError("phase9.governance.repository", field="repository_root")
    try:
        if root.is_symlink() or root.resolve(strict=True) != root:
            raise Phase9GovernanceError("phase9.governance.repository", field="repository_root")
    except OSError:
        raise Phase9GovernanceError("phase9.governance.repository", field="repository_root") from None
    if _git(root, "rev-parse", "--show-toplevel") != str(root):
        raise Phase9GovernanceError("phase9.governance.repository", field="repository_root")
    if _git(root, "branch", "--show-current") != "main":
        raise Phase9GovernanceError("phase9.governance.repository", field="branch")
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "rev-parse", "origin/main")
    if head != origin:
        raise Phase9GovernanceError("phase9.governance.repository", field="synchronization")
    if _git(root, "status", "--porcelain=v2", "--untracked-files=all") != "":
        raise Phase9GovernanceError("phase9.governance.repository", field="clean")
    for ancestor in (PHASE8_CLOSURE_COMMIT, PHASE8_RESULT_COMMIT, PHASE9_CONTRACT_COMMIT):
        try:
            subprocess.run(("git", "merge-base", "--is-ancestor", ancestor, str(head)), cwd=root, check=True)
        except (OSError, subprocess.CalledProcessError):
            raise Phase9GovernanceError("phase9.governance.repository", field="ancestor") from None
    for relative in PROTECTED_PATHS:
        try:
            live = (root / relative).read_bytes()
        except OSError:
            raise Phase9GovernanceError("phase9.governance.repository", field="protected_path") from None
        accepted = _git(root, "show", f"{PHASE8_CLOSURE_COMMIT}:{relative}", binary=True)
        if live != accepted:
            raise Phase9GovernanceError("phase9.governance.repository", field="protected_path")
    for relative, expected_sha256 in TRACKED_IDENTITY_SHA256S.items():
        if _sha_file(root / relative) != expected_sha256:
            raise Phase9GovernanceError(
                "phase9.governance.repository", field="protected_path"
            )
    if _sha_file(root / "docs/GENERATION_EVALUATION_SPEC.md") != PHASE9_SPEC_SHA256:
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    try:
        log = (root / "EXPERIMENT_LOG.md").read_bytes()
        log.decode("utf-8", errors="strict")
    except (OSError, UnicodeError):
        raise Phase9GovernanceError("phase9.governance.repository", field="record") from None
    _validate_phase8_result_record(log)
    _validate_runtime(root)
    return root


def _validate_runtime(root: Path) -> None:
    if _live_runtime_mapping(root) != ACCEPTED_RUNTIME_IDENTITY:
        raise Phase9GovernanceError("phase9.governance.runtime", field="runtime")


def _live_runtime_mapping(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": "cpu",
        "parameter_dtype": str(torch.float32),
        "token_dtype": "torch.long",
        "torch_intra_op_threads": torch.get_num_threads(),
        "torch_inter_op_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.get_default_device()),
        "grad_mode_enabled": torch.is_grad_enabled(),
        "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
        "inference_mode_enabled": torch.is_inference_mode_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_build_config_sha256": hashlib.sha256(torch.__config__.show().encode("utf-8")).hexdigest(),
        "torch_parallel_info_sha256": hashlib.sha256(torch.__config__.parallel_info().encode("utf-8")).hexdigest(),
        "requirements_lock_sha256": _sha_file(root / "requirements.lock"),
    }


def _canonical_catalog(value: dict[str, object]) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError):
        raise Phase9CheckpointError("phase9.checkpoint.catalog", field="catalog") from None


def _parse_catalog(content: bytes) -> dict[str, object]:
    if len(content) > CATALOG_LIMIT or not content:
        raise Phase9CheckpointError("phase9.checkpoint.catalog", field="catalog")
    try:
        value = json.loads(content.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Phase9CheckpointError("phase9.checkpoint.catalog", field="catalog") from None
    if type(value) is not dict or set(value) != _CATALOG_KEYS or _canonical_catalog(value) != content:
        raise Phase9CheckpointError("phase9.checkpoint.catalog", field="catalog")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="catalog")
    if type(value["run_id"]) is not str or not _RUN.fullmatch(value["run_id"]):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="catalog")
    return value


def _parse_reference(value: object, role: str, run_id: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(_REFERENCE_KEYS):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="reference")
    if (
        type(value["run_id"]) is not str or value["run_id"] != run_id
        or type(value["role"]) is not str or value["role"] != role
        or type(value["logical_id"]) is not str
        or type(value["epoch"]) is not int or not 1 <= value["epoch"] <= 10
        or value["logical_id"] != f"epoch-{value['epoch']:04d}"
        or type(value["validation_loss_hex"]) is not str
        or type(value["sha256"]) is not str or not _SHA.fullmatch(value["sha256"])
        or type(value["relative_path"]) is not str
        or value["relative_path"] != f"objects/{value['sha256']}.pt"
    ):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="reference")
    try:
        loss = float.fromhex(value["validation_loss_hex"])
    except ValueError:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="reference") from None
    if not math.isfinite(loss) or loss.hex() != value["validation_loss_hex"]:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="reference")
    return value


def _keys(value: object, expected: tuple[str, ...], field: str) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != expected:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field=field)
    return value


def _exact_type(value: object, expected: type[object]) -> bool:
    return type(value) is expected


def _digest(value: object) -> bool:
    return type(value) is str and bool(_SHA.fullmatch(value))


def _validate_runtime_mapping(value: object) -> dict[str, object]:
    runtime = _keys(value, _RUNTIME_KEYS, "runtime")
    expected_types: dict[str, type[object]] = {
        "schema_version": int, "python_version": str, "python_implementation": str,
        "torch_version": str, "operating_system": str, "operating_system_release": str,
        "machine": str, "processor": str, "device": str, "parameter_dtype": str,
        "token_dtype": str, "torch_intra_op_threads": int, "torch_inter_op_threads": int,
        "deterministic_algorithms": bool, "deterministic_warn_only": bool,
        "float32_matmul_precision": str, "default_dtype": str, "default_device": str,
        "grad_mode_enabled": bool, "cpu_autocast_enabled": bool,
        "inference_mode_enabled": bool, "mkldnn_enabled": bool,
        "torch_build_config_sha256": str, "torch_parallel_info_sha256": str,
        "requirements_lock_sha256": str,
    }
    fixed = {
        "schema_version": 1, "python_version": "3.14.4", "python_implementation": "CPython",
        "torch_version": "2.14.0", "operating_system": "Darwin", "machine": "arm64",
        "device": "cpu", "parameter_dtype": "torch.float32", "token_dtype": "torch.long",
        "torch_intra_op_threads": 1, "torch_inter_op_threads": 1,
        "deterministic_algorithms": True, "deterministic_warn_only": False,
        "float32_matmul_precision": "highest", "default_dtype": "torch.float32",
        "default_device": "cpu", "grad_mode_enabled": True, "cpu_autocast_enabled": False,
        "inference_mode_enabled": False, "mkldnn_enabled": False,
    }
    if any(not _exact_type(runtime[name], expected) for name, expected in expected_types.items()):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="runtime")
    if any(runtime[name] != expected for name, expected in fixed.items()):
        raise Phase9CheckpointError("phase9.checkpoint.provenance", field="runtime")
    if any(not _digest(runtime[name]) for name in ("torch_build_config_sha256", "torch_parallel_info_sha256", "requirements_lock_sha256")):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="runtime")
    return runtime


def _validate_evaluation(value: object, split: str) -> dict[str, object]:
    evaluation = _keys(value, _EVALUATION_KEYS, "metric")
    if (
        not _exact_type(evaluation["split"], str) or evaluation["split"] != split
        or not _exact_type(evaluation["loss"], float) or not math.isfinite(evaluation["loss"])
        or evaluation["loss"] < 0.0
        or not _exact_type(evaluation["loss_hex"], str)
        or evaluation["loss_hex"] != evaluation["loss"].hex()
        or not _exact_type(evaluation["target_count"], int) or evaluation["target_count"] <= 0
        or not _exact_type(evaluation["window_count"], int) or evaluation["window_count"] <= 0
    ):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="metric")
    return evaluation


def _validate_authority(value: object) -> dict[str, object]:
    authority = _keys(value, _AUTHORITY_KEYS, "authority")
    fixed = {
        "phase7_closure_commit": PHASE7_CLOSURE_COMMIT,
        "phase7_contract_commit": PHASE7_CONTRACT_COMMIT,
        "phase7_implementation_commit": PHASE7_IMPLEMENTATION_COMMIT,
        "mini_gpt_spec_sha256": MINI_GPT_SPEC_SHA256,
        "mini_gpt_source_sha256": MINI_GPT_SOURCE_SHA256,
        "mini_gpt_test_sha256": MINI_GPT_TEST_SHA256,
        "phase8_contract_commit": PHASE8_CONTRACT_COMMIT,
        "phase8_spec_sha256": PHASE8_SPEC_SHA256,
        "sealed_test_access": "none",
    }
    if any(not _exact_type(authority[name], str) or authority[name] != expected for name, expected in fixed.items()):
        raise Phase9CheckpointError("phase9.checkpoint.provenance", field="authority")
    if not _digest(authority["requirements_lock_sha256"]):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="authority")
    tokenizer = _keys(authority["tokenizer"], _TOKENIZER_KEYS, "tokenizer")
    expected_tokenizer = {
        "implementation_commit": TOKENIZER_IMPLEMENTATION_COMMIT,
        "tokenizer_id": "shakespeare-code-point-v1", "schema_version": 1,
        "vocabulary_size": 81,
        "artifact_path": "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
        "artifact_sha256": VOCABULARY_SHA256,
    }
    if any(not _exact_type(tokenizer[name], type(expected)) or tokenizer[name] != expected for name, expected in expected_tokenizer.items()):
        raise Phase9CheckpointError("phase9.checkpoint.provenance", field="tokenizer")
    dataset = _keys(authority["dataset"], _DATASET_KEYS, "dataset")
    if (
        dataset["dataset_id"] != "shakespeare-eight-play"
        or dataset["manifest_sha256"] != PHASE1_MANIFEST_SHA256
        or dataset["processing_manifest_sha256"] != PROCESSING_MANIFEST_SHA256
        or type(dataset["training_works"]) is not tuple or len(dataset["training_works"]) != 6
        or type(dataset["validation_works"]) is not tuple or len(dataset["validation_works"]) != 1
    ):
        raise Phase9CheckpointError("phase9.checkpoint.provenance", field="dataset")
    expected_works = ((1, "hamlet", "train"), (2, "romeo-and-juliet", "train"), (3, "macbeth", "train"), (4, "a-midsummer-nights-dream", "train"), (5, "much-ado-about-nothing", "train"), (6, "henry-v", "train"), (7, "the-tempest", "validation"))
    for work, expected in zip((*dataset["training_works"], *dataset["validation_works"]), expected_works, strict=True):
        work = _keys(work, ("manifest_order", "work_id", "split", "processed_sha256"), "work")
        if (
            type(work["manifest_order"]) is not int
            or type(work["work_id"]) is not str
            or type(work["split"]) is not str
            or (work["manifest_order"], work["work_id"], work["split"]) != expected
            or not _digest(work["processed_sha256"])
        ):
            raise Phase9CheckpointError("phase9.checkpoint.provenance", field="work")
    model = _keys(authority["model"], _MODEL_KEYS, "model")
    expected_model = {"model_width": 32, "max_sequence_length": 256, "number_of_blocks": 4, "number_of_heads": 4, "head_width": 8, "hidden_width": 128, "layer_norm_epsilon": 1e-5, "dropout_probability": 0.1, "embedding_seed": 1337, "block_parameter_seeds": (7001, 7002, 7003, 7004), "output_head_seed": 7005, "parameter_device": "cpu", "parameter_dtype": "torch.float32", "trainable_tensor_count": 90, "parameter_count": 63_825}
    if any(not _exact_type(model[name], type(expected)) or model[name] != expected for name, expected in expected_model.items()):
        raise Phase9CheckpointError("phase9.checkpoint.provenance", field="model")
    runtime = _validate_runtime_mapping(authority["runtime"])
    if authority["requirements_lock_sha256"] != runtime["requirements_lock_sha256"]:
        raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="runtime")
    return authority


def _validate_payload(
    payload: object,
    *,
    expected_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    if type(payload) is not dict or tuple(payload) != _TOP_LEVEL_KEYS:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="payload")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="payload")
    run = _keys(payload["run"], _RUN_KEYS, "run")
    if (
        type(run.get("run_id")) is not str or run.get("run_id") != RUN_ID
        or type(run.get("logical_id")) is not str
        or type(run.get("completed_epoch")) is not int
        or not 1 <= run["completed_epoch"] <= 10
        or run.get("logical_id") != f"epoch-{run['completed_epoch']:04d}"
        or type(run.get("code_commit")) is not str
        or run.get("code_commit") != PHASE8_IMPLEMENTATION_COMMIT
    ):
        raise Phase9CheckpointError("phase9.checkpoint.provenance", field="run")
    authority = _validate_authority(payload["authority"])
    configuration = _keys(payload["configuration"], _CONFIG_KEYS, "configuration")
    expected_configuration = {
        "schema_version": 1, "context_length": 256, "stride": 256,
        "retain_unpadded_tail": True, "logical_batch_capacity": 8,
        "allow_final_partial_batch": True, "maximum_epochs": 10, "order_seed": 8001,
        "order_algorithm": "torch_randperm_local_cpu_generator", "optimizer_name": "AdamW",
        "learning_rate": 3e-4, "betas": (0.9, 0.999), "epsilon": 1e-8,
        "weight_decay": 0.01, "amsgrad": False, "maximize": False, "foreach": False,
        "capturable": False, "differentiable": False, "fused": False,
        "scheduler_name": None, "warmup_steps": 0, "gradient_clip_norm_type": 2.0,
        "gradient_clip_max_norm": 1.0, "gradient_clip_epsilon": 1e-6,
        "evaluate_initialized_state": True, "evaluation_interval_epochs": 1,
        "evaluation_split_order": ("train", "validation"), "best_comparison": "strict_lower",
        "validation_early_stopping": False, "parameter_device": "cpu",
        "parameter_dtype": "torch.float32", "token_dtype": "torch.long",
        "checkpoint_schema_version": 1, "catalog_schema_version": 1,
        "maximum_catalog_bytes": CATALOG_LIMIT, "maximum_checkpoint_object_bytes": OBJECT_LIMIT,
    }
    if any(not _exact_type(configuration[name], type(expected)) or configuration[name] != expected for name, expected in expected_configuration.items()):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="configuration")
    _validate_runtime_mapping(configuration["runtime"])
    if configuration["runtime"] != authority["runtime"]:
        raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="runtime")
    if expected_runtime is not None and configuration["runtime"] != expected_runtime:
        raise Phase9CheckpointError("phase9.checkpoint.provenance", field="runtime")
    model_state = payload["model_state"]
    if type(model_state) is not OrderedDict or tuple(model_state) != tuple(name for name, _ in _MODEL_STATE_ORACLE):
        raise Phase9CheckpointError("phase9.checkpoint.model_state", field="model")
    for (name, shape), tensor in zip(_MODEL_STATE_ORACLE, model_state.values(), strict=True):
        if type(name) is not str or type(tensor) is not torch.Tensor or tuple(tensor.shape) != shape or tensor.device != torch.device("cpu") or tensor.dtype is not torch.float32 or not bool(torch.all(torch.isfinite(tensor)).item()):
            raise Phase9CheckpointError("phase9.checkpoint.model_state", field="model")
    optimizer = payload["optimizer_state"]
    if type(optimizer) is not dict or tuple(optimizer) != ("state", "param_groups") or type(optimizer["state"]) is not dict or type(optimizer["param_groups"]) is not list or len(optimizer["param_groups"]) != 1:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="optimizer")
    progress = _keys(payload["progress"], _PROGRESS_KEYS, "progress")
    integer_fields = ("completed_epochs", "next_epoch", "next_example_offset", "optimizer_updates", "examples_processed", "targets_processed")
    if any(not _exact_type(progress[name], int) for name in integer_fields) or any(progress[name] <= 0 for name in ("optimizer_updates", "examples_processed", "targets_processed")) or progress["completed_epochs"] != run["completed_epoch"] or progress["next_epoch"] != run["completed_epoch"] + 1 or progress["next_example_offset"] != 0 or type(progress["next_ordering_action"]) is not str or progress["next_ordering_action"] != "draw_next_epoch_permutation":
        raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="progress")
    for name, total in (("per_epoch_optimizer_updates", "optimizer_updates"), ("per_epoch_example_counts", "examples_processed"), ("per_epoch_target_counts", "targets_processed")):
        values = progress[name]
        if type(values) is not tuple or len(values) != run["completed_epoch"] or any(type(item) is not int or item <= 0 for item in values) or sum(values) != progress[total]:
            raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="progress")
    group = _keys(optimizer["param_groups"][0], ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused", "decoupled_weight_decay", "params"), "optimizer")
    expected_group = {"lr": 3e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False, "maximize": False, "foreach": False, "capturable": False, "differentiable": False, "fused": False, "decoupled_weight_decay": True, "params": list(range(90))}
    if any(not _exact_type(group[name], type(expected)) or group[name] != expected for name, expected in expected_group.items()) or tuple(optimizer["state"]) != tuple(range(90)) or any(type(key) is not int for key in optimizer["state"]):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="optimizer")
    for position, (_, shape) in enumerate(_MODEL_STATE_ORACLE):
        state = _keys(optimizer["state"][position], ("step", "exp_avg", "exp_avg_sq"), "optimizer")
        step = state["step"]
        if type(step) is not torch.Tensor or step.shape != torch.Size([]) or step.device != torch.device("cpu") or step.dtype is not torch.float32 or not math.isfinite(float(step.item())) or not float(step.item()).is_integer() or int(step.item()) != progress["optimizer_updates"]:
            raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="optimizer")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state[name]
            if type(tensor) is not torch.Tensor or tuple(tensor.shape) != shape or tensor.device != torch.device("cpu") or tensor.dtype is not torch.float32 or not bool(torch.all(torch.isfinite(tensor)).item()):
                raise Phase9CheckpointError("phase9.checkpoint.schema", field="optimizer")
    random_state = payload["random_state"]
    if type(random_state) is not dict or tuple(random_state) != ("order_generator", "global_cpu", "dropout_generators"):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="rng")
    for name in ("order_generator", "global_cpu"):
        value = random_state[name]
        if type(value) is not torch.Tensor or value.device != torch.device("cpu") or value.dtype is not torch.uint8 or value.dim() != 1:
            raise Phase9CheckpointError("phase9.checkpoint.schema", field="rng")
    if type(random_state["dropout_generators"]) is not dict or tuple(random_state["dropout_generators"]) != _DROPOUT_NAMES:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="rng")
    for value in random_state["dropout_generators"].values():
        if type(value) is not torch.Tensor or value.device != torch.device("cpu") or value.dtype is not torch.uint8 or value.dim() != 1:
            raise Phase9CheckpointError("phase9.checkpoint.schema", field="rng")
    expected_seeds = (
        8001,
        None,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
        ATTENTION_DROPOUT_SEED,
        FEED_FORWARD_DROPOUT_SEED,
    )
    for state, seed in zip(
        (random_state["order_generator"], random_state["global_cpu"], *random_state["dropout_generators"].values()),
        expected_seeds,
        strict=True,
    ):
        generator = torch.Generator(device="cpu")
        try:
            generator.set_state(state)
        except BaseException as error:
            raise Phase9CheckpointError("phase9.checkpoint.schema", field="rng") from error
        if seed is not None and generator.initial_seed() != seed:
            raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="rng")
    mode = payload["mode_state"]
    if type(mode) is not dict or tuple(mode) != ("training", "named_modules") or mode["training"] is not True or mode["named_modules"] != tuple((name, True) for name in _MODE_NAMES):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="mode")
    metrics = _keys(payload["metric_state"], _METRIC_KEYS, "metric")
    initialized_training = _validate_evaluation(metrics["initialized_training"], "train")
    initialized_validation = _validate_evaluation(metrics["initialized_validation"], "validation")
    for name, split in (("epoch_training", "train"), ("epoch_validation", "validation")):
        sequence = metrics[name]
        if type(sequence) is not tuple or len(sequence) != run["completed_epoch"]:
            raise Phase9CheckpointError("phase9.checkpoint.schema", field="metric")
        for value in sequence:
            _validate_evaluation(value, split)
    current_training = _validate_evaluation(metrics["current_training"], "train")
    current_validation = _validate_evaluation(metrics["current_validation"], "validation")
    if (
        current_training != metrics["epoch_training"][-1]
        or current_validation != metrics["epoch_validation"][-1]
        or any(
            (value["target_count"], value["window_count"])
            != (initialized_training["target_count"], initialized_training["window_count"])
            for value in metrics["epoch_training"]
        )
        or any(
            (value["target_count"], value["window_count"])
            != (initialized_validation["target_count"], initialized_validation["window_count"])
            for value in metrics["epoch_validation"]
        )
    ):
        raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="metric")
    validation_losses = tuple(value["loss"] for value in metrics["epoch_validation"])
    best_loss = min(validation_losses)
    best_epoch = validation_losses.index(best_loss) + 1
    if (
        not _exact_type(metrics["best_validation_loss"], float)
        or not math.isfinite(metrics["best_validation_loss"])
        or not _exact_type(metrics["best_validation_loss_hex"], str)
        or not _exact_type(metrics["best_validation_epoch"], int)
        or not _exact_type(metrics["best_logical_id"], str)
        or not _exact_type(metrics["best_comparison"], str)
        or metrics["best_validation_loss"] != best_loss
        or metrics["best_validation_loss_hex"] != best_loss.hex()
        or metrics["best_validation_epoch"] != best_epoch
        or metrics["best_logical_id"] != f"epoch-{best_epoch:04d}"
        or metrics["best_comparison"] != "strict_lower"
    ):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="metric")
    for index, evaluation in enumerate(metrics["epoch_training"]):
        if progress["per_epoch_example_counts"][index] != evaluation["window_count"] or progress["per_epoch_target_counts"][index] != evaluation["target_count"] or progress["per_epoch_optimizer_updates"][index] != math.ceil(evaluation["window_count"] / 8):
            raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="metric")
    if (
        progress["examples_processed"]
        != initialized_training["window_count"] * run["completed_epoch"]
        or progress["targets_processed"]
        != initialized_training["target_count"] * run["completed_epoch"]
    ):
        raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="metric")
    lineage = _keys(payload["resume_lineage"], _LINEAGE_KEYS, "lineage")
    hashes = lineage["checkpoint_sha256s"]
    if type(lineage["root_run_id"]) is not str or lineage["root_run_id"] != RUN_ID or type(lineage["resume_count"]) is not int or type(hashes) is not tuple or lineage["resume_count"] != len(hashes) or any(not _digest(value) for value in hashes) or (not hashes and lineage["resumed_from_checkpoint_sha256"] is not None) or (hashes and (type(lineage["resumed_from_checkpoint_sha256"]) is not str or lineage["resumed_from_checkpoint_sha256"] != hashes[-1])):
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="lineage")
    return payload


def _load_payload(
    content: bytes,
    *,
    expected_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    if not content or len(content) > OBJECT_LIMIT:
        raise Phase9CheckpointError("phase9.checkpoint.path", field="checkpoint")
    try:
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except Exception as error:
        raise Phase9CheckpointError("phase9.checkpoint.schema", field="payload") from error
    return _validate_payload(payload, expected_runtime=expected_runtime)


def _validate_graph(
    catalog_content: bytes,
    object_reader: Callable[[dict[str, object]], bytes],
    *,
    expected_catalog_sha256: str | None,
    expected_checkpoint_sha256: str | None,
    expected_runtime: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if expected_catalog_sha256 is not None and hashlib.sha256(catalog_content).hexdigest() != expected_catalog_sha256:
        raise Phase9CheckpointError("phase9.checkpoint.hash", field="catalog")
    catalog = _parse_catalog(catalog_content)
    run_id = catalog["run_id"]
    assert isinstance(run_id, str)
    latest = _parse_reference(catalog["latest"], "latest", run_id)
    best = _parse_reference(catalog["best_validation"], "best_validation", run_id)
    if expected_checkpoint_sha256 is not None and best["sha256"] != expected_checkpoint_sha256:
        raise Phase9CheckpointError("phase9.checkpoint.hash", field="checkpoint")
    payloads: dict[str, dict[str, object]] = {}
    for reference in (latest, best):
        digest = reference["sha256"]
        assert isinstance(digest, str)
        if digest not in payloads:
            content = object_reader(reference)
            if hashlib.sha256(content).hexdigest() != digest:
                raise Phase9CheckpointError("phase9.checkpoint.hash", field="checkpoint")
            payloads[digest] = _load_payload(content, expected_runtime=expected_runtime)
        payload = payloads[digest]
        run = payload["run"]
        metrics = payload["metric_state"]
        assert isinstance(run, dict) and isinstance(metrics, dict)
        if run["run_id"] != run_id or run["logical_id"] != reference["logical_id"] or run["completed_epoch"] != reference["epoch"]:
            raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="reference")
        expected_loss = metrics.get("current_validation", {}).get("loss_hex") if isinstance(metrics.get("current_validation"), dict) else None
        if expected_loss != reference["validation_loss_hex"]:
            raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="reference")
    if best["epoch"] > latest["epoch"]:
        raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="reference")
    latest_metrics = payloads[latest["sha256"]]["metric_state"]  # type: ignore[index]
    best_metrics = payloads[best["sha256"]]["metric_state"]  # type: ignore[index]
    assert isinstance(latest_metrics, dict) and isinstance(best_metrics, dict)
    if (
        latest_metrics["best_logical_id"] != best["logical_id"]
        or latest_metrics["best_validation_epoch"] != best["epoch"]
        or latest_metrics["best_validation_loss_hex"] != best["validation_loss_hex"]
        or best_metrics["best_logical_id"] != best["logical_id"]
        or best_metrics["best_validation_epoch"] != best["epoch"]
        or best_metrics["best_validation_loss_hex"] != best["validation_loss_hex"]
    ):
        raise Phase9CheckpointError("phase9.checkpoint.cross_field", field="reference")
    return best, payloads[best["sha256"]]  # type: ignore[index]


def _read_fd(
    directory_fd: int,
    name: str,
    maximum: int,
    *,
    retained_entries: list[tuple[int, str, tuple[int, int, int, int]]] | None = None,
) -> bytes:
    fd: int | None = None
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise OSError
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(fd, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(fd)
        if (
            len(content) != before.st_size
            or (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mode, after.st_size)
        ):
            raise OSError
        if retained_entries is not None:
            retained_entries.append(
                (
                    directory_fd,
                    name,
                    (before.st_dev, before.st_ino, before.st_mode, before.st_size),
                )
            )
        return content
    except OSError:
        raise Phase9CheckpointError("phase9.checkpoint.path", field="checkpoint") from None
    finally:
        if fd is not None:
            os.close(fd)


def _open_directory(parent_fd: int | None, path: str | Path) -> tuple[int, tuple[int, int, int]]:
    fd: int | None = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(path, flags) if parent_fd is None else os.open(path, flags, dir_fd=parent_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError
        return fd, (metadata.st_dev, metadata.st_ino, metadata.st_mode)
    except OSError:
        if fd is not None:
            os.close(fd)
        raise Phase9CheckpointError("phase9.checkpoint.path", field="checkpoint") from None


def _revalidate_directories(opened: tuple[tuple[int, tuple[int, int, int]], ...]) -> None:
    try:
        for fd, identity in opened:
            metadata = os.fstat(fd)
            if (metadata.st_dev, metadata.st_ino, metadata.st_mode) != identity:
                raise OSError
    except OSError:
        raise Phase9CheckpointError("phase9.checkpoint.path", field="checkpoint") from None


def _revalidate_entries(
    directory_entries: tuple[tuple[int, str, tuple[int, int, int]], ...],
    file_entries: tuple[tuple[int, str, tuple[int, int, int, int]], ...],
) -> None:
    try:
        for parent_fd, name, identity in directory_entries:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino, metadata.st_mode) != identity
            ):
                raise OSError
        for parent_fd, name, identity in file_entries:
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size)
                != identity
            ):
                raise OSError
    except OSError:
        raise Phase9CheckpointError("phase9.checkpoint.path", field="checkpoint") from None


def _construct_model(vocabulary: object) -> MiniGPT:
    return MiniGPT(
        vocabulary,  # type: ignore[arg-type]
        model_width=32, max_sequence_length=256, number_of_blocks=4,
        number_of_heads=4, head_width=8, hidden_width=128,
        layer_norm_epsilon=1e-5, dropout_probability=0.1,
        embedding_seed=1337, block_parameter_seeds=(7001, 7002, 7003, 7004),
        output_head_seed=7005, device=torch.device("cpu"), dtype=torch.float32,
    )


def _load_validated_artifacts(
    root: Path,
    *,
    expected_catalog_sha256: str = CATALOG_SHA256,
    expected_checkpoint_sha256: str = CHECKPOINT_SHA256,
    expected_runtime: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    opened: list[tuple[int, tuple[int, int, int]]] = []
    directory_entries: list[tuple[int, str, tuple[int, int, int]]] = []
    file_entries: list[tuple[int, str, tuple[int, int, int, int]]] = []
    try:
        root_fd, identity = _open_directory(None, root)
        opened.append((root_fd, identity))
        parent = root_fd
        for component in ("checkpoints", "phase8", RUN_ID):
            parent_fd = parent
            parent, identity = _open_directory(parent_fd, component)
            opened.append((parent, identity))
            directory_entries.append((parent_fd, component, identity))
        run_fd = parent
        objects_fd, identity = _open_directory(run_fd, "objects")
        opened.append((objects_fd, identity))
        directory_entries.append((run_fd, "objects", identity))
        catalog_content = _read_fd(
            run_fd,
            "checkpoint-catalog.json",
            CATALOG_LIMIT,
            retained_entries=file_entries,
        )

        def reader(reference: dict[str, object]) -> bytes:
            digest = reference["sha256"]
            assert isinstance(digest, str)
            return _read_fd(
                objects_fd,
                f"{digest}.pt",
                OBJECT_LIMIT,
                retained_entries=file_entries,
            )

        result = _validate_graph(
            catalog_content,
            reader,
            expected_catalog_sha256=expected_catalog_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_runtime=(
                _live_runtime_mapping(root)
                if expected_runtime is None
                else expected_runtime
            ),
        )
        _revalidate_directories(tuple(opened))
        _revalidate_entries(tuple(directory_entries), tuple(file_entries))
        try:
            root_metadata = os.stat(root, follow_symlinks=False)
        except OSError:
            raise Phase9CheckpointError(
                "phase9.checkpoint.path", field="checkpoint"
            ) from None
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (root_metadata.st_dev, root_metadata.st_ino, root_metadata.st_mode)
            != opened[0][1]
        ):
            raise Phase9CheckpointError("phase9.checkpoint.path", field="checkpoint")
        return result
    finally:
        for fd, _ in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass


def load_phase9_inference_bundle(repository_root: Path) -> Phase9InferenceBundle:
    global_rng = torch.get_rng_state().clone()
    try:
        root = _validate_repository(repository_root)
        reference, payload = _load_validated_artifacts(root)
        if (
            reference["role"] != "best_validation"
            or reference["logical_id"] != "epoch-0010"
            or reference["epoch"] != 10
            or reference["validation_loss_hex"] != VALIDATION_LOSS_HEX
            or reference["sha256"] != CHECKPOINT_SHA256
        ):
            raise Phase9CheckpointError("phase9.checkpoint.provenance", field="reference")
        vocabulary = load_accepted_vocabulary_binding(root)
        tokenizer = CodePointTokenizer(vocabulary.code_points)
        model = _construct_model(vocabulary)
        dropout_before = tuple(
            generator.get_state().clone()
            for block in model.blocks
            for generator in (
                block.attention_dropout._generator,
                block.feed_forward_dropout._generator,
            )
        )
        model_state = payload["model_state"]
        assert isinstance(model_state, OrderedDict)
        expected = tuple(model.state_dict())
        if tuple(model_state) != expected:
            raise Phase9CheckpointError("phase9.checkpoint.model_state", field="model")
        for name, target in model.state_dict().items():
            source = model_state[name]
            if tuple(source.shape) != tuple(target.shape):
                raise Phase9CheckpointError("phase9.checkpoint.model_state", field="model")
        model.load_state_dict(model_state, strict=True)
        model.eval()
        named_parameters = tuple(model.named_parameters())
        dropout_after = tuple(
            generator.get_state().clone()
            for block in model.blocks
            for generator in (
                block.attention_dropout._generator,
                block.feed_forward_dropout._generator,
            )
        )
        if (
            tuple(name for name, _ in named_parameters)
            != tuple(name for name, _ in _MODEL_STATE_ORACLE)
            or any(
                tuple(parameter.shape) != shape
                or parameter.device != torch.device("cpu")
                or parameter.dtype is not torch.float32
                or parameter.requires_grad is not True
                or parameter.grad is not None
                or not bool(torch.all(torch.isfinite(parameter)).item())
                or not torch.equal(parameter.detach(), model_state[name])
                for (name, shape), (_, parameter) in zip(
                    _MODEL_STATE_ORACLE, named_parameters, strict=True
                )
            )
            or len({id(parameter) for _, parameter in named_parameters}) != 90
            or len({parameter.data_ptr() for _, parameter in named_parameters}) != 90
            or tuple(name for name, _ in model.named_modules()) != _MODE_NAMES
            or any(module.training for _, module in model.named_modules())
            or any(
                not torch.equal(before, after)
                for before, after in zip(dropout_before, dropout_after, strict=True)
            )
        ):
            raise Phase9ContractError("phase9.contract.bundle", field="model")
        _validate_phase9_minigpt_structure(model)
        identity = Phase9CheckpointIdentity(
            run_id=RUN_ID,
            role="best_validation",
            logical_id="epoch-0010",
            epoch=10,
            validation_loss_hex=VALIDATION_LOSS_HEX,
            checkpoint_sha256=CHECKPOINT_SHA256,
            checkpoint_relative_path=f"objects/{CHECKPOINT_SHA256}.pt",
            catalog_sha256=CATALOG_SHA256,
        )
        result = _make_inference_bundle(
            model=model, tokenizer=tokenizer, vocabulary=vocabulary, checkpoint=identity
        )
        if not torch.equal(torch.get_rng_state(), global_rng):
            raise Phase9ContractError("phase9.contract.bundle", field="rng")
        return result
    finally:
        if not torch.equal(torch.get_rng_state(), global_rng):
            torch.set_rng_state(global_rng)


__all__ = ["load_phase9_inference_bundle"]
