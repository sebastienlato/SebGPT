"""Governed Phase 9 evaluation orchestration and durable evidence lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
import fcntl
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from sebgpt.inference.phase9_checkpoint import (
    CHECKPOINT_SHA256,
    PHASE9_CONTRACT_COMMIT,
    PHASE9_SPEC_SHA256,
    load_phase9_inference_bundle,
    _git,
    _live_runtime_mapping,
    _validate_repository,
)
from sebgpt.inference.phase9_evaluation import (
    _evaluate_laplace_internal,
    _evaluate_model_internal,
    _evaluate_uniform_internal,
    build_phase9_windows,
    evaluate_phase9_laplace_unigram,
    evaluate_phase9_model,
    evaluate_phase9_uniform,
    fit_phase9_laplace_unigram,
)
from sebgpt.inference.phase9_generation import generate_phase9_text
from sebgpt.inference.phase9_types import (
    Phase9CheckpointError,
    Phase9ContractError,
    Phase9EvidenceReference,
    Phase9GovernanceError,
    Phase9Metrics,
    Phase9NumericalError,
    Phase9PublicationError,
    Phase9TokenDocument,
    Phase9TypeError,
    Phase9Window,
    _MODEL_PARAMETER_ORACLE,
    _make_laplace_unigram,
)
from sebgpt.model.phase4_corpus import load_phase4_permitted_corpus
from sebgpt.data.shifted_examples import (
    Phase4ContractError,
    Phase4GovernanceError,
    Phase4TypeError,
)
from sebgpt.model.mini_gpt import MiniGPTContractError, MiniGPTNumericalError, MiniGPTTypeError
from sebgpt.tokenization.code_point import (
    InvalidTokenIdError,
    TokenizationTypeError,
    TokenizerStateError,
    UnknownCodePointError,
)


ID_PATTERN = re.compile(r"EXP-[0-9]{8}-[0-9]{2}\Z")
HEADING_PATTERN = re.compile(
    r"### (EXP-[0-9]{8}-[0-9]{2}) — Phase 9 "
    r"(development plan|development authorization|development result|"
    r"development supersession|sealed-test plan|sealed-test authorization|"
    r"sealed-test result)\n\Z"
)
FIELD_PATTERN = re.compile(r"- \*\*([^*]+):\*\* (.+)\n\Z")

NOT_PUBLISHED = "not_published"
KNOWN_ABSENT = "known_absent"
KNOWN_PRESENT = "known_present"
UNKNOWN = "unknown"

_EVIDENCE_ROWS = {
    ("phase9_development_attempt", "consumed"): ("phase9-development-attempt.json", 16_384),
    ("phase9_development", "completed"): ("phase9-development-evidence.json", 1_048_576),
    ("phase9_development_failure", "failed"): ("phase9-development-failure.json", 1_048_576),
    ("phase9_sealed_test_access", "consumed"): ("phase9-sealed-test-access.json", 16_384),
    ("phase9_sealed_test", "completed"): ("phase9-sealed-test-evidence.json", 262_144),
    ("phase9_sealed_test_failure", "failed"): ("phase9-sealed-test-failure.json", 262_144),
}

_KIND_BY_SUFFIX = {
    "development plan": "development_plan",
    "development authorization": "development_authorization",
    "development result": "development_result",
    "development supersession": "development_supersession",
    "sealed-test plan": "sealed_test_plan",
    "sealed-test authorization": "sealed_test_authorization",
    "sealed-test result": "sealed_test_result",
}

_LABELS = {
    "development_plan": (
        "Evaluation ID", "Entry kind", "Scope", "Status", "Question",
        "Predecessor evaluation", "Phase 8 authority", "Checkpoint authority",
        "Phase 9 contract authority", "Phase 9 implementation authority",
        "Runtime identity", "Dataset authority", "Tokenizer authority",
        "Model authority", "Evaluation configuration", "Qualitative matrix",
        "Attempt marker policy", "Evidence policy", "Sealed-test access",
        "Success policy", "Next gate",
    ),
    "development_authorization": (
        "Evaluation ID", "Entry kind", "Scope", "Status",
        "Development plan record SHA-256", "Development pre-registration commit",
        "Phase 9 implementation commit", "Checkpoint SHA-256", "Authorization", "Next gate",
    ),
    "development_result": (
        "Evaluation ID", "Entry kind", "Scope", "Status", "Recorded at UTC",
        "Started at UTC", "Finished at UTC", "Predecessor evaluation",
        "Development plan record SHA-256", "Development pre-registration commit",
        "Development authorization record SHA-256", "Development authorization commit",
        "Attempt marker", "Development evidence", "Model validation",
        "Uniform validation", "Laplace unigram baseline",
        "Laplace unigram validation", "Qualitative samples", "Sealed-test access",
        "Failure", "Observations", "Conclusions", "Limitations", "Next gate",
    ),
    "development_supersession": (
        "Evaluation ID", "Entry kind", "Scope", "Status", "Prior terminal status",
        "Prior result record SHA-256", "Successor evaluation ID", "Reason",
        "Recorded at UTC", "Next gate",
    ),
    "sealed_test_plan": (
        "Evaluation ID", "Entry kind", "Scope", "Status",
        "Accepted development result commit", "Development result record SHA-256",
        "Development evidence SHA-256", "Frozen authority",
        "Frozen evaluation configuration", "Frozen qualitative matrix",
        "Access marker policy", "Sealed evidence policy", "Interpretation policy",
        "Sealed-test access", "Next gate",
    ),
    "sealed_test_authorization": (
        "Evaluation ID", "Entry kind", "Scope", "Status", "Sealed plan record SHA-256",
        "Sealed pre-registration commit", "Development evidence SHA-256",
        "Phase 9 implementation commit", "Checkpoint SHA-256", "Authorization", "Next gate",
    ),
    "sealed_test_result": (
        "Evaluation ID", "Entry kind", "Scope", "Status", "Recorded at UTC",
        "Started at UTC", "Finished at UTC", "Sealed plan record SHA-256",
        "Sealed pre-registration commit", "Sealed authorization record SHA-256",
        "Sealed authorization commit", "Access marker", "Sealed evidence",
        "Model test", "Uniform test", "Laplace unigram test", "Sealed-test access",
        "Failure", "Observations", "Conclusions", "Limitations", "Next gate",
    ),
}

LIMITATIONS = (
    "small character-level Shakespeare corpus",
    "CPU-float32 model with maximum 256-token context",
    "teacher-forced metrics do not measure free-running coherence",
    "qualitative samples are fixed observations, not broad generalization evidence",
    "rolling context discards history older than 256 tokens",
    "no BOS, EOS, unknown, or padding token",
    "validation guided development; sealed test remains separately gated and one-shot",
)

_FAILURE_FIELD_VOCABULARY = {
    "repository_root", "evaluation_id", "branch", "synchronization", "clean",
    "ancestor", "protected_path", "record", "configuration", "runtime", "catalog",
    "reference", "checkpoint", "payload", "authority", "vocabulary", "model", "rng",
    "manifest", "work_count", "work_identity", "path", "bytes", "utf8",
    "normalization", "documents", "vocabulary_size", "context_length", "stride",
    "token", "counts", "probabilities", "split", "window_sha256", "window_count",
    "target_count", "prompt", "generated_token_count", "mode", "temperature",
    "top_k", "seed", "logits", "output", "mutation", "evidence",
}

_PHASE9_INVARIANTS = {
    "phase9.type.argument", "phase9.type.record", "phase9.contract.bundle",
    "phase9.contract.generation", "phase9.contract.window",
    "phase9.governance.repository", "phase9.governance.runtime",
    "phase9.governance.dataset", "phase9.governance.sealed_test",
    "phase9.checkpoint.path", "phase9.checkpoint.catalog", "phase9.checkpoint.hash",
    "phase9.checkpoint.schema", "phase9.checkpoint.provenance",
    "phase9.checkpoint.cross_field", "phase9.checkpoint.model_state",
    "phase9.generation.prompt", "phase9.generation.count", "phase9.generation.mode",
    "phase9.generation.settings", "phase9.generation.context",
    "phase9.generation.output", "phase9.generation.mutation",
    "phase9.sampling.logits", "phase9.sampling.temperature", "phase9.sampling.top_k",
    "phase9.sampling.probabilities", "phase9.sampling.generator",
    "phase9.evaluation.split", "phase9.evaluation.windows",
    "phase9.evaluation.metric", "phase9.evaluation.mutation",
    "phase9.baseline.training_only", "phase9.baseline.counts",
    "phase9.baseline.probabilities", "phase9.evidence.schema",
    "phase9.evidence.amendment", "phase9.evidence.publication",
}
_LOSS_INVARIANTS = {
    "loss.logits.tensor", "loss.targets.tensor", "loss.logits.rank",
    "loss.targets.rank", "loss.logits.dtype", "loss.targets.dtype",
    "loss.input.device", "loss.logits.class_dimension", "loss.input.length_match",
    "loss.input.nonempty", "loss.logits.finite", "loss.targets.range",
    "loss.unrepresentable.shifted", "loss.unrepresentable.shifted_exp",
    "loss.unrepresentable.shifted_sum", "loss.unrepresentable.log_shifted_sum",
    "loss.unrepresentable.target_margin", "loss.unrepresentable.row",
    "loss.unrepresentable.scaled_row", "loss.unrepresentable.reduction",
}
_TOKENIZER_INVARIANTS = {
    "code_points.exact_tuple", "code_points.nonempty", "code_points.element.exact_int",
    "code_points.element.range", "code_points.element.duplicate",
    "code_points.strictly_increasing", "encode.text.isinstance_str",
    "encode.unknown_code_point", "decode.token_ids.sequence",
    "decode.token_id.exact_int_and_range", "decode.invalid_token_id",
}
_MINIGPT_TYPE_INVARIANTS = {
    "lm_head.input.tensor", "mini_gpt.submodules.structure",
    "mini_gpt.blocks.structure", "mini_gpt.modes.type", "mini_gpt.input.tensor",
    "lm_head.parameter.output_weight.type", "lm_head.parameter.output_bias.type",
    *(f"mini_gpt.parameter.{name}.type" for name, _ in _MODEL_PARAMETER_ORACLE),
}
_MINIGPT_CONTRACT_INVARIANTS = {
    "lm_head.parameters.structure", "lm_head.input.rank", "lm_head.input.dtype",
    "lm_head.input.device", "lm_head.input.model_width", "lm_head.input.sequence_length",
    "mini_gpt.submodules.structure", "mini_gpt.blocks.structure",
    "mini_gpt.modes.consistent", "mini_gpt.parameters.enumeration",
    "mini_gpt.parameters.unique", "mini_gpt.head.untied", "mini_gpt.input.rank",
    "mini_gpt.input.dtype", "mini_gpt.input.device", "mini_gpt.input.sequence_length",
    "mini_gpt.input.token_id_range", "mini_gpt.initialization.global_rng",
    *(f"lm_head.parameter.{name}.{suffix}" for name in ("output_weight", "output_bias") for suffix in ("shape", "device", "dtype")),
    *(f"mini_gpt.parameter.{name}.{suffix}" for name, _ in _MODEL_PARAMETER_ORACLE for suffix in ("ownership", "shape", "device", "dtype")),
}
_MINIGPT_NUMERICAL_INVARIANTS = {
    "lm_head.input.finite", "lm_head.output.finite", "mini_gpt.embedding_output.finite",
    "mini_gpt.final_normalization_output.finite", "mini_gpt.output.finite",
    "lm_head.parameter.output_weight.finite", "lm_head.parameter.output_bias.finite",
    *(f"mini_gpt.block_output.{index}.finite" for index in range(4)),
    *(f"mini_gpt.parameter.{name}.finite" for name, _ in _MODEL_PARAMETER_ORACLE),
}

QUALITATIVE_MATRIX = (
    ("romeo-greedy", "ROMEO:\n", 256, "greedy", None, None, None),
    ("romeo-focused", "ROMEO:\n", 256, "categorical", 0.8, 20, 9001),
    ("romeo-full", "ROMEO:\n", 256, "categorical", 1.0, 81, 9002),
    ("to-be-greedy", "To be", 256, "greedy", None, None, None),
    ("to-be-focused", "To be", 256, "categorical", 0.8, 20, 9003),
    ("to-be-full", "To be", 256, "categorical", 1.0, 81, 9004),
)

_SAMPLE_STAGE_TO_ID = {
    "sample_romeo_greedy": "romeo-greedy",
    "sample_romeo_focused": "romeo-focused",
    "sample_romeo_full": "romeo-full",
    "sample_to_be_greedy": "to-be-greedy",
    "sample_to_be_focused": "to-be-focused",
    "sample_to_be_full": "to-be-full",
}

DEVELOPMENT_QUESTION = (
    "Does the accepted Phase 9 best-validation model produce the fixed development "
    "metrics, baseline comparisons, and complete qualitative matrix under the accepted contract?"
)
DEVELOPMENT_AUTHORIZATION = (
    "Authorize exactly one governed development attempt for this evaluation ID; "
    "sealed-test access remains unauthorized."
)
DEVELOPMENT_SUCCESS_POLICY = (
    "Contracted execution and evidence completion—not score, baseline victory, or prose quality—determines PASS."
)
SEALED_AUTHORIZATION = (
    "Authorize one durable-marker-consumed Twelfth Night metric evaluation for this "
    "evaluation ID with no generation, tuning, or retry."
)
SEALED_INTERPRETATION_POLICY = "report_observation_without_threshold_or_post_test_tuning"

_PHASE8_AUTHORITY = {
    "phase8_closure_commit": "6d086625cb293f61700bd59b551ea310a5b680b9",
    "phase8_result_commit": "334119e63c716e4922cd0b67da4f5fc221faa886",
    "phase8_experiment_id": "EXP-20260912-01",
    "phase8_implementation_commit": "809834323d53407cb4a54ae539585bb3d78856eb",
    "phase8_contract_commit": "c50d77ac935bdf924b9b429a5776c419982a5d11",
    "phase8_spec_sha256": "1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd",
}
_CHECKPOINT_AUTHORITY = {
    "catalog_path": "checkpoints/phase8/EXP-20260912-01/checkpoint-catalog.json",
    "catalog_sha256": "6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241",
    "role": "best_validation",
    "logical_id": "epoch-0010",
    "epoch": 10,
    "validation_loss_hex": "0x1.3f94b678f5807p+1",
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "checkpoint_relative_path": f"objects/{CHECKPOINT_SHA256}.pt",
}
_CONTRACT_AUTHORITY = {
    "decision": "DEC-0021",
    "specification_path": "docs/GENERATION_EVALUATION_SPEC.md",
    "specification_sha256": PHASE9_SPEC_SHA256,
    "contract_commit": PHASE9_CONTRACT_COMMIT,
}
_TOKENIZER_AUTHORITY = {
    "implementation_commit": "de7a7f096fbbd8607c944412eaef30be9b686b56",
    "tokenizer_id": "shakespeare-code-point-v1",
    "schema_version": 1,
    "vocabulary_size": 81,
    "artifact_path": "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json",
    "artifact_sha256": "9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e",
}
_MODEL_AUTHORITY = {
    "model_width": 32, "max_sequence_length": 256, "number_of_blocks": 4,
    "number_of_heads": 4, "head_width": 8, "hidden_width": 128,
    "layer_norm_epsilon": 1e-5, "dropout_probability": 0.1,
    "embedding_seed": 1337, "block_parameter_seeds": [7001, 7002, 7003, 7004],
    "output_head_seed": 7005, "parameter_device": "cpu",
    "parameter_dtype": "torch.float32", "trainable_tensor_count": 90,
    "parameter_count": 63_825,
}
_DATASET_AUTHORITY = {
    "dataset_id": "shakespeare-eight-play",
    "manifest_sha256": "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb",
    "processing_manifest_sha256": "bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc",
    "training_works": [
        {"manifest_order": 1, "work_id": "hamlet", "split": "train", "processed_sha256": "281070a10b0fe06b877a6e9342cedecad1e5f65657d58fc51b14ce378c70a243"},
        {"manifest_order": 2, "work_id": "romeo-and-juliet", "split": "train", "processed_sha256": "59bfe35a3f7ccb63353aa4b8823f1a33fca4bd27c47943515ecc5fb8d6775775"},
        {"manifest_order": 3, "work_id": "macbeth", "split": "train", "processed_sha256": "fb24ddc7c0c35f7e7d6e0989858e3d9cdb30d00208f2122938d7cd900a3699c6"},
        {"manifest_order": 4, "work_id": "a-midsummer-nights-dream", "split": "train", "processed_sha256": "a9314798205c72c7cb7988813eb9df3d36e9731e45d882f761c907f31e178bd1"},
        {"manifest_order": 5, "work_id": "much-ado-about-nothing", "split": "train", "processed_sha256": "fbb46df4f5297329599a4e58b160d7182b30262b1c497c0de4002e27f536ce7b"},
        {"manifest_order": 6, "work_id": "henry-v", "split": "train", "processed_sha256": "eb8c7e711ac4182800d5b6280d1cf5c0c1df98f43c727d48cd04ca8288321560"},
    ],
    "validation_works": [
        {"manifest_order": 7, "work_id": "the-tempest", "split": "validation", "processed_sha256": "68a090b9d905967817948397410045f00592e4ea80b4115a13c860f67e1ca617"},
    ],
}

_KNOWN_EXECUTION_ERRORS = (
    Phase9TypeError,
    Phase9ContractError,
    Phase9GovernanceError,
    Phase9CheckpointError,
    Phase9NumericalError,
    Phase9PublicationError,
    TokenizationTypeError,
    TokenizerStateError,
    UnknownCodePointError,
    InvalidTokenIdError,
    MiniGPTTypeError,
    MiniGPTContractError,
    MiniGPTNumericalError,
    Phase4ContractError,
    Phase4TypeError,
)


@dataclass(frozen=True)
class _Record:
    evaluation_id: str
    kind: str
    values: Mapping[str, object]
    raw: bytes
    sha256: str


@dataclass(frozen=True)
class _Publication:
    state: str
    reference: Phase9EvidenceReference | None
    operation: str = "publish_no_clobber"


def _canonical_json(value: object, *, pretty: bool = True) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence") from None
    return (rendered + "\n").encode("ascii")


def _validate_id(value: object) -> str:
    if type(value) is not str:
        raise Phase9TypeError("phase9.type.argument", field="evaluation_id")
    if not ID_PATTERN.fullmatch(value):
        raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
    return value


def _validate_reference(value: object) -> Phase9EvidenceReference:
    if type(value) is not Phase9EvidenceReference:
        raise Phase9TypeError("phase9.type.record", field="evidence")
    if any(type(field) is not str for field in (value.evaluation_id, value.kind, value.status, value.relative_path, value.sha256)) or type(value.byte_count) is not int:
        raise Phase9TypeError("phase9.type.record", field="evidence")
    row = _EVIDENCE_ROWS.get((value.kind, value.status))
    if row is None:
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    filename, maximum = row
    expected = f"experiments/{value.evaluation_id}/artifacts/{filename}"
    if (
        not ID_PATTERN.fullmatch(value.evaluation_id)
        or value.relative_path != expected
        or type(value.byte_count) is not int
        or not 0 < value.byte_count <= maximum
        or type(value.sha256) is not str
        or len(value.sha256) != 64
        or any(character not in "0123456789abcdef" for character in value.sha256)
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return value


def _exact_mapping(value: object, keys: tuple[str, ...], *, field: str = "evidence") -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys):
        raise Phase9ContractError("phase9.evidence.schema", field=field)
    return value


def _is_digest(value: object, *, length: int = 64) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_timestamp(value: object) -> bool:
    if type(value) is not str or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.microsecond == 0 and parsed.tzinfo is not None


def _reference_from_mapping(value: object) -> Phase9EvidenceReference:
    mapping = _exact_mapping(
        value,
        ("evaluation_id", "kind", "status", "relative_path", "byte_count", "sha256"),
    )
    reference = Phase9EvidenceReference(
        mapping["evaluation_id"],  # type: ignore[arg-type]
        mapping["kind"],  # type: ignore[arg-type]
        mapping["status"],  # type: ignore[arg-type]
        mapping["relative_path"],  # type: ignore[arg-type]
        mapping["byte_count"],  # type: ignore[arg-type]
        mapping["sha256"],  # type: ignore[arg-type]
    )
    return _validate_reference(reference)


def _validate_metric_mapping(value: object, *, split: str, predictor: str | None = None) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "predictor", "split", "window_sha256", "nll", "nll_hex",
            "perplexity", "perplexity_hex", "correct_count", "target_count",
            "top1_accuracy", "top1_accuracy_hex", "window_count",
        ),
    )
    if (
        type(mapping["predictor"]) is not str
        or not mapping["predictor"]
        or (predictor is not None and mapping["predictor"] != predictor)
        or mapping["split"] != split
        or not _is_digest(mapping["window_sha256"])
        or any(type(mapping[name]) is not float for name in ("nll", "perplexity", "top1_accuracy"))
        or any(not isinstance(mapping[name], float) or not math.isfinite(mapping[name]) for name in ("nll", "perplexity", "top1_accuracy"))
        or mapping["nll"] < 0.0
        or mapping["perplexity"] < 1.0
        or not 0.0 <= mapping["top1_accuracy"] <= 1.0
        or mapping["nll_hex"] != mapping["nll"].hex()
        or mapping["perplexity_hex"] != mapping["perplexity"].hex()
        or mapping["top1_accuracy_hex"] != mapping["top1_accuracy"].hex()
        or type(mapping["correct_count"]) is not int
        or type(mapping["target_count"]) is not int
        or type(mapping["window_count"]) is not int
        or not 0 <= mapping["correct_count"] <= mapping["target_count"]
        or mapping["target_count"] <= 0
        or mapping["window_count"] <= 0
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    try:
        expected_perplexity = math.exp(mapping["nll"])
    except OverflowError:
        raise Phase9ContractError("phase9.evidence.schema", field="evidence") from None
    if (
        mapping["perplexity"] != expected_perplexity
        or mapping["top1_accuracy"]
        != mapping["correct_count"] / mapping["target_count"]
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return mapping


def _validate_baseline_mapping(value: object) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "vocabulary_size", "training_target_count", "raw_counts",
            "smoothed_counts", "probabilities", "probabilities_hex", "top_token_id",
        ),
    )
    size = mapping["vocabulary_size"]
    target_count = mapping["training_target_count"]
    raw = mapping["raw_counts"]
    smoothed = mapping["smoothed_counts"]
    probabilities = mapping["probabilities"]
    probability_hex = mapping["probabilities_hex"]
    if (
        type(size) is not int or size != 81
        or type(target_count) is not int or target_count != 792_699
        or type(raw) is not list or len(raw) != size
        or type(smoothed) is not list or len(smoothed) != size
        or type(probabilities) is not list or len(probabilities) != size
        or type(probability_hex) is not list or len(probability_hex) != size
        or any(type(item) is not int or item < 0 for item in raw)
        or any(type(item) is not int or item <= 0 for item in smoothed)
        or any(right != left + 1 for left, right in zip(raw, smoothed, strict=True))
        or sum(raw) != target_count
        or sum(smoothed) != target_count + size
        or any(type(item) is not float or not math.isfinite(item) or item <= 0 for item in probabilities)
        or any(item != count / (target_count + size) for item, count in zip(probabilities, smoothed, strict=True))
        or any(type(item) is not str or item != probability.hex() for item, probability in zip(probability_hex, probabilities, strict=True))
        or abs(math.fsum(probabilities) - 1.0) > 8 * sys.float_info.epsilon
        or type(mapping["top_token_id"]) is not int
        or mapping["top_token_id"] != min(range(size), key=lambda index: (-raw[index], index))
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return mapping


def _validate_configuration_row(value: object, expected: tuple[object, ...]) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "sample_id", "prompt", "generated_token_count", "mode", "temperature",
            "temperature_hex", "top_k", "seed",
        ),
    )
    sample_id, prompt, count, mode, temperature, top_k, seed = expected
    expected_mapping = {
        "sample_id": sample_id,
        "prompt": prompt,
        "generated_token_count": count,
        "mode": mode,
        "temperature": temperature,
        "temperature_hex": temperature.hex() if isinstance(temperature, float) else None,
        "top_k": top_k,
        "seed": seed,
    }
    if mapping != expected_mapping:
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return mapping


def _validate_sample_mapping(value: object, expected: tuple[object, ...]) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "sample_id", "prompt", "generated_token_count", "mode", "temperature",
            "temperature_hex", "top_k", "seed", "prompt_sha256", "checkpoint_role",
            "checkpoint_sha256", "generated_text", "generated_byte_count",
            "generated_code_point_count", "generated_text_sha256", "full_text_sha256",
            "local_generator_final_state_sha256",
        ),
    )
    config = {name: mapping[name] for name in (
        "sample_id", "prompt", "generated_token_count", "mode", "temperature",
        "temperature_hex", "top_k", "seed",
    )}
    _validate_configuration_row(config, expected)
    prompt = mapping["prompt"]
    generated = mapping["generated_text"]
    mode = mapping["mode"]
    if (
        mapping["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        or mapping["checkpoint_role"] != "best_validation"
        or mapping["checkpoint_sha256"] != CHECKPOINT_SHA256
        or type(generated) is not str
        or type(mapping["generated_byte_count"]) is not int
        or mapping["generated_byte_count"] != len(generated.encode("utf-8"))
        or type(mapping["generated_code_point_count"]) is not int
        or mapping["generated_code_point_count"] != 256
        or mapping["generated_text_sha256"] != hashlib.sha256(generated.encode("utf-8")).hexdigest()
        or mapping["full_text_sha256"] != hashlib.sha256((prompt + generated).encode("utf-8")).hexdigest()
        or (mode == "greedy" and mapping["local_generator_final_state_sha256"] is not None)
        or (mode == "categorical" and not _is_digest(mapping["local_generator_final_state_sha256"]))
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return mapping


def _validate_failure_mapping(value: object, *, sealed: bool) -> dict[str, object]:
    mapping = _exact_mapping(value, ("stage", "exception_type", "invariant", "safe_facts"))
    stage = mapping["stage"]
    facts = mapping["safe_facts"]
    if type(stage) is not str or type(mapping["exception_type"]) is not str or type(mapping["invariant"]) is not str or type(facts) is not dict:
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if sealed:
        if stage in ("sealed_corpus_open", "sealed_corpus_validation"):
            expected_keys = {"field"}
        elif stage == "sealed_tokenization":
            expected_keys = {"field", "position"}
        elif stage == "sealed_window_build":
            expected_keys = {"field"}
        elif stage in (
            "sealed_model_evaluation", "sealed_uniform_evaluation",
            "sealed_laplace_unigram_evaluation", "sealed_cross_predictor_comparison",
        ):
            expected_keys = {"field", "split"}
        elif stage == "sealed_evidence_publication":
            expected_keys = {"operation", "publication_state"}
        else:
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    else:
        if stage in ("repository_preflight", "inference_load", "training_window_build", "validation_window_build", "laplace_unigram_fit"):
            expected_keys = {"field"}
        elif stage == "development_corpus":
            expected_keys = {"field", "position"}
        elif stage in ("model_validation", "uniform_validation", "laplace_unigram_validation", "cross_predictor_comparison"):
            expected_keys = {"field", "split"}
        elif stage in _SAMPLE_STAGE_TO_ID:
            expected_keys = {"sample_id", "field", "completed_token_count"}
        elif stage == "development_evidence_publication":
            expected_keys = {"operation", "publication_state"}
        else:
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    operations = {
        "open_parent", "create_temporary", "write", "fsync_file",
        "publish_no_clobber", "fsync_parent", "revalidate_identity",
    }
    sample_ids = {row[0] for row in QUALITATIVE_MATRIX}
    if set(facts) != expected_keys:
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if "field" in facts and (type(facts["field"]) is not str or facts["field"] not in _FAILURE_FIELD_VOCABULARY):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if "position" in facts and facts["position"] is not None and (
        type(facts["position"]) is not int or facts["position"] < 0
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if "split" in facts and (
        type(facts["split"]) is not str
        or facts["split"] != ("test" if sealed else "validation")
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if "sample_id" in facts and (
        type(facts["sample_id"]) is not str
        or facts["sample_id"] not in sample_ids
        or stage not in _SAMPLE_STAGE_TO_ID
        or facts["sample_id"] != _SAMPLE_STAGE_TO_ID[stage]
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if "completed_token_count" in facts and (
        type(facts["completed_token_count"]) is not int
        or not 0 <= facts["completed_token_count"] <= 256
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if "operation" in facts and (
        type(facts["operation"]) is not str or facts["operation"] not in operations
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if "publication_state" in facts and (
        type(facts["publication_state"]) is not str
        or facts["publication_state"] not in (KNOWN_ABSENT, UNKNOWN)
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")

    phase9_type = {"Phase9TypeError": {"phase9.type.argument", "phase9.type.record"}}
    phase9_checkpoint = {
        "Phase9GovernanceError": {"phase9.governance.repository", "phase9.governance.runtime"},
        "Phase9CheckpointError": {
            "phase9.checkpoint.path", "phase9.checkpoint.catalog", "phase9.checkpoint.hash",
            "phase9.checkpoint.schema", "phase9.checkpoint.provenance",
            "phase9.checkpoint.cross_field", "phase9.checkpoint.model_state",
        },
    }
    evaluation_rules = {
        **phase9_type,
        "Phase9ContractError": {
            "phase9.evaluation.split", "phase9.evaluation.windows",
            "phase9.evaluation.metric", "phase9.evaluation.mutation",
            "phase9.baseline.counts", "phase9.baseline.probabilities",
        },
        "Phase9NumericalError": {
            "phase9.evaluation.metric", "phase9.baseline.probabilities",
        },
        "MiniGPTTypeError": _MINIGPT_TYPE_INVARIANTS,
        "MiniGPTContractError": _MINIGPT_CONTRACT_INVARIANTS,
        "MiniGPTNumericalError": _MINIGPT_NUMERICAL_INVARIANTS,
        "Phase4TypeError": _LOSS_INVARIANTS,
        "Phase4ContractError": _LOSS_INVARIANTS,
    }
    sample_rules = {
        **phase9_type,
        "Phase9ContractError": {
            "phase9.contract.bundle", "phase9.contract.generation",
            "phase9.generation.prompt", "phase9.generation.count",
            "phase9.generation.mode", "phase9.generation.settings",
            "phase9.generation.context", "phase9.generation.output",
            "phase9.generation.mutation", "phase9.sampling.logits",
            "phase9.sampling.temperature", "phase9.sampling.top_k",
            "phase9.sampling.probabilities", "phase9.sampling.generator",
        },
        "Phase9NumericalError": {
            "phase9.sampling.logits", "phase9.sampling.temperature",
            "phase9.sampling.top_k", "phase9.sampling.probabilities",
            "phase9.sampling.generator",
        },
        "TokenizationTypeError": _TOKENIZER_INVARIANTS,
        "TokenizerStateError": _TOKENIZER_INVARIANTS,
        "UnknownCodePointError": {"encode.unknown_code_point"},
        "InvalidTokenIdError": {"decode.invalid_token_id"},
        "MiniGPTTypeError": _MINIGPT_TYPE_INVARIANTS,
        "MiniGPTContractError": _MINIGPT_CONTRACT_INVARIANTS,
        "MiniGPTNumericalError": _MINIGPT_NUMERICAL_INVARIANTS,
    }
    if stage == "repository_preflight":
        rules = {**phase9_type, "Phase9GovernanceError": {"phase9.governance.repository", "phase9.governance.runtime"}}
    elif stage == "inference_load":
        rules = phase9_checkpoint
    elif stage == "development_corpus":
        rules = {"Phase9GovernanceError": {"phase9.governance.dataset"}}
    elif stage in ("training_window_build", "validation_window_build", "sealed_window_build"):
        rules = {**phase9_type, "Phase9ContractError": {"phase9.contract.window"}}
    elif stage == "laplace_unigram_fit":
        rules = {
            **phase9_type,
            "Phase9ContractError": {"phase9.baseline.training_only", "phase9.baseline.counts"},
            "Phase9NumericalError": {"phase9.baseline.probabilities"},
        }
    elif stage in (
        "model_validation", "uniform_validation", "laplace_unigram_validation",
        "sealed_model_evaluation", "sealed_uniform_evaluation",
        "sealed_laplace_unigram_evaluation",
    ):
        rules = evaluation_rules
    elif stage in _SAMPLE_STAGE_TO_ID:
        rules = sample_rules
    elif stage in ("cross_predictor_comparison", "sealed_cross_predictor_comparison"):
        rules = {"Phase9ContractError": {"phase9.evaluation.metric"}}
    elif stage in ("development_evidence_publication", "sealed_evidence_publication", "sealed_access_marker_publication"):
        rules = {"Phase9PublicationError": {"phase9.evidence.publication"}}
    elif stage in ("sealed_corpus_open", "sealed_corpus_validation"):
        rules = {"Phase9GovernanceError": {"phase9.governance.dataset", "phase9.governance.sealed_test"}}
    elif stage == "sealed_tokenization":
        rules = {
            "TokenizationTypeError": _TOKENIZER_INVARIANTS,
            "TokenizerStateError": _TOKENIZER_INVARIANTS,
            "UnknownCodePointError": {"encode.unknown_code_point"},
        }
    else:
        rules = {}
    exception_type = mapping["exception_type"]
    invariant = mapping["invariant"]
    if (
        exception_type not in rules
        or type(invariant) is not str
        or invariant not in rules[exception_type]
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return mapping


def _validate_authority_mapping(value: object, *, sealed: bool) -> dict[str, object]:
    development_keys = (
        "phase8_closure_commit", "phase8_result_commit", "phase8_experiment_id",
        "phase8_implementation_commit", "phase8_contract_commit", "phase8_spec_sha256",
        "phase9_contract_commit", "phase9_contract_sha256", "phase9_implementation_commit",
        "development_plan_record_sha256", "development_pre_registration_commit",
        "development_authorization_record_sha256", "development_authorization_commit",
        "catalog_sha256", "checkpoint_role", "checkpoint_sha256", "vocabulary_sha256",
        "dataset_manifest_sha256", "processing_manifest_sha256", "requirements_lock_sha256",
    )
    sealed_extra = (
        "development_result_record_sha256", "development_result_commit",
        "development_evidence_sha256", "sealed_plan_record_sha256",
        "sealed_pre_registration_commit", "sealed_authorization_record_sha256",
        "sealed_authorization_commit",
    )
    mapping = _exact_mapping(
        value, development_keys + sealed_extra if sealed else development_keys,
        field="authority",
    )
    fixed = {
        **_PHASE8_AUTHORITY,
        "phase9_contract_commit": PHASE9_CONTRACT_COMMIT,
        "phase9_contract_sha256": PHASE9_SPEC_SHA256,
        "catalog_sha256": _CHECKPOINT_AUTHORITY["catalog_sha256"],
        "checkpoint_role": "best_validation",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vocabulary_sha256": _TOKENIZER_AUTHORITY["artifact_sha256"],
        "dataset_manifest_sha256": _DATASET_AUTHORITY["manifest_sha256"],
        "processing_manifest_sha256": _DATASET_AUTHORITY["processing_manifest_sha256"],
        "requirements_lock_sha256": "8ddf7a11bf67d36bce8ba58abffc5416a0fb778cdc20d03c077d318715122cfa",
    }
    if any(mapping[name] != expected for name, expected in fixed.items()):
        raise Phase9ContractError("phase9.evidence.schema", field="authority")
    commit_fields = (
        "phase9_implementation_commit", "development_pre_registration_commit",
        "development_authorization_commit",
    ) + (("development_result_commit", "sealed_pre_registration_commit", "sealed_authorization_commit") if sealed else ())
    digest_fields = (
        "development_plan_record_sha256", "development_authorization_record_sha256",
    ) + (("development_result_record_sha256", "development_evidence_sha256", "sealed_plan_record_sha256", "sealed_authorization_record_sha256") if sealed else ())
    if (
        any(type(mapping[name]) is not str or not re.fullmatch(r"[0-9a-f]{40}", mapping[name]) for name in commit_fields)
        or any(not _is_digest(mapping[name]) for name in digest_fields)
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="authority")
    return mapping


def _parse_records(content: bytes) -> tuple[_Record, ...]:
    if not content or len(content) > 1_048_576:
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise Phase9GovernanceError("phase9.governance.repository", field="record") from None
    lines = text.splitlines(keepends=True)
    output: list[_Record] = []
    index = 0
    while index < len(lines):
        match = HEADING_PATTERN.fullmatch(lines[index])
        if match is None:
            if re.match(r"### EXP-[0-9]{8}-[0-9]{2} — Phase 9 ", lines[index]):
                raise Phase9GovernanceError("phase9.governance.repository", field="record")
            index += 1
            continue
        start = index
        evaluation_id, suffix = match.groups()
        kind = _KIND_BY_SUFFIX[suffix]
        labels = _LABELS[kind]
        index += 1
        if index >= len(lines) or lines[index] != "\n":
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
        index += 1
        values: dict[str, object] = {}
        for expected_label in labels:
            if index >= len(lines):
                raise Phase9GovernanceError("phase9.governance.repository", field="record")
            field_match = FIELD_PATTERN.fullmatch(lines[index])
            if field_match is None or field_match.group(1) != expected_label:
                raise Phase9GovernanceError("phase9.governance.repository", field="record")
            try:
                values[expected_label] = json.loads(field_match.group(2))
            except json.JSONDecodeError:
                raise Phase9GovernanceError("phase9.governance.repository", field="record") from None
            if json.dumps(values[expected_label], ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) != field_match.group(2):
                raise Phase9GovernanceError("phase9.governance.repository", field="record")
            index += 1
        if index >= len(lines) or lines[index] != "\n":
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
        index += 1
        raw = "".join(lines[start:index]).encode("utf-8")
        if values["Evaluation ID"] != evaluation_id or values["Entry kind"] != kind:
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
        output.append(_Record(evaluation_id, kind, values, raw, hashlib.sha256(raw).hexdigest()))
    return tuple(output)


def _render_record_bytes(evaluation_id: str, kind: str, values: Mapping[str, object]) -> bytes:
    suffix = next(
        (suffix for suffix, candidate in _KIND_BY_SUFFIX.items() if candidate == kind),
        None,
    )
    if suffix is None or set(values) != set(_LABELS[kind]):
        raise Phase9ContractError("phase9.evidence.schema", field="record")
    lines = [f"### {evaluation_id} — Phase 9 {suffix}\n", "\n"]
    for label in _LABELS[kind]:
        rendered = json.dumps(
            values[label], ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        lines.append(f"- **{label}:** {rendered}\n")
    lines.append("\n")
    return "".join(lines).encode("utf-8")


def _append_sealed_uncertain_record(
    root: Path,
    evaluation_id: str,
    plan: _Record,
    authorization: _Record,
    *,
    authorization_commit: str,
    operation: str,
    started_at_utc: str,
) -> None:
    finished = _now()
    recorded = _now()
    values = {
        "Evaluation ID": evaluation_id,
        "Entry kind": "sealed_test_result",
        "Scope": "sealed_test",
        "Status": "uncertain",
        "Recorded at UTC": recorded,
        "Started at UTC": started_at_utc,
        "Finished at UTC": finished,
        "Sealed plan record SHA-256": plan.sha256,
        "Sealed pre-registration commit": authorization.values["Sealed pre-registration commit"],
        "Sealed authorization record SHA-256": authorization.sha256,
        "Sealed authorization commit": authorization_commit,
        "Access marker": None,
        "Sealed evidence": None,
        "Model test": None,
        "Uniform test": None,
        "Laplace unigram test": None,
        "Sealed-test access": "one_shot_consumed_uncertain",
        "Failure": {
            "stage": "sealed_access_marker_publication",
            "exception_type": "Phase9PublicationError",
            "invariant": "phase9.evidence.publication",
            "safe_facts": {"operation": operation, "publication_state": UNKNOWN},
        },
        "Observations": [],
        "Conclusions": [],
        "Limitations": list(LIMITATIONS),
        "Next gate": "independent_sealed_result_review",
    }
    raw = _render_record_bytes(evaluation_id, "sealed_test_result", values)
    record = _Record(
        evaluation_id, "sealed_test_result", values, raw,
        hashlib.sha256(raw).hexdigest(),
    )
    _validate_sealed_result_record(record)
    root_fd: int | None = None
    log_fd: int | None = None
    temporary_fd: int | None = None
    log_locked = False
    root_locked = False
    renamed = False
    temporary = f".EXPERIMENT_LOG.{evaluation_id}.{authorization.sha256}.tmp"
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        root_locked = True
        log_fd = os.open("EXPERIMENT_LOG.md", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        fcntl.flock(log_fd, fcntl.LOCK_EX)
        log_locked = True
        before = os.fstat(log_fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1_048_576:
            raise OSError
        original = bytearray()
        while len(original) <= 1_048_576:
            chunk = os.read(log_fd, min(1_048_576, 1_048_577 - len(original)))
            if not chunk:
                break
            original.extend(chunk)
        if len(original) != before.st_size or raw in original:
            raise OSError
        separator = b"" if bytes(original).endswith(b"\n\n") else b"\n"
        addition = separator + raw
        if len(original) + len(addition) > 1_048_576:
            raise OSError
        candidate = bytes(original) + addition
        try:
            os.unlink(temporary, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileNotFoundError:
            pass
        temporary_fd = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IMODE(before.st_mode),
            dir_fd=root_fd,
        )
        os.fchmod(temporary_fd, stat.S_IMODE(before.st_mode))
        offset = 0
        while offset < len(candidate):
            written = os.write(temporary_fd, candidate[offset:])
            if written <= 0:
                raise OSError
            offset += written
        os.fsync(temporary_fd)
        temporary_metadata = os.fstat(temporary_fd)
        os.lseek(temporary_fd, 0, os.SEEK_SET)
        verified = bytearray()
        while len(verified) <= len(candidate):
            chunk = os.read(
                temporary_fd,
                min(1_048_576, len(candidate) + 1 - len(verified)),
            )
            if not chunk:
                break
            verified.extend(chunk)
        current_entry = os.stat(
            "EXPERIMENT_LOG.md", dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_size != len(candidate)
            or bytes(verified) != candidate
            or (current_entry.st_dev, current_entry.st_ino, current_entry.st_mode, current_entry.st_size)
            != (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        ):
            raise OSError
        os.close(temporary_fd)
        temporary_fd = None
        os.rename(
            temporary,
            "EXPERIMENT_LOG.md",
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        renamed = True
        os.fsync(root_fd)
        final_fd = os.open(
            "EXPERIMENT_LOG.md", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd
        )
        try:
            after = os.fstat(final_fd)
            final_content = bytearray()
            while len(final_content) <= len(candidate):
                chunk = os.read(
                    final_fd,
                    min(1_048_576, len(candidate) + 1 - len(final_content)),
                )
                if not chunk:
                    break
                final_content.extend(chunk)
        finally:
            os.close(final_fd)
        entry = os.stat(
            "EXPERIMENT_LOG.md", dir_fd=root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_size != len(candidate)
            or bytes(final_content) != candidate
            or (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_size)
            != (after.st_dev, after.st_ino, after.st_mode, after.st_size)
        ):
            raise OSError
        _revalidate_root(root, root_fd)
    except (OSError, TypeError, ValueError):
        raise Phase9PublicationError(
            "phase9.evidence.publication",
            operation="write",
            publication_state=UNKNOWN,
        ) from None
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if root_fd is not None and not renamed:
            try:
                os.unlink(temporary, dir_fd=root_fd)
                os.fsync(root_fd)
            except OSError:
                pass
        if log_fd is not None:
            if log_locked:
                fcntl.flock(log_fd, fcntl.LOCK_UN)
            os.close(log_fd)
        if root_fd is not None:
            if root_locked:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)


def _candidate_blocks(content: bytes) -> tuple[bytes, ...]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise Phase9GovernanceError("phase9.governance.repository", field="record") from None
    lines = text.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith("### ") and " — Phase 9 " in line
    ]
    blocks: list[bytes] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        blocks.append("".join(lines[start:end]).encode("utf-8"))
    return tuple(blocks)


def _parse_records_scoped(content: bytes, relevant_ids: set[str]) -> tuple[_Record, ...]:
    if not content or len(content) > 1_048_576:
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    output: list[_Record] = []
    for block in _candidate_blocks(content):
        first_line = block.splitlines(keepends=True)[0].decode("utf-8")
        match = HEADING_PATTERN.fullmatch(first_line)
        heading_id = None
        suffix = None
        if match is not None:
            heading_id, suffix = match.groups()
        endpoint_relevant = suffix == "development supersession" and any(
            evaluation_id.encode("ascii") in block for evaluation_id in relevant_ids
        )
        required = heading_id in relevant_ids or endpoint_relevant
        try:
            parsed = _parse_records(block)
        except Phase9GovernanceError:
            if required:
                raise
            continue
        if len(parsed) != 1:
            if required:
                raise Phase9GovernanceError("phase9.governance.repository", field="record")
            continue
        output.append(parsed[0])
    return tuple(output)


def _parse_selected_chain(content: bytes, evaluation_id: str) -> tuple[_Record, ...]:
    relevant = {evaluation_id}
    while True:
        records = _parse_records_scoped(content, relevant)
        plan = _one(records, evaluation_id, "development_plan")
        current = plan
        discovered = set(relevant)
        for _ in range(65):
            predecessor = current.values["Predecessor evaluation"]
            if predecessor is None:
                break
            if type(predecessor) is not str or not ID_PATTERN.fullmatch(predecessor):
                raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
            discovered.add(predecessor)
            matches = _matches(records, predecessor, "development_plan")
            if len(matches) != 1:
                break
            current = matches[0]
        else:
            raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
        if discovered == relevant:
            return records
        relevant = discovered


def _matches(records: tuple[_Record, ...], evaluation_id: str, kind: str) -> tuple[_Record, ...]:
    return tuple(value for value in records if value.evaluation_id == evaluation_id and value.kind == kind)


def _one(records: tuple[_Record, ...], evaluation_id: str, kind: str) -> _Record:
    matches = _matches(records, evaluation_id, kind)
    if len(matches) != 1:
        raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
    return matches[0]


def _development_authority(records: tuple[_Record, ...], evaluation_id: str) -> tuple[_Record, _Record]:
    selected_plan = _one(records, evaluation_id, "development_plan")
    selected_authorization = _one(records, evaluation_id, "development_authorization")
    if (
        _matches(records, evaluation_id, "development_result")
        or _matches(records, evaluation_id, "development_supersession")
        or selected_authorization.values["Development plan record SHA-256"] != selected_plan.sha256
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
    visited: set[str] = set()
    current = evaluation_id
    for _ in range(65):
        if current in visited:
            raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
        visited.add(current)
        plan = _one(records, current, "development_plan")
        predecessor = plan.values["Predecessor evaluation"]
        if predecessor is None:
            incoming = [
                item for item in records
                if item.kind == "development_supersession"
                and item.values["Successor evaluation ID"] == current
            ]
            if incoming:
                raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
            break
        if type(predecessor) is not str or not ID_PATTERN.fullmatch(predecessor) or predecessor == current:
            raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
        prior_plan = _one(records, predecessor, "development_plan")
        prior_result = _one(records, predecessor, "development_result")
        supersession = _one(records, predecessor, "development_supersession")
        if (
            prior_result.values["Status"] not in ("failed", "uncertain")
            or supersession.values["Prior result record SHA-256"] != prior_result.sha256
            or supersession.values["Successor evaluation ID"] != current
        ):
            raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
        outgoing = [item for item in records if item.kind == "development_supersession" and item.evaluation_id == predecessor]
        incoming = [item for item in records if item.kind == "development_supersession" and item.values["Successor evaluation ID"] == current]
        if len(outgoing) != 1 or len(incoming) != 1:
            raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
        current = prior_plan.evaluation_id
    else:
        raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
    return selected_plan, selected_authorization


def _validate_predecessor_chain(
    root: Path,
    records: tuple[_Record, ...],
    evaluation_id: str,
) -> None:
    current_id = evaluation_id
    visited: set[str] = set()
    for _ in range(65):
        if current_id in visited:
            raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
        visited.add(current_id)
        plan = _one(records, current_id, "development_plan")
        _validate_plan_record(
            root,
            plan,
            require_live_implementation=current_id == evaluation_id,
        )
        predecessor = plan.values["Predecessor evaluation"]
        if predecessor is None:
            return
        assert isinstance(predecessor, str)
        prior_plan = _one(records, predecessor, "development_plan")
        prior_result = _one(records, predecessor, "development_result")
        _validate_development_result_record(prior_result)
        if prior_result.values["Predecessor evaluation"] != prior_plan.values["Predecessor evaluation"]:
            raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")
        supersession = _one(records, predecessor, "development_supersession")
        _validate_supersession_record(supersession, prior_result, current_id)
        current_id = predecessor
    raise Phase9GovernanceError("phase9.governance.repository", field="evaluation_id")


def _inspect_final(path: Path) -> str:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return KNOWN_ABSENT
    except OSError:
        return UNKNOWN
    return UNKNOWN if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) else KNOWN_PRESENT


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _open_child_directory(parent_fd: int, name: str, *, create: bool) -> tuple[int, tuple[int, int, int]]:
    if not name or name in (".", "..") or "/" in name or os.sep in name:
        raise OSError
    created = False
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
    if created:
        os.fsync(parent_fd)
        published = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode):
            raise OSError
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    metadata = os.fstat(fd)
    if not created:
        os.fsync(parent_fd)
        published = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or not stat.S_ISDIR(published.st_mode)
        or _directory_identity(metadata) != _directory_identity(published)
    ):
        os.close(fd)
        raise OSError
    return fd, _directory_identity(metadata)


def _open_artifact_directory(
    root: Path,
    evaluation_id: str,
    *,
    create: bool,
) -> tuple[list[int], list[tuple[int, str, tuple[int, int, int]]]]:
    if not ID_PATTERN.fullmatch(evaluation_id):
        raise OSError
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fds = [root_fd]
    entries: list[tuple[int, str, tuple[int, int, int]]] = []
    parent = root_fd
    try:
        for name in ("experiments", evaluation_id, "artifacts"):
            child, identity = _open_child_directory(parent, name, create=create)
            fds.append(child)
            entries.append((parent, name, identity))
            parent = child
        return fds, entries
    except BaseException:
        for fd in reversed(fds):
            os.close(fd)
        raise


def _inspect_fd_entry(directory_fd: int, name: str) -> str:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return KNOWN_ABSENT
    except OSError:
        return UNKNOWN
    return KNOWN_PRESENT if stat.S_ISREG(metadata.st_mode) else UNKNOWN


def _revalidate_publication_directories(
    entries: list[tuple[int, str, tuple[int, int, int]]],
) -> None:
    for parent_fd, name, identity in entries:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or _directory_identity(metadata) != identity:
            raise OSError


def _revalidate_root(root: Path, root_fd: int) -> None:
    opened = os.fstat(root_fd)
    live = os.stat(root, follow_symlinks=False)
    if (
        not stat.S_ISDIR(live.st_mode)
        or _directory_identity(opened) != _directory_identity(live)
    ):
        raise OSError


def _publish(
    root: Path,
    evaluation_id: str,
    *,
    filename: str,
    kind: str,
    status: str,
    content: bytes,
) -> _Publication:
    row = _EVIDENCE_ROWS.get((kind, status))
    if row is None or row[0] != filename or not 0 < len(content) <= row[1]:
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    if type(root) is not type(Path()) or not root.is_absolute() or root.resolve() != root:
        raise Phase9ContractError("phase9.evidence.schema", field="path")
    fds: list[int] = []
    entries: list[tuple[int, str, tuple[int, int, int]]] = []
    try:
        fds, entries = _open_artifact_directory(root, evaluation_id, create=True)
    except OSError:
        return _Publication(UNKNOWN, None, "open_parent")
    artifacts_fd = fds[-1]
    if _inspect_fd_entry(artifacts_fd, filename) != KNOWN_ABSENT:
        for directory_fd in reversed(fds):
            os.close(directory_fd)
        return _Publication(UNKNOWN, None)
    temporary = f".{filename}.{secrets.token_hex(16)}.tmp"
    fd: int | None = None
    linked = False
    temporary_identity: tuple[int, int] | None = None
    operation = "create_temporary"
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=artifacts_fd,
        )
        metadata = os.fstat(fd)
        temporary_identity = (metadata.st_dev, metadata.st_ino)
        offset = 0
        operation = "write"
        while offset < len(content):
            written = os.write(fd, content[offset:])
            if written <= 0:
                raise OSError
            offset += written
        operation = "fsync_file"
        os.fsync(fd)
        os.close(fd)
        fd = None
        operation = "publish_no_clobber"
        os.link(
            temporary,
            filename,
            src_dir_fd=artifacts_fd,
            dst_dir_fd=artifacts_fd,
            follow_symlinks=False,
        )
        linked = True
        operation = "revalidate_identity"
        final_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=artifacts_fd)
        try:
            final_metadata = os.fstat(final_fd)
            if (
                not stat.S_ISREG(final_metadata.st_mode)
                or final_metadata.st_size != len(content)
                or (final_metadata.st_dev, final_metadata.st_ino) != temporary_identity
            ):
                raise OSError
            observed = bytearray()
            while len(observed) <= len(content):
                chunk = os.read(final_fd, min(1_048_576, len(content) + 1 - len(observed)))
                if not chunk:
                    break
                observed.extend(chunk)
            if bytes(observed) != content:
                raise OSError
        finally:
            os.close(final_fd)
        operation = "fsync_parent"
        os.fsync(artifacts_fd)
        operation = "revalidate_identity"
        _revalidate_publication_directories(entries)
        _revalidate_root(root, fds[0])
        final_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=artifacts_fd)
        try:
            final_metadata = os.fstat(final_fd)
            observed = bytearray()
            while len(observed) <= len(content):
                chunk = os.read(final_fd, min(1_048_576, len(content) + 1 - len(observed)))
                if not chunk:
                    break
                observed.extend(chunk)
            if (
                (final_metadata.st_dev, final_metadata.st_ino) != temporary_identity
                or final_metadata.st_size != len(content)
                or bytes(observed) != content
            ):
                raise OSError
        finally:
            os.close(final_fd)
        reference = Phase9EvidenceReference(
            evaluation_id=evaluation_id,
            kind=kind,
            status=status,
            relative_path=f"experiments/{evaluation_id}/artifacts/{filename}",
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        _validate_reference(reference)
        return _Publication(KNOWN_PRESENT, reference, operation)
    except FileExistsError:
        return _Publication(UNKNOWN, None, "publish_no_clobber")
    except OSError:
        return _Publication(UNKNOWN if linked else KNOWN_ABSENT, None, operation)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            metadata = os.stat(temporary, dir_fd=artifacts_fd, follow_symlinks=False)
            if temporary_identity == (metadata.st_dev, metadata.st_ino):
                os.unlink(temporary, dir_fd=artifacts_fd)
        except OSError:
            pass
        for directory_fd in reversed(fds):
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _publish_terminal(
    root: Path,
    evaluation_id: str,
    *,
    filename: str,
    opposite_filename: str,
    kind: str,
    status: str,
    content: bytes,
) -> _Publication:
    try:
        fds, _ = _open_artifact_directory(root, evaluation_id, create=True)
    except OSError:
        return _Publication(UNKNOWN, None, "open_parent")
    locked = False
    try:
        fcntl.flock(fds[-1], fcntl.LOCK_EX)
        locked = True
        desired_state = _inspect_fd_entry(fds[-1], filename)
        opposite_state = _inspect_fd_entry(fds[-1], opposite_filename)
        if desired_state != KNOWN_ABSENT or opposite_state != KNOWN_ABSENT:
            return _Publication(UNKNOWN, None)
        result = _publish(
            root,
            evaluation_id,
            filename=filename,
            kind=kind,
            status=status,
            content=content,
        )
        if result.state != KNOWN_PRESENT:
            desired_after = _inspect_fd_entry(fds[-1], filename)
            opposite_after = _inspect_fd_entry(fds[-1], opposite_filename)
            if desired_after != KNOWN_ABSENT or opposite_after != KNOWN_ABSENT:
                return _Publication(UNKNOWN, None, result.operation)
        return result
    except OSError:
        return _Publication(UNKNOWN, None, "open_parent")
    finally:
        if locked:
            fcntl.flock(fds[-1], fcntl.LOCK_UN)
        for directory_fd in reversed(fds):
            os.close(directory_fd)


def _publish_sealed_access_marker(
    root: Path,
    evaluation_id: str,
    authorization_sha256: str,
    content: bytes,
    *,
    fds: list[int],
    entries: list[tuple[int, str, tuple[int, int, int]]],
) -> _Publication:
    filename = "phase9-sealed-test-access.json"
    temporary = f".{filename}.{authorization_sha256}.tmp"
    artifacts_fd = fds[-1]
    if (
        _inspect_fd_entry(artifacts_fd, filename) != KNOWN_ABSENT
        or _inspect_fd_entry(artifacts_fd, temporary) != KNOWN_ABSENT
    ):
        return _Publication(UNKNOWN, None, "publish_no_clobber")
    fd: int | None = None
    temporary_identity: tuple[int, int, int, int] | None = None
    temporary_durable = False
    operation = "create_temporary"
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=artifacts_fd,
        )
        before = os.fstat(fd)
        temporary_identity = (
            before.st_dev, before.st_ino, before.st_mode, len(content)
        )
        operation = "write"
        offset = 0
        while offset < len(content):
            written = os.write(fd, content[offset:])
            if written <= 0:
                raise OSError
            offset += written
        operation = "fsync_file"
        os.fsync(fd)
        os.close(fd)
        fd = None
        operation = "fsync_parent"
        os.fsync(artifacts_fd)
        operation = "revalidate_identity"
        metadata = os.stat(temporary, dir_fd=artifacts_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size)
            != temporary_identity
        ):
            raise OSError
        temporary_durable = True
        operation = "publish_no_clobber"
        os.link(
            temporary,
            filename,
            src_dir_fd=artifacts_fd,
            dst_dir_fd=artifacts_fd,
            follow_symlinks=False,
        )
        operation = "fsync_parent"
        os.fsync(artifacts_fd)
        operation = "revalidate_identity"
        final_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=artifacts_fd)
        try:
            final_metadata = os.fstat(final_fd)
            observed = bytearray()
            while len(observed) <= len(content):
                chunk = os.read(
                    final_fd,
                    min(1_048_576, len(content) + 1 - len(observed)),
                )
                if not chunk:
                    break
                observed.extend(chunk)
            if (
                temporary_identity is None
                or (final_metadata.st_dev, final_metadata.st_ino)
                != temporary_identity[:2]
                or final_metadata.st_size != len(content)
                or bytes(observed) != content
            ):
                raise OSError
        finally:
            os.close(final_fd)
        _revalidate_publication_directories(entries)
        _revalidate_root(root, fds[0])
        reference = Phase9EvidenceReference(
            evaluation_id=evaluation_id,
            kind="phase9_sealed_test_access",
            status="consumed",
            relative_path=(
                f"experiments/{evaluation_id}/artifacts/{filename}"
            ),
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        _validate_reference(reference)
        try:
            os.unlink(temporary, dir_fd=artifacts_fd)
            os.fsync(artifacts_fd)
        except OSError:
            pass
        return _Publication(KNOWN_PRESENT, reference, operation)
    except FileExistsError:
        return _Publication(UNKNOWN, None, operation)
    except OSError:
        if temporary_durable:
            return _Publication(UNKNOWN, None, operation)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            fd = None
        try:
            os.unlink(temporary, dir_fd=artifacts_fd)
            os.fsync(artifacts_fd)
            if _inspect_fd_entry(artifacts_fd, temporary) == KNOWN_ABSENT:
                return _Publication(KNOWN_ABSENT, None, operation)
        except OSError:
            pass
        return _Publication(UNKNOWN, None, operation)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _serialized_sealed_access_publication(
    root: Path,
    evaluation_id: str,
    plan: _Record,
    authorization: _Record,
    *,
    authorization_commit: str,
    marker: dict[str, object],
) -> _Publication:
    try:
        fds, entries = _open_artifact_directory(root, evaluation_id, create=True)
    except OSError:
        publication = _Publication(UNKNOWN, None, "open_parent")
        _append_sealed_uncertain_record(
            root,
            evaluation_id,
            plan,
            authorization,
            authorization_commit=authorization_commit,
            operation=publication.operation,
            started_at_utc=marker["created_at_utc"],  # type: ignore[arg-type]
        )
        return publication
    locked = False
    try:
        fcntl.flock(fds[-1], fcntl.LOCK_EX)
        locked = True
        try:
            current_log = (root / "EXPERIMENT_LOG.md").read_bytes()
            current_records = _parse_records_scoped(current_log, {evaluation_id})
        except (OSError, Phase9GovernanceError):
            raise Phase9GovernanceError(
                "phase9.governance.sealed_test", field="record"
            ) from None
        existing = _matches(current_records, evaluation_id, "sealed_test_result")
        if existing:
            if len(existing) == 1:
                _validate_sealed_result_record(existing[0])
            raise Phase9GovernanceError(
                "phase9.governance.sealed_test", field="evaluation_id"
            )
        _require_artifacts_absent(
            root,
            evaluation_id,
            (
                "phase9-sealed-test-access.json",
                "phase9-sealed-test-evidence.json",
                "phase9-sealed-test-failure.json",
            ),
        )
        publication = _publish_sealed_access_marker(
            root,
            evaluation_id,
            authorization.sha256,
            _canonical_json(marker),
            fds=fds,
            entries=entries,
        )
        if publication.state == UNKNOWN:
            _append_sealed_uncertain_record(
                root,
                evaluation_id,
                plan,
                authorization,
                authorization_commit=authorization_commit,
                operation=publication.operation,
                started_at_utc=marker["created_at_utc"],  # type: ignore[arg-type]
            )
        return publication
    finally:
        if locked:
            fcntl.flock(fds[-1], fcntl.LOCK_UN)
        for directory_fd in reversed(fds):
            os.close(directory_fd)


def _metric_mapping(value: Phase9Metrics) -> dict[str, object]:
    return {
        "predictor": value.predictor, "split": value.split,
        "window_sha256": value.window_sha256, "nll": value.nll,
        "nll_hex": value.nll.hex(), "perplexity": value.perplexity,
        "perplexity_hex": value.perplexity.hex(), "correct_count": value.correct_count,
        "target_count": value.target_count, "top1_accuracy": value.top1_accuracy,
        "top1_accuracy_hex": value.top1_accuracy.hex(), "window_count": value.window_count,
    }


def _compare_metrics(*values: Phase9Metrics) -> None:
    if len(values) != 3:
        raise Phase9ContractError("phase9.evaluation.metric", field="counts")
    expected = (values[0].split, values[0].window_sha256, values[0].window_count, values[0].target_count)
    if any((value.split, value.window_sha256, value.window_count, value.target_count) != expected for value in values[1:]):
        raise Phase9ContractError("phase9.evaluation.metric", field="window_sha256")


def _configuration_mapping() -> dict[str, object]:
    return {
        "context_length": 256,
        "stride": 256,
        "retain_unpadded_tail": True,
        "maximum_generated_tokens": 1024,
        "metric_names": ["nll", "perplexity", "top1_accuracy"],
        "baseline_names": ["uniform_81", "training_laplace_unigram_add_one"],
        "unigram_smoothing": "add_one",
        "qualitative_matrix": [
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "generated_token_count": count,
                "mode": mode,
                "temperature": temperature,
                "temperature_hex": temperature.hex() if temperature is not None else None,
                "top_k": top_k,
                "seed": seed,
            }
            for sample_id, prompt, count, mode, temperature, top_k, seed in QUALITATIVE_MATRIX
        ],
    }


def _validate_implementation_authority(
    root: Path,
    value: object,
    *,
    require_live: bool = True,
) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        ("implementation_commit", "source_sha256s", "test_sha256s"),
        field="authority",
    )
    commit = mapping["implementation_commit"]
    source_paths = (
        "src/sebgpt/inference/__init__.py",
        "src/sebgpt/inference/phase9_types.py",
        "src/sebgpt/inference/phase9_checkpoint.py",
        "src/sebgpt/inference/phase9_generation.py",
        "src/sebgpt/inference/phase9_evaluation.py",
        "src/sebgpt/inference/phase9_experiment.py",
    )
    test_paths = ("tests/test_phase9.py",)
    if type(commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    for name, paths in (("source_sha256s", source_paths), ("test_sha256s", test_paths)):
        values = mapping[name]
        if type(values) is not dict or set(values) != set(paths) or any(not _is_digest(values[path]) for path in paths):
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
        for path in paths:
            committed = _git(root, "show", f"{commit}:{path}", binary=True)
            if (
                not isinstance(committed, bytes)
                or hashlib.sha256(committed).hexdigest() != values[path]
                or (
                    require_live
                    and hashlib.sha256((root / path).read_bytes()).hexdigest()
                    != values[path]
                )
            ):
                raise Phase9GovernanceError("phase9.governance.repository", field="record")
    return mapping


def _validate_marker_policy(value: object, evaluation_id: str, *, sealed: bool) -> None:
    mapping = _exact_mapping(
        value,
        ("schema_version", "relative_path", "maximum_bytes", "no_clobber", "commit_point"),
        field="configuration",
    )
    filename = "phase9-sealed-test-access.json" if sealed else "phase9-development-attempt.json"
    if mapping != {
        "schema_version": 1,
        "relative_path": f"experiments/{evaluation_id}/artifacts/{filename}",
        "maximum_bytes": 16_384,
        "no_clobber": True,
        "commit_point": "known_present",
    }:
        raise Phase9GovernanceError("phase9.governance.repository", field="configuration")


def _validate_evidence_policy(value: object, evaluation_id: str, *, sealed: bool) -> None:
    mapping = _exact_mapping(
        value,
        (
            "schema_version", "completed_relative_path", "failure_relative_path",
            "maximum_bytes", "canonical_json", "generated_text_storage",
            "tracked_record_storage",
        ),
        field="configuration",
    )
    prefix = "phase9-sealed-test" if sealed else "phase9-development"
    maximum = 262_144 if sealed else 1_048_576
    expected_generated_storage = (
        "none" if sealed else "ignored_immutable_machine_readable_evidence"
    )
    if (
        type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
        or mapping["completed_relative_path"] != f"experiments/{evaluation_id}/artifacts/{prefix}-evidence.json"
        or mapping["failure_relative_path"] != f"experiments/{evaluation_id}/artifacts/{prefix}-failure.json"
        or mapping["maximum_bytes"] != maximum
        or mapping["canonical_json"] != "utf8_ensure_ascii_sorted_keys_indent_2_allow_nan_false_final_lf"
        or mapping["generated_text_storage"] != expected_generated_storage
        or mapping["tracked_record_storage"] != "references_and_hashes_without_generated_or_sealed_text"
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="configuration")


def _record_introduced_by_commit(root: Path, record: _Record, commit: str) -> bool:
    try:
        content = _git(root, "show", f"{commit}:EXPERIMENT_LOG.md", binary=True)
    except Phase9GovernanceError:
        return False
    if not isinstance(content, bytes) or content.count(record.raw) != 1:
        return False
    try:
        parent = _git(root, "show", f"{commit}^:EXPERIMENT_LOG.md", binary=True)
    except Phase9GovernanceError:
        parent = b""
    return isinstance(parent, bytes) and record.raw not in parent


def _validate_plan_record(
    root: Path,
    plan: _Record,
    *,
    require_live_implementation: bool = True,
) -> dict[str, object]:
    values = plan.values
    evaluation_id = plan.evaluation_id
    if (
        values["Evaluation ID"] != evaluation_id
        or values["Entry kind"] != "development_plan"
        or values["Scope"] != "development"
        or values["Status"] != "planned"
        or values["Question"] != DEVELOPMENT_QUESTION
        or (values["Predecessor evaluation"] is not None and (type(values["Predecessor evaluation"]) is not str or not ID_PATTERN.fullmatch(values["Predecessor evaluation"])))
        or values["Phase 8 authority"] != _PHASE8_AUTHORITY
        or values["Checkpoint authority"] != _CHECKPOINT_AUTHORITY
        or values["Phase 9 contract authority"] != _CONTRACT_AUTHORITY
        or values["Runtime identity"] != _live_runtime_mapping(root)
        or values["Dataset authority"] != _DATASET_AUTHORITY
        or values["Tokenizer authority"] != _TOKENIZER_AUTHORITY
        or values["Model authority"] != _MODEL_AUTHORITY
        or values["Evaluation configuration"] != _configuration_mapping()
        or values["Qualitative matrix"] != _configuration_mapping()["qualitative_matrix"]
        or values["Sealed-test access"] != "none"
        or values["Success policy"] != DEVELOPMENT_SUCCESS_POLICY
        or values["Next gate"] != "independent_pre_registration_review"
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    implementation = _validate_implementation_authority(
        root,
        values["Phase 9 implementation authority"],
        require_live=require_live_implementation,
    )
    _validate_marker_policy(values["Attempt marker policy"], evaluation_id, sealed=False)
    _validate_evidence_policy(values["Evidence policy"], evaluation_id, sealed=False)
    return implementation


def _validate_laplace_result_reference(value: object, evaluation_id: str) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "schema_version", "baseline_kind", "vocabulary_size",
            "training_target_count", "raw_counts_sha256", "smoothed_counts_sha256",
            "probabilities_hex_sha256", "top_token_id", "baseline_sha256", "evidence",
        ),
    )
    evidence = _reference_from_mapping(mapping["evidence"])
    if (
        type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
        or mapping["baseline_kind"] != "training_laplace_unigram_add_one"
        or mapping["vocabulary_size"] != 81
        or mapping["training_target_count"] != 792_699
        or any(not _is_digest(mapping[name]) for name in (
            "raw_counts_sha256", "smoothed_counts_sha256",
            "probabilities_hex_sha256", "baseline_sha256",
        ))
        or type(mapping["top_token_id"]) is not int
        or not 0 <= mapping["top_token_id"] < 81
        or evidence.evaluation_id != evaluation_id
        or evidence.kind not in ("phase9_development", "phase9_development_failure")
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    return mapping


def _validate_sample_reference(
    value: object,
    evaluation_id: str,
    index: int,
) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "sample_index", "sample_id", "configuration_sha256", "prompt_sha256",
            "checkpoint_sha256", "mode", "temperature_hex", "top_k", "seed",
            "generated_token_count", "generated_byte_count", "generated_code_point_count",
            "generated_text_sha256", "full_text_sha256",
            "local_generator_final_state_sha256", "evidence",
        ),
    )
    if not 0 <= index < len(QUALITATIVE_MATRIX):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    expected = QUALITATIVE_MATRIX[index]
    config = _configuration_mapping()["qualitative_matrix"][index]
    config_sha = hashlib.sha256(_canonical_json(config, pretty=False)).hexdigest()
    evidence = _reference_from_mapping(mapping["evidence"])
    sample_id, prompt, count, mode, temperature, top_k, seed = expected
    if (
        mapping["sample_index"] != index
        or mapping["sample_id"] != sample_id
        or mapping["configuration_sha256"] != config_sha
        or mapping["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        or mapping["checkpoint_sha256"] != CHECKPOINT_SHA256
        or mapping["mode"] != mode
        or mapping["temperature_hex"] != (temperature.hex() if isinstance(temperature, float) else None)
        or mapping["top_k"] != top_k
        or mapping["seed"] != seed
        or mapping["generated_token_count"] != count
        or type(mapping["generated_byte_count"]) is not int or mapping["generated_byte_count"] <= 0
        or mapping["generated_code_point_count"] != count
        or any(not _is_digest(mapping[name]) for name in (
            "generated_text_sha256", "full_text_sha256",
        ))
        or (mode == "greedy" and mapping["local_generator_final_state_sha256"] is not None)
        or (mode == "categorical" and not _is_digest(mapping["local_generator_final_state_sha256"]))
        or evidence.evaluation_id != evaluation_id
        or evidence.kind not in ("phase9_development", "phase9_development_failure")
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    return mapping


def _validate_development_result_record(record: _Record) -> None:
    values = record.values
    status = values["Status"]
    if (
        values["Evaluation ID"] != record.evaluation_id
        or values["Entry kind"] != "development_result"
        or values["Scope"] != "development"
        or status not in ("completed", "failed", "uncertain")
        or not all(_is_timestamp(values[name]) for name in ("Recorded at UTC", "Started at UTC", "Finished at UTC"))
        or not values["Started at UTC"] <= values["Finished at UTC"] <= values["Recorded at UTC"]
        or (values["Predecessor evaluation"] is not None and (type(values["Predecessor evaluation"]) is not str or not ID_PATTERN.fullmatch(values["Predecessor evaluation"])))
        or any(not _is_digest(values[name]) for name in (
            "Development plan record SHA-256", "Development authorization record SHA-256",
        ))
        or any(type(values[name]) is not str or not re.fullmatch(r"[0-9a-f]{40}", values[name]) for name in (
            "Development pre-registration commit", "Development authorization commit",
        ))
        or values["Sealed-test access"] != "none"
        or type(values["Observations"]) is not list or any(type(item) is not str for item in values["Observations"])
        or type(values["Conclusions"]) is not list or any(type(item) is not str for item in values["Conclusions"])
        or values["Limitations"] != list(LIMITATIONS)
        or values["Next gate"] != ("independent_development_result_review" if status == "completed" else "adjudicate_failure_and_plan_successor")
        or type(values["Qualitative samples"]) is not list
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    marker = values["Attempt marker"]
    evidence_value = values["Development evidence"]
    if marker is not None:
        marker_reference = _reference_from_mapping(marker)
        if marker_reference.evaluation_id != record.evaluation_id or marker_reference.kind != "phase9_development_attempt":
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
    if evidence_value is not None:
        evidence = _reference_from_mapping(evidence_value)
        expected_kind = "phase9_development" if status == "completed" else "phase9_development_failure"
        if evidence.evaluation_id != record.evaluation_id or evidence.kind != expected_kind:
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
    for name, predictor in (
        ("Model validation", "mini_gpt_best_validation"),
        ("Uniform validation", "uniform_81"),
        ("Laplace unigram validation", "training_laplace_unigram_add_one"),
    ):
        if values[name] is not None:
            _validate_metric_mapping(values[name], split="validation", predictor=predictor)
    if values["Laplace unigram baseline"] is not None:
        baseline_reference = _validate_laplace_result_reference(
            values["Laplace unigram baseline"], record.evaluation_id
        )
        if evidence_value is not None and baseline_reference["evidence"] != evidence_value:
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
    for index, sample in enumerate(values["Qualitative samples"]):
        sample_reference = _validate_sample_reference(sample, record.evaluation_id, index)
        if evidence_value is not None and sample_reference["evidence"] != evidence_value:
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
    if status == "completed":
        if (
            marker is None or evidence_value is None or values["Failure"] is not None
            or any(values[name] is None for name in ("Model validation", "Uniform validation", "Laplace unigram baseline", "Laplace unigram validation"))
            or len(values["Qualitative samples"]) != 6
        ):
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
    elif status == "failed":
        if marker is None or values["Failure"] is None:
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
        _validate_failure_mapping(values["Failure"], sealed=False)
    else:
        failure = _exact_mapping(values["Failure"], ("stage", "exception_type", "invariant", "safe_facts"))
        if (
            failure["stage"] != "attempt_uncertain"
            or failure["exception_type"] != "ProcessInterruption"
            or failure["invariant"] != "phase9.evidence.amendment"
            or type(failure["safe_facts"]) is not dict
            or set(failure["safe_facts"]) != {"attempt_marker_present", "final_evidence_present"}
            or any(type(item) is not bool for item in failure["safe_facts"].values())
        ):
            raise Phase9GovernanceError("phase9.governance.repository", field="record")


def _validate_supersession_record(record: _Record, prior_result: _Record, successor: str) -> None:
    values = record.values
    if (
        values["Evaluation ID"] != record.evaluation_id
        or values["Entry kind"] != "development_supersession"
        or values["Scope"] != "development"
        or values["Status"] != "superseded"
        or values["Prior terminal status"] != prior_result.values["Status"]
        or values["Prior result record SHA-256"] != prior_result.sha256
        or values["Successor evaluation ID"] != successor
        or values["Reason"] not in ("corrected_implementation", "corrected_contract", "operational_replacement")
        or not _is_timestamp(values["Recorded at UTC"])
        or values["Next gate"] != "review_successor_plan"
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")


def _expected_baseline_result_reference(
    baseline: dict[str, object],
    evidence: Phase9EvidenceReference,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "baseline_kind": "training_laplace_unigram_add_one",
        "vocabulary_size": baseline["vocabulary_size"],
        "training_target_count": baseline["training_target_count"],
        "raw_counts_sha256": hashlib.sha256(_canonical_json(baseline["raw_counts"], pretty=False)).hexdigest(),
        "smoothed_counts_sha256": hashlib.sha256(_canonical_json(baseline["smoothed_counts"], pretty=False)).hexdigest(),
        "probabilities_hex_sha256": hashlib.sha256(_canonical_json(baseline["probabilities_hex"], pretty=False)).hexdigest(),
        "top_token_id": baseline["top_token_id"],
        "baseline_sha256": hashlib.sha256(_canonical_json(baseline, pretty=False)).hexdigest(),
        "evidence": asdict(evidence),
    }


def _expected_sample_reference(
    sample: dict[str, object],
    index: int,
    evidence: Phase9EvidenceReference,
) -> dict[str, object]:
    configuration = {
        name: sample[name]
        for name in (
            "sample_id", "prompt", "generated_token_count", "mode", "temperature",
            "temperature_hex", "top_k", "seed",
        )
    }
    return {
        "sample_index": index,
        "sample_id": sample["sample_id"],
        "configuration_sha256": hashlib.sha256(_canonical_json(configuration, pretty=False)).hexdigest(),
        "prompt_sha256": sample["prompt_sha256"],
        "checkpoint_sha256": sample["checkpoint_sha256"],
        "mode": sample["mode"],
        "temperature_hex": sample["temperature_hex"],
        "top_k": sample["top_k"],
        "seed": sample["seed"],
        "generated_token_count": sample["generated_token_count"],
        "generated_byte_count": sample["generated_byte_count"],
        "generated_code_point_count": sample["generated_code_point_count"],
        "generated_text_sha256": sample["generated_text_sha256"],
        "full_text_sha256": sample["full_text_sha256"],
        "local_generator_final_state_sha256": sample["local_generator_final_state_sha256"],
        "evidence": asdict(evidence),
    }


def _validate_sealed_result_record(record: _Record) -> None:
    values = record.values
    status = values["Status"]
    if (
        values["Evaluation ID"] != record.evaluation_id
        or values["Entry kind"] != "sealed_test_result"
        or values["Scope"] != "sealed_test"
        or status not in ("completed", "failed", "uncertain")
        or not all(_is_timestamp(values[name]) for name in ("Recorded at UTC", "Started at UTC", "Finished at UTC"))
        or not values["Started at UTC"] <= values["Finished at UTC"] <= values["Recorded at UTC"]
        or any(not _is_digest(values[name]) for name in (
            "Sealed plan record SHA-256", "Sealed authorization record SHA-256",
        ))
        or any(type(values[name]) is not str or not re.fullmatch(r"[0-9a-f]{40}", values[name]) for name in (
            "Sealed pre-registration commit", "Sealed authorization commit",
        ))
        or type(values["Observations"]) is not list or any(type(item) is not str for item in values["Observations"])
        or type(values["Conclusions"]) is not list or any(type(item) is not str for item in values["Conclusions"])
        or values["Limitations"] != list(LIMITATIONS)
        or values["Next gate"] != "independent_sealed_result_review"
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    marker_value = values["Access marker"]
    evidence_value = values["Sealed evidence"]
    marker = _reference_from_mapping(marker_value) if marker_value is not None else None
    evidence = _reference_from_mapping(evidence_value) if evidence_value is not None else None
    if marker is not None and (
        marker.evaluation_id != record.evaluation_id
        or marker.kind != "phase9_sealed_test_access"
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    if evidence is not None:
        expected_kind = "phase9_sealed_test" if status == "completed" else "phase9_sealed_test_failure"
        if evidence.evaluation_id != record.evaluation_id or evidence.kind != expected_kind:
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    metrics = (
        ("Model test", "mini_gpt_best_validation"),
        ("Uniform test", "uniform_81"),
        ("Laplace unigram test", "training_laplace_unigram_add_one"),
    )
    for name, predictor in metrics:
        if values[name] is not None:
            _validate_metric_mapping(values[name], split="test", predictor=predictor)
    if status == "completed":
        if (
            marker is None or evidence is None
            or any(values[name] is None for name, _ in metrics)
            or values["Sealed-test access"] != "one_shot_completed"
            or values["Failure"] is not None
        ):
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    elif status == "failed":
        if (
            marker is None
            or values["Sealed-test access"] != "one_shot_consumed_failed"
            or values["Failure"] is None
        ):
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
        _validate_failure_mapping(values["Failure"], sealed=True)
    else:
        failure = _exact_mapping(values["Failure"], ("stage", "exception_type", "invariant", "safe_facts"))
        if (
            values["Sealed-test access"] != "one_shot_consumed_uncertain"
            or failure["stage"] not in ("attempt_uncertain", "sealed_access_marker_publication")
            or failure["invariant"] not in ("phase9.evidence.amendment", "phase9.evidence.publication")
        ):
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
        facts = failure["safe_facts"]
        if failure["stage"] == "attempt_uncertain":
            if (
                failure["exception_type"] != "ProcessInterruption"
                or type(facts) is not dict
                or set(facts) != {"attempt_marker_present", "final_evidence_present"}
                or any(type(item) is not bool for item in facts.values())
            ):
                raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
        elif (
            failure["exception_type"] != "Phase9PublicationError"
            or type(facts) is not dict
            or set(facts) != {"operation", "publication_state"}
            or facts["publication_state"] != UNKNOWN
            or facts["operation"] not in {
                "open_parent", "create_temporary", "write", "fsync_file",
                "publish_no_clobber", "fsync_parent", "revalidate_identity",
            }
        ):
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")


def _validate_development_launch(root: Path, plan: _Record, authorization: _Record) -> str:
    implementation = _validate_plan_record(root, plan)
    if (
        authorization.values["Evaluation ID"] != plan.evaluation_id
        or authorization.values["Entry kind"] != "development_authorization"
        or authorization.values["Scope"] != "development"
        or authorization.values["Status"] != "authorized_once"
        or authorization.values["Development plan record SHA-256"] != plan.sha256
        or authorization.values["Phase 9 implementation commit"] != implementation["implementation_commit"]
        or authorization.values["Authorization"] != DEVELOPMENT_AUTHORIZATION
        or authorization.values["Next gate"] != "execute_authorized_development_attempt"
        or authorization.values["Checkpoint SHA-256"] != CHECKPOINT_SHA256
    ):
        raise Phase9GovernanceError("phase9.governance.repository", field="configuration")
    head = str(_git(root, "rev-parse", "HEAD"))
    for field in ("Development pre-registration commit", "Phase 9 implementation commit"):
        commit = authorization.values[field]
        if type(commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise Phase9GovernanceError("phase9.governance.repository", field="record")
        try:
            _git(root, "merge-base", "--is-ancestor", commit, head)
        except Phase9GovernanceError:
            raise Phase9GovernanceError("phase9.governance.repository", field="ancestor") from None
    pre_registration = authorization.values["Development pre-registration commit"]
    assert isinstance(pre_registration, str)
    if not _record_introduced_by_commit(root, plan, pre_registration) or not _record_introduced_by_commit(root, authorization, head):
        raise Phase9GovernanceError("phase9.governance.repository", field="record")
    return head


def _validate_sealed_launch(
    root: Path,
    records: tuple[_Record, ...],
    evaluation_id: str,
    development_result: _Record,
    plan: _Record,
    authorization: _Record,
    development: dict[str, object],
    development_digest: str,
) -> tuple[str, dict[str, object]]:
    _validate_development_result_record(development_result)
    if (
        development_result.values["Status"] != "completed"
        or development_result.values["Failure"] is not None
        or development_result.values["Development evidence"] is None
        or any(
            development_result.values[name] is None
            for name in (
                "Model validation", "Uniform validation", "Laplace unigram baseline",
                "Laplace unigram validation",
            )
        )
        or type(development_result.values["Qualitative samples"]) is not list
        or len(development_result.values["Qualitative samples"]) != 6
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    development_plan = _one(records, evaluation_id, "development_plan")
    development_authorization = _one(records, evaluation_id, "development_authorization")
    implementation = _validate_plan_record(root, development_plan)
    if (
        development_authorization.values["Evaluation ID"] != evaluation_id
        or development_authorization.values["Entry kind"] != "development_authorization"
        or development_authorization.values["Scope"] != "development"
        or development_authorization.values["Status"] != "authorized_once"
        or development_authorization.values["Development plan record SHA-256"] != development_plan.sha256
        or development_authorization.values["Phase 9 implementation commit"] != implementation["implementation_commit"]
        or development_authorization.values["Checkpoint SHA-256"] != CHECKPOINT_SHA256
        or development_authorization.values["Authorization"] != DEVELOPMENT_AUTHORIZATION
        or development_authorization.values["Next gate"] != "execute_authorized_development_attempt"
        or development_result.values["Development plan record SHA-256"] != development_plan.sha256
        or development_result.values["Development authorization record SHA-256"] != development_authorization.sha256
        or development_result.values["Development pre-registration commit"] != development_authorization.values["Development pre-registration commit"]
        or development_result.values["Development authorization commit"] is None
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    development_commit = development_result.values["Development authorization commit"]
    assert isinstance(development_commit, str)
    expected_development_authority = _development_authority_mapping(
        development_plan, development_authorization, development_commit
    )
    _validate_development_evidence_mapping(
        development,
        evaluation_id=evaluation_id,
        marker_sha256=_reference_from_mapping(development_result.values["Attempt marker"]).sha256,
        authority=expected_development_authority,
        configuration=_configuration_mapping(),
        completed=True,
    )
    evidence_reference = _reference_from_mapping(development_result.values["Development evidence"])
    if (
        evidence_reference.kind != "phase9_development"
        or evidence_reference.status != "completed"
        or evidence_reference.sha256 != development_digest
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="evidence")
    baseline_mapping = development["laplace_unigram_baseline"]
    samples = development["samples"]
    assert isinstance(baseline_mapping, dict) and isinstance(samples, list)
    if (
        development_result.values["Model validation"] != development["model_validation"]
        or development_result.values["Uniform validation"] != development["uniform_validation"]
        or development_result.values["Laplace unigram validation"] != development["laplace_unigram_validation"]
        or development_result.values["Laplace unigram baseline"]
        != _expected_baseline_result_reference(baseline_mapping, evidence_reference)
        or development_result.values["Qualitative samples"]
        != [
            _expected_sample_reference(sample, index, evidence_reference)
            for index, sample in enumerate(samples)
        ]
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    if (
        plan.values["Evaluation ID"] != evaluation_id
        or plan.values["Entry kind"] != "sealed_test_plan"
        or plan.values["Scope"] != "sealed_test"
        or plan.values["Status"] != "planned"
        or plan.values["Accepted development result commit"] is None
        or plan.values["Development result record SHA-256"] != development_result.sha256
        or plan.values["Development evidence SHA-256"] != development_digest
        or plan.values["Frozen authority"] != expected_development_authority
        or plan.values["Frozen evaluation configuration"] != _configuration_mapping()
        or plan.values["Frozen qualitative matrix"] != _configuration_mapping()["qualitative_matrix"]
        or plan.values["Interpretation policy"] != SEALED_INTERPRETATION_POLICY
        or plan.values["Sealed-test access"] != "pending_explicit_authorization"
        or plan.values["Next gate"] != "independent_sealed_plan_review"
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    _validate_marker_policy(plan.values["Access marker policy"], evaluation_id, sealed=True)
    _validate_evidence_policy(plan.values["Sealed evidence policy"], evaluation_id, sealed=True)
    if (
        authorization.values["Evaluation ID"] != evaluation_id
        or authorization.values["Entry kind"] != "sealed_test_authorization"
        or authorization.values["Scope"] != "sealed_test"
        or authorization.values["Status"] != "authorized_once"
        or authorization.values["Sealed plan record SHA-256"] != plan.sha256
        or authorization.values["Development evidence SHA-256"] != development_digest
        or authorization.values["Phase 9 implementation commit"] != implementation["implementation_commit"]
        or authorization.values["Checkpoint SHA-256"] != CHECKPOINT_SHA256
        or authorization.values["Authorization"] != SEALED_AUTHORIZATION
        or authorization.values["Next gate"] != "execute_authorized_sealed_test_once"
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    head = str(_git(root, "rev-parse", "HEAD"))
    development_pre_registration = development_authorization.values["Development pre-registration commit"]
    development_authorization_commit = development_result.values["Development authorization commit"]
    development_result_commit = plan.values["Accepted development result commit"]
    sealed_pre_registration = authorization.values["Sealed pre-registration commit"]
    implementation_commit = authorization.values["Phase 9 implementation commit"]
    chain = (
        implementation_commit,
        development_pre_registration,
        development_authorization_commit,
        development_result_commit,
        sealed_pre_registration,
        head,
    )
    if (
        any(type(commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", commit) for commit in chain)
        or len(set(chain)) != len(chain)
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    for commit in (
        plan.values["Accepted development result commit"],
        authorization.values["Sealed pre-registration commit"],
        authorization.values["Phase 9 implementation commit"],
        development_pre_registration,
        development_authorization_commit,
    ):
        if type(commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
        try:
            _git(root, "merge-base", "--is-ancestor", commit, head)
        except Phase9GovernanceError:
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="ancestor") from None
    for earlier, later in zip(chain, chain[1:]):
        try:
            _git(root, "merge-base", "--is-ancestor", earlier, later)
        except Phase9GovernanceError:
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="ancestor") from None
    if (
        not _record_introduced_by_commit(root, development_plan, development_pre_registration)
        or not _record_introduced_by_commit(root, development_authorization, development_authorization_commit)
        or not _record_introduced_by_commit(root, development_result, plan.values["Accepted development result commit"])
        or not _record_introduced_by_commit(root, plan, authorization.values["Sealed pre-registration commit"])
        or not _record_introduced_by_commit(root, authorization, head)
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    return head, expected_development_authority


def _sealed_test_metadata(root: Path) -> dict[str, object]:
    paths = (
        (root / "docs/data/shakespeare-eight-play-manifest.json", "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"),
        (root / "data/processed/shakespeare-eight-play/processing-manifest.json", "bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc"),
    )
    parsed: list[dict[str, object]] = []
    for path, expected_sha in paths:
        try:
            content = path.read_bytes()
            value = json.loads(content.decode("utf-8", errors="strict"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="manifest") from None
        if hashlib.sha256(content).hexdigest() != expected_sha or type(value) is not dict:
            raise Phase9GovernanceError("phase9.governance.sealed_test", field="manifest")
        parsed.append(value)
    public_works = parsed[0].get("works")
    processing_works = parsed[1].get("works")
    if type(public_works) is not list or type(processing_works) is not list:
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="manifest")
    public = next((item for item in public_works if type(item) is dict and item.get("order") == 8), None)
    processing = next((item for item in processing_works if type(item) is dict and item.get("manifest_order") == 8), None)
    expected_path = "data/processed/shakespeare-eight-play/test/twelfth-night.txt"
    if (
        type(public) is not dict or type(processing) is not dict
        or public.get("work_id") != "twelfth-night" or public.get("split") != "test"
        or public.get("processed_path") != expected_path
        or processing.get("work_id") != "twelfth-night" or processing.get("split") != "test"
        or processing.get("output_path") != expected_path
        or type(processing.get("processed")) is not dict
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="manifest")
    result = processing["processed"]
    assert isinstance(result, dict)
    if result != {
        "byte_count": 116_285,
        "code_point_count": 114_812,
        "line_count": 4_455,
        "sha256": "ce0cdd5c9c096365c5beed105971a9f2187b30a1be3acd721b0fb6003b260ff6",
        "word_count": 21_240,
    }:
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="manifest")
    return {"relative_path": expected_path, **result}


def _read_sealed_test_once(root: Path, expected: dict[str, object]) -> str:
    components = tuple(str(expected["relative_path"]).split("/"))
    if components != (
        "data", "processed", "shakespeare-eight-play", "test", "twelfth-night.txt"
    ):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="path")
    fds: list[int] = []
    entries: list[tuple[int, str, tuple[int, int, int]]] = []
    file_fd: int | None = None
    content_validation_started = False
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(root_fd)
        parent = root_fd
        for component in components[:-1]:
            child, identity = _open_child_directory(parent, component, create=False)
            fds.append(child)
            entries.append((parent, component, identity))
            parent = child
        file_fd = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        content_validation_started = True
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected["byte_count"]:
            raise OSError
        content = bytearray()
        maximum = int(expected["byte_count"]) + 1
        while len(content) < maximum:
            chunk = os.read(file_fd, min(1_048_576, maximum - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        after = os.fstat(file_fd)
        if (
            len(content) != expected["byte_count"]
            or hashlib.sha256(content).hexdigest() != expected["sha256"]
            or (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mode, after.st_size)
        ):
            raise OSError
        entry = os.stat(components[-1], dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISREG(entry.st_mode)
            or (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_size)
            != (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        ):
            raise OSError
        _revalidate_publication_directories(entries)
        _revalidate_root(root, fds[0])
        text = bytes(content).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise Phase9GovernanceError("phase9.governance.dataset", field="utf8") from None
    except OSError:
        raise Phase9GovernanceError(
            "phase9.governance.dataset",
            field="bytes" if content_validation_started else "path",
        ) from None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(fds):
            os.close(directory_fd)
    if (
        "\r" in text
        or not text.endswith("\n")
        or len(text) != expected["code_point_count"]
        or text.count("\n") != expected["line_count"]
    ):
        raise Phase9GovernanceError("phase9.governance.dataset", field="normalization")
    return text


def _require_artifacts_absent(root: Path, evaluation_id: str, names: tuple[str, ...]) -> None:
    try:
        fds, _ = _open_artifact_directory(root, evaluation_id, create=False)
    except FileNotFoundError:
        return
    except OSError:
        raise Phase9GovernanceError(
            "phase9.governance.repository", field="evaluation_id"
        ) from None
    try:
        if any(_inspect_fd_entry(fds[-1], name) != KNOWN_ABSENT for name in names):
            raise Phase9GovernanceError(
                "phase9.governance.repository", field="evaluation_id"
            )
    finally:
        for directory_fd in reversed(fds):
            os.close(directory_fd)


def _documents(bundle: object, corpus: object) -> tuple[tuple[Phase9TokenDocument, ...], tuple[Phase9TokenDocument, ...]]:
    training = tuple(
        Phase9TokenDocument(work.work_id, work.manifest_order, "train", bundle.tokenizer.encode(work.processed_text))
        for work in corpus.training_works
    )
    validation = tuple(
        Phase9TokenDocument(work.work_id, work.manifest_order, "validation", bundle.tokenizer.encode(work.processed_text))
        for work in corpus.validation_works
    )
    return training, validation


def _load_development_corpus(root: Path) -> object:
    try:
        return load_phase4_permitted_corpus(root)
    except (Phase4GovernanceError, Phase4TypeError):
        raise Phase9GovernanceError("phase9.governance.dataset", field="manifest") from None


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json_artifact(path: Path, *, maximum: int, expected_sha256: str) -> dict[str, object]:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise OSError
        content = path.read_bytes()
    except OSError:
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence") from None
    if len(content) != metadata.st_size or hashlib.sha256(content).hexdigest() != expected_sha256:
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence")
    try:
        value = json.loads(content.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence") from None
    if type(value) is not dict or _canonical_json(value) != content:
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence")
    return value


def _read_evidence_artifact(
    root: Path,
    evaluation_id: str,
    filename: str,
    *,
    maximum: int,
    expected_sha256: str,
) -> dict[str, object]:
    fds: list[int] = []
    try:
        fds, entries = _open_artifact_directory(root, evaluation_id, create=False)
        artifacts_fd = fds[-1]
        file_fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=artifacts_fd)
    except OSError:
        for directory_fd in reversed(fds):
            os.close(directory_fd)
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence") from None
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise OSError
        content = bytearray()
        while len(content) <= maximum:
            chunk = os.read(file_fd, min(1_048_576, maximum + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        after = os.fstat(file_fd)
        entry = os.stat(filename, dir_fd=artifacts_fd, follow_symlinks=False)
        if (
            len(content) != before.st_size
            or hashlib.sha256(content).hexdigest() != expected_sha256
            or (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mode, after.st_size)
            or (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_size)
            != (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        ):
            raise OSError
        _revalidate_publication_directories(entries)
        _revalidate_root(root, fds[0])
    except OSError:
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence") from None
    finally:
        os.close(file_fd)
        for directory_fd in reversed(fds):
            os.close(directory_fd)
    try:
        value = json.loads(bytes(content).decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence") from None
    if type(value) is not dict or _canonical_json(value) != bytes(content):
        raise Phase9GovernanceError("phase9.governance.repository", field="evidence")
    return value


def _failure(stage: str, error: Exception) -> dict[str, object]:
    details = getattr(error, "details", {})
    invariant = details.get("invariant")
    if invariant is None:
        invariant = getattr(error, "invariant", None)
    if isinstance(error, UnknownCodePointError):
        invariant = "encode.unknown_code_point"
    elif isinstance(error, InvalidTokenIdError):
        invariant = "decode.invalid_token_id"
    if type(invariant) is not str:
        invariant = "phase9.evidence.schema"
    field = details.get("field", "evidence")
    if isinstance(error, UnknownCodePointError):
        field = "prompt"
    elif isinstance(error, InvalidTokenIdError):
        field = "output"
    if field not in _FAILURE_FIELD_VOCABULARY:
        field = {
            "rank": "logits", "shape": "logits", "dtype": "logits",
            "device": "logits", "finite": "logits", "baseline": "counts",
            "metric": "evidence", "perplexity": "evidence", "predictor": "evidence",
            "work": "work_identity", "run": "checkpoint", "optimizer": "payload",
            "progress": "payload", "lineage": "payload", "bundle": "model",
            "tokenizer": "vocabulary", "dataset": "manifest", "windows": "documents",
        }.get(field, "evidence")
    if stage in ("development_evidence_publication", "sealed_evidence_publication"):
        facts: dict[str, object] = {
            "operation": details.get("operation", "publish_no_clobber"),
            "publication_state": details.get("publication_state", UNKNOWN),
        }
    elif stage in _SAMPLE_STAGE_TO_ID:
        facts = {
            "sample_id": _SAMPLE_STAGE_TO_ID[stage],
            "field": field,
            "completed_token_count": 0,
        }
    elif stage == "development_corpus":
        facts = {"field": field, "position": details.get("position")}
    elif stage in (
        "model_validation", "uniform_validation", "laplace_unigram_validation",
        "cross_predictor_comparison",
    ):
        facts = {"field": field, "split": "validation"}
    elif stage == "sealed_tokenization":
        facts = {
            "field": field,
            "position": details.get("position", getattr(error, "position", None)),
        }
    elif stage in (
        "sealed_model_evaluation", "sealed_uniform_evaluation",
        "sealed_laplace_unigram_evaluation", "sealed_cross_predictor_comparison",
    ):
        facts = {"field": field, "split": "test"}
    else:
        facts = {"field": field}
    return {
        "stage": stage,
        "exception_type": type(error).__name__,
        "invariant": invariant,
        "safe_facts": facts,
    }


def _development_authority_mapping(
    plan: _Record,
    authorization: _Record,
    authorization_commit: str,
) -> dict[str, object]:
    return {
        "phase8_closure_commit": "6d086625cb293f61700bd59b551ea310a5b680b9",
        "phase8_result_commit": "334119e63c716e4922cd0b67da4f5fc221faa886",
        "phase8_experiment_id": "EXP-20260912-01",
        "phase8_implementation_commit": "809834323d53407cb4a54ae539585bb3d78856eb",
        "phase8_contract_commit": "c50d77ac935bdf924b9b429a5776c419982a5d11",
        "phase8_spec_sha256": "1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd",
        "phase9_contract_commit": PHASE9_CONTRACT_COMMIT,
        "phase9_contract_sha256": PHASE9_SPEC_SHA256,
        "phase9_implementation_commit": authorization.values["Phase 9 implementation commit"],
        "development_plan_record_sha256": plan.sha256,
        "development_pre_registration_commit": authorization.values["Development pre-registration commit"],
        "development_authorization_record_sha256": authorization.sha256,
        "development_authorization_commit": authorization_commit,
        "catalog_sha256": "6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241",
        "checkpoint_role": "best_validation",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "vocabulary_sha256": "9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e",
        "dataset_manifest_sha256": "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb",
        "processing_manifest_sha256": "bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc",
        "requirements_lock_sha256": "8ddf7a11bf67d36bce8ba58abffc5416a0fb778cdc20d03c077d318715122cfa",
    }


def _development_evidence(
    evaluation_id: str,
    marker: Phase9EvidenceReference,
    authority: dict[str, object],
    configuration: dict[str, object],
    model_metrics: Phase9Metrics,
    uniform_metrics: Phase9Metrics,
    baseline: object,
    unigram_metrics: Phase9Metrics,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    baseline_mapping = {
        "vocabulary_size": baseline.vocabulary_size,
        "training_target_count": baseline.training_target_count,
        "raw_counts": list(baseline.raw_counts), "smoothed_counts": list(baseline.smoothed_counts),
        "probabilities": list(baseline.probabilities),
        "probabilities_hex": [value.hex() for value in baseline.probabilities],
        "top_token_id": baseline.top_token_id,
    }
    return {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "kind": "phase9_development", "status": "completed", "recorded_at_utc": _now(),
        "attempt_marker_sha256": marker.sha256, "authority": authority,
        "configuration": configuration,
        "model_validation": _metric_mapping(model_metrics),
        "uniform_validation": _metric_mapping(uniform_metrics),
        "laplace_unigram_baseline": baseline_mapping,
        "laplace_unigram_validation": _metric_mapping(unigram_metrics),
        "samples": samples, "sealed_test_access": "none", "failure": None,
        "limitations": list(LIMITATIONS),
    }


def _validate_attempt_marker(
    value: object,
    *,
    evaluation_id: str,
    plan: _Record,
    authorization: _Record,
    authorization_commit: str,
) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "schema_version", "evaluation_id", "kind",
            "development_plan_record_sha256", "development_pre_registration_commit",
            "development_authorization_record_sha256", "development_authorization_commit",
            "phase9_contract_sha256", "phase9_implementation_commit", "checkpoint_sha256",
            "created_at_utc", "consumed",
        ),
    )
    expected = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "kind": "phase9_development_attempt",
        "development_plan_record_sha256": plan.sha256,
        "development_pre_registration_commit": authorization.values["Development pre-registration commit"],
        "development_authorization_record_sha256": authorization.sha256,
        "development_authorization_commit": authorization_commit,
        "phase9_contract_sha256": PHASE9_SPEC_SHA256,
        "phase9_implementation_commit": authorization.values["Phase 9 implementation commit"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "created_at_utc": mapping["created_at_utc"],
        "consumed": True,
    }
    if mapping != expected or not _is_timestamp(mapping["created_at_utc"]):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return mapping


def _validate_development_evidence_mapping(
    value: object,
    *,
    evaluation_id: str,
    marker_sha256: str,
    authority: dict[str, object],
    configuration: dict[str, object],
    completed: bool,
) -> dict[str, object]:
    keys = (
        "schema_version", "evaluation_id", "kind", "status", "recorded_at_utc",
        "attempt_marker_sha256", "authority", "configuration", "model_validation",
        "uniform_validation", "laplace_unigram_baseline", "laplace_unigram_validation",
        "samples", "sealed_test_access", "failure", "limitations",
    ) if completed else (
        "schema_version", "evaluation_id", "kind", "status", "recorded_at_utc",
        "started_at_utc", "finished_at_utc", "attempt_marker_sha256", "authority",
        "configuration", "model_validation", "uniform_validation",
        "laplace_unigram_baseline", "laplace_unigram_validation", "samples",
        "sealed_test_access", "failure", "limitations",
    )
    mapping = _exact_mapping(value, keys)
    if (
        type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
        or mapping["evaluation_id"] != evaluation_id
        or mapping["kind"] != ("phase9_development" if completed else "phase9_development_failure")
        or mapping["status"] != ("completed" if completed else "failed")
        or not _is_timestamp(mapping["recorded_at_utc"])
        or mapping["attempt_marker_sha256"] != marker_sha256
        or mapping["authority"] != authority
        or mapping["configuration"] != configuration
        or mapping["sealed_test_access"] != "none"
        or mapping["limitations"] != list(LIMITATIONS)
        or type(mapping["samples"]) is not list
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    _validate_authority_mapping(mapping["authority"], sealed=False)
    for index, sample in enumerate(mapping["samples"]):
        if index >= len(QUALITATIVE_MATRIX):
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
        _validate_sample_mapping(sample, QUALITATIVE_MATRIX[index])
    metric_specs = (
        ("model_validation", "mini_gpt_best_validation"),
        ("uniform_validation", "uniform_81"),
        ("laplace_unigram_validation", "training_laplace_unigram_add_one"),
    )
    present_metrics: list[Phase9Metrics] = []
    for name, predictor in metric_specs:
        metric = mapping[name]
        if metric is None:
            if completed:
                raise Phase9ContractError("phase9.evidence.schema", field="evidence")
            continue
        parsed = _validate_metric_mapping(metric, split="validation", predictor=predictor)
        present_metrics.append(
            Phase9Metrics(
                parsed["predictor"], parsed["split"], parsed["window_sha256"],
                parsed["nll"], parsed["perplexity"], parsed["correct_count"],
                parsed["target_count"], parsed["top1_accuracy"], parsed["window_count"],
            )
        )
    if len(present_metrics) == 3:
        _compare_metrics(*present_metrics)
        if (
            any(
                metric.target_count != 98_295 or metric.window_count != 384
                for metric in present_metrics
            )
            or present_metrics[0].nll.hex() != "0x1.3f94b678f5807p+1"
        ):
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    baseline = mapping["laplace_unigram_baseline"]
    if baseline is None:
        if completed:
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    else:
        _validate_baseline_mapping(baseline)
    if completed:
        if len(mapping["samples"]) != 6 or mapping["failure"] is not None:
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    else:
        if not all(_is_timestamp(mapping[name]) for name in ("started_at_utc", "finished_at_utc")):
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
        if not mapping["started_at_utc"] <= mapping["finished_at_utc"] <= mapping["recorded_at_utc"]:
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
        _validate_failure_mapping(mapping["failure"], sealed=False)
    return mapping


def _validate_access_marker(
    value: object,
    *,
    evaluation_id: str,
    plan: _Record,
    authorization: _Record,
    authorization_commit: str,
    development_evidence_sha256: str,
) -> dict[str, object]:
    mapping = _exact_mapping(
        value,
        (
            "schema_version", "evaluation_id", "kind", "sealed_plan_record_sha256",
            "sealed_pre_registration_commit", "sealed_authorization_record_sha256",
            "sealed_authorization_commit", "phase9_contract_sha256",
            "phase9_implementation_commit", "checkpoint_sha256",
            "development_evidence_sha256", "created_at_utc", "consumed",
        ),
    )
    if (
        type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
        or mapping["evaluation_id"] != evaluation_id
        or mapping["kind"] != "phase9_sealed_test_access"
        or mapping["sealed_plan_record_sha256"] != plan.sha256
        or mapping["sealed_pre_registration_commit"] != authorization.values["Sealed pre-registration commit"]
        or mapping["sealed_authorization_record_sha256"] != authorization.sha256
        or mapping["sealed_authorization_commit"] != authorization_commit
        or mapping["phase9_contract_sha256"] != PHASE9_SPEC_SHA256
        or mapping["phase9_implementation_commit"] != authorization.values["Phase 9 implementation commit"]
        or mapping["checkpoint_sha256"] != CHECKPOINT_SHA256
        or mapping["development_evidence_sha256"] != development_evidence_sha256
        or not _is_timestamp(mapping["created_at_utc"])
        or mapping["consumed"] is not True
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    return mapping


def _validate_sealed_evidence_mapping(
    value: object,
    *,
    evaluation_id: str,
    marker_sha256: str,
    development_evidence_sha256: str,
    authority: dict[str, object],
    completed: bool,
) -> dict[str, object]:
    keys = (
        "schema_version", "evaluation_id", "kind", "status", "recorded_at_utc",
        "authority", "access_marker_sha256", "development_evidence_sha256",
        "model_test", "uniform_test", "laplace_unigram_test", "sealed_test_access",
        "failure", "limitations",
    ) if completed else (
        "schema_version", "evaluation_id", "kind", "status", "recorded_at_utc",
        "started_at_utc", "finished_at_utc", "authority", "access_marker_sha256",
        "development_evidence_sha256", "model_test", "uniform_test",
        "laplace_unigram_test", "sealed_test_access", "failure", "limitations",
    )
    mapping = _exact_mapping(value, keys)
    if (
        type(mapping["schema_version"]) is not int or mapping["schema_version"] != 1
        or mapping["evaluation_id"] != evaluation_id
        or mapping["kind"] != ("phase9_sealed_test" if completed else "phase9_sealed_test_failure")
        or mapping["status"] != ("completed" if completed else "failed")
        or not _is_timestamp(mapping["recorded_at_utc"])
        or mapping["authority"] != authority
        or mapping["access_marker_sha256"] != marker_sha256
        or mapping["development_evidence_sha256"] != development_evidence_sha256
        or mapping["sealed_test_access"] != ("one_shot_completed" if completed else "one_shot_consumed_failed")
        or mapping["limitations"] != list(LIMITATIONS)
    ):
        raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    _validate_authority_mapping(mapping["authority"], sealed=True)
    specs = (
        ("model_test", "mini_gpt_best_validation"),
        ("uniform_test", "uniform_81"),
        ("laplace_unigram_test", "training_laplace_unigram_add_one"),
    )
    present: list[Phase9Metrics] = []
    for name, predictor in specs:
        metric = mapping[name]
        if metric is None:
            if completed:
                raise Phase9ContractError("phase9.evidence.schema", field="evidence")
            continue
        parsed = _validate_metric_mapping(metric, split="test", predictor=predictor)
        present.append(Phase9Metrics(
            parsed["predictor"], parsed["split"], parsed["window_sha256"], parsed["nll"],
            parsed["perplexity"], parsed["correct_count"], parsed["target_count"],
            parsed["top1_accuracy"], parsed["window_count"],
        ))
    if len(present) == 3:
        _compare_metrics(*present)
    if completed:
        if mapping["failure"] is not None:
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
    else:
        if not all(_is_timestamp(mapping[name]) for name in ("started_at_utc", "finished_at_utc")):
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
        if not mapping["started_at_utc"] <= mapping["finished_at_utc"] <= mapping["recorded_at_utc"]:
            raise Phase9ContractError("phase9.evidence.schema", field="evidence")
        _validate_failure_mapping(mapping["failure"], sealed=True)
    return mapping


def run_fixed_phase9_development_evaluation(
    repository_root: Path,
    *,
    evaluation_id: str,
) -> Phase9EvidenceReference:
    root = _validate_repository(repository_root)
    evaluation_id = _validate_id(evaluation_id)
    try:
        record_bytes = (root / "EXPERIMENT_LOG.md").read_bytes()
    except OSError:
        raise Phase9GovernanceError("phase9.governance.repository", field="record") from None
    records = _parse_selected_chain(record_bytes, evaluation_id)
    plan, authorization = _development_authority(records, evaluation_id)
    _validate_predecessor_chain(root, records, evaluation_id)
    authorization_commit = _validate_development_launch(root, plan, authorization)
    _require_artifacts_absent(
        root,
        evaluation_id,
        (
            "phase9-development-attempt.json",
            "phase9-development-evidence.json",
            "phase9-development-failure.json",
        ),
    )
    authority = _development_authority_mapping(plan, authorization, authorization_commit)
    configuration = _configuration_mapping()
    marker_value = {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "kind": "phase9_development_attempt", "development_plan_record_sha256": plan.sha256,
        "development_pre_registration_commit": authorization.values["Development pre-registration commit"],
        "development_authorization_record_sha256": authorization.sha256,
        "development_authorization_commit": authorization_commit,
        "phase9_contract_sha256": PHASE9_SPEC_SHA256,
        "phase9_implementation_commit": authorization.values["Phase 9 implementation commit"],
        "checkpoint_sha256": CHECKPOINT_SHA256, "created_at_utc": _now(), "consumed": True,
    }
    _validate_attempt_marker(
        marker_value,
        evaluation_id=evaluation_id,
        plan=plan,
        authorization=authorization,
        authorization_commit=authorization_commit,
    )
    marker_publication = _publish(
        root, evaluation_id, filename="phase9-development-attempt.json",
        kind="phase9_development_attempt", status="consumed",
        content=_canonical_json(marker_value),
    )
    if marker_publication.state != KNOWN_PRESENT or marker_publication.reference is None:
        raise Phase9PublicationError("phase9.evidence.publication", operation=marker_publication.operation, publication_state=marker_publication.state)
    marker = marker_publication.reference
    stage = "inference_load"
    model_metrics: Phase9Metrics | None = None
    uniform: Phase9Metrics | None = None
    baseline: object | None = None
    unigram: Phase9Metrics | None = None
    samples: list[dict[str, object]] = []
    try:
        bundle = load_phase9_inference_bundle(root)
        stage = "development_corpus"
        corpus = _load_development_corpus(root)
        training_documents, validation_documents = _documents(bundle, corpus)
        stage = "training_window_build"
        training_windows = build_phase9_windows(training_documents, vocabulary_size=81, context_length=256, stride=256)
        stage = "validation_window_build"
        validation_windows = build_phase9_windows(validation_documents, vocabulary_size=81, context_length=256, stride=256)
        if (len(training_windows), sum(len(w.target_ids) for w in training_windows)) != (3099, 792699) or (len(validation_windows), sum(len(w.target_ids) for w in validation_windows)) != (384, 98295):
            raise Phase9GovernanceError("phase9.governance.dataset", field="counts")
        stage = "laplace_unigram_fit"
        baseline = fit_phase9_laplace_unigram(training_windows, vocabulary_size=81)
        stage = "model_validation"
        model_metrics = evaluate_phase9_model(bundle.model, validation_windows, predictor="mini_gpt_best_validation", split="validation")
        if model_metrics.nll.hex() != "0x1.3f94b678f5807p+1":
            raise Phase9ContractError("phase9.evaluation.metric", field="metric")
        stage = "uniform_validation"
        uniform = evaluate_phase9_uniform(validation_windows, vocabulary_size=81, split="validation")
        stage = "laplace_unigram_validation"
        unigram = evaluate_phase9_laplace_unigram(baseline, validation_windows, split="validation")
        for sample_id, prompt, count, mode, temperature, top_k, seed in QUALITATIVE_MATRIX:
            stage = f"sample_{sample_id.replace('-', '_')}"
            generated = generate_phase9_text(bundle, prompt, generated_token_count=count, mode=mode, temperature=temperature, top_k=top_k, seed=seed)  # type: ignore[arg-type]
            generated_bytes = generated.generated_text.encode("utf-8")
            samples.append({
                "sample_id": sample_id, "prompt": prompt,
                "generated_token_count": count, "mode": mode, "temperature": temperature,
                "temperature_hex": temperature.hex() if temperature is not None else None,
                "top_k": top_k, "seed": seed,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "checkpoint_role": "best_validation", "checkpoint_sha256": CHECKPOINT_SHA256,
                "generated_text": generated.generated_text,
                "generated_byte_count": len(generated_bytes),
                "generated_code_point_count": len(generated.generated_text),
                "generated_text_sha256": hashlib.sha256(generated_bytes).hexdigest(),
                "full_text_sha256": hashlib.sha256(generated.full_text.encode("utf-8")).hexdigest(),
                "local_generator_final_state_sha256": generated.local_generator_final_state_sha256,
            })
        stage = "cross_predictor_comparison"
        _compare_metrics(model_metrics, uniform, unigram)
        evidence = _development_evidence(
            evaluation_id,
            marker,
            authority,
            configuration,
            model_metrics,
            uniform,
            baseline,
            unigram,
            samples,
        )
        _validate_development_evidence_mapping(
            evidence,
            evaluation_id=evaluation_id,
            marker_sha256=marker.sha256,
            authority=authority,
            configuration=configuration,
            completed=True,
        )
        stage = "development_evidence_publication"
        publication = _publish_terminal(root, evaluation_id, filename="phase9-development-evidence.json", opposite_filename="phase9-development-failure.json", kind="phase9_development", status="completed", content=_canonical_json(evidence))
        if publication.state != KNOWN_PRESENT or publication.reference is None:
            raise Phase9PublicationError("phase9.evidence.publication", operation=publication.operation, publication_state=publication.state)
        return publication.reference
    except _KNOWN_EXECUTION_ERRORS as error:
        if stage == "development_evidence_publication" and getattr(error, "details", {}).get("publication_state") == UNKNOWN:
            raise
        finished_at = _now()
        recorded_at = _now()
        failure_value = {
            "schema_version": 1, "evaluation_id": evaluation_id,
            "kind": "phase9_development_failure", "status": "failed",
            "recorded_at_utc": recorded_at, "started_at_utc": marker_value["created_at_utc"],
            "finished_at_utc": finished_at, "attempt_marker_sha256": marker.sha256,
            "authority": authority, "configuration": configuration,
            "model_validation": _metric_mapping(model_metrics) if model_metrics is not None else None,
            "uniform_validation": _metric_mapping(uniform) if uniform is not None else None,
            "laplace_unigram_baseline": ({
                "vocabulary_size": baseline.vocabulary_size,
                "training_target_count": baseline.training_target_count,
                "raw_counts": list(baseline.raw_counts),
                "smoothed_counts": list(baseline.smoothed_counts),
                "probabilities": list(baseline.probabilities),
                "probabilities_hex": [value.hex() for value in baseline.probabilities],
                "top_token_id": baseline.top_token_id,
            } if baseline is not None else None),
            "laplace_unigram_validation": _metric_mapping(unigram) if unigram is not None else None,
            "samples": samples, "sealed_test_access": "none",
            "failure": _failure(stage, error),
            "limitations": list(LIMITATIONS),
        }
        _validate_development_evidence_mapping(
            failure_value,
            evaluation_id=evaluation_id,
            marker_sha256=marker.sha256,
            authority=authority,
            configuration=configuration,
            completed=False,
        )
        publication = _publish_terminal(root, evaluation_id, filename="phase9-development-failure.json", opposite_filename="phase9-development-evidence.json", kind="phase9_development_failure", status="failed", content=_canonical_json(failure_value))
        if publication.state == KNOWN_PRESENT and publication.reference is not None:
            return publication.reference
        raise


def run_fixed_phase9_sealed_test_evaluation(
    repository_root: Path,
    *,
    evaluation_id: str,
) -> Phase9EvidenceReference:
    root = _validate_repository(repository_root)
    evaluation_id = _validate_id(evaluation_id)
    try:
        record_bytes = (root / "EXPERIMENT_LOG.md").read_bytes()
    except OSError:
        raise Phase9GovernanceError("phase9.governance.repository", field="record") from None
    records = _parse_selected_chain(record_bytes, evaluation_id)
    _validate_predecessor_chain(root, records, evaluation_id)
    development_result = _one(records, evaluation_id, "development_result")
    plan = _one(records, evaluation_id, "sealed_test_plan")
    authorization = _one(records, evaluation_id, "sealed_test_authorization")
    existing_results = _matches(records, evaluation_id, "sealed_test_result")
    if existing_results:
        if len(existing_results) == 1:
            _validate_sealed_result_record(existing_results[0])
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="evaluation_id")
    development_digest = authorization.values["Development evidence SHA-256"]
    if not _is_digest(development_digest):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    development = _read_evidence_artifact(
        root,
        evaluation_id,
        "phase9-development-evidence.json",
        maximum=1_048_576,
        expected_sha256=development_digest,
    )
    sealed_authorization_commit, development_authority = _validate_sealed_launch(
        root,
        records,
        evaluation_id,
        development_result,
        plan,
        authorization,
        development,
        development_digest,
    )
    sealed_metadata = _sealed_test_metadata(root)
    baseline_value = development.get("laplace_unigram_baseline")
    if type(baseline_value) is not dict:
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record")
    _validate_baseline_mapping(baseline_value)
    try:
        baseline = _make_laplace_unigram(
            vocabulary_size=baseline_value["vocabulary_size"],
            training_target_count=baseline_value["training_target_count"],
            raw_counts=tuple(baseline_value["raw_counts"]),
            smoothed_counts=tuple(baseline_value["smoothed_counts"]),
            probabilities=tuple(baseline_value["probabilities"]),
            top_token_id=baseline_value["top_token_id"],
        )
    except (KeyError, TypeError):
        raise Phase9GovernanceError("phase9.governance.sealed_test", field="record") from None
    bundle = load_phase9_inference_bundle(root)
    marker = {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "kind": "phase9_sealed_test_access", "sealed_plan_record_sha256": plan.sha256,
        "sealed_pre_registration_commit": authorization.values["Sealed pre-registration commit"],
        "sealed_authorization_record_sha256": authorization.sha256,
        "sealed_authorization_commit": sealed_authorization_commit,
        "phase9_contract_sha256": PHASE9_SPEC_SHA256,
        "phase9_implementation_commit": authorization.values["Phase 9 implementation commit"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "development_evidence_sha256": development_digest,
        "created_at_utc": _now(), "consumed": True,
    }
    _validate_access_marker(
        marker,
        evaluation_id=evaluation_id,
        plan=plan,
        authorization=authorization,
        authorization_commit=sealed_authorization_commit,
        development_evidence_sha256=development_digest,
    )
    publication = _serialized_sealed_access_publication(
        root,
        evaluation_id,
        plan,
        authorization,
        authorization_commit=sealed_authorization_commit,
        marker=marker,
    )
    if publication.state != KNOWN_PRESENT or publication.reference is None:
        raise Phase9PublicationError("phase9.evidence.publication", operation=publication.operation, publication_state=publication.state)
    sealed_authority = dict(development_authority)
    sealed_authority.update({
        "development_result_record_sha256": plan.values["Development result record SHA-256"],
        "development_result_commit": plan.values["Accepted development result commit"],
        "development_evidence_sha256": development_digest,
        "sealed_plan_record_sha256": plan.sha256,
        "sealed_pre_registration_commit": authorization.values["Sealed pre-registration commit"],
        "sealed_authorization_record_sha256": authorization.sha256,
        "sealed_authorization_commit": sealed_authorization_commit,
    })
    stage = "sealed_corpus_open"
    model_metrics: Phase9Metrics | None = None
    uniform_metrics: Phase9Metrics | None = None
    unigram_metrics: Phase9Metrics | None = None
    try:
        try:
            text = _read_sealed_test_once(root, sealed_metadata)
        except Phase9GovernanceError as error:
            if error.details.get("field") != "path":
                stage = "sealed_corpus_validation"
            raise
        stage = "sealed_tokenization"
        tokens = bundle.tokenizer.encode(text)
        document = Phase9TokenDocument("twelfth-night", 8, "sealed", tokens)
        stage = "sealed_window_build"
        windows = build_phase9_windows((document,), vocabulary_size=81, context_length=256, stride=256)
        test_windows = tuple(Phase9Window(w.document_id, w.document_order, "test", w.start_index, w.input_ids, w.target_ids) for w in windows)
        if (len(test_windows), sum(len(window.target_ids) for window in test_windows)) != (449, 114_811):
            raise Phase9GovernanceError("phase9.governance.dataset", field="counts")
        stage = "sealed_model_evaluation"
        model_metrics = _evaluate_model_internal(bundle.model, test_windows, predictor="mini_gpt_best_validation", split="test", allow_test=True)
        stage = "sealed_uniform_evaluation"
        uniform_metrics = _evaluate_uniform_internal(test_windows, vocabulary_size=81, split="test", allow_test=True)
        stage = "sealed_laplace_unigram_evaluation"
        unigram_metrics = _evaluate_laplace_internal(baseline, test_windows, split="test", allow_test=True)
        stage = "sealed_cross_predictor_comparison"
        _compare_metrics(model_metrics, uniform_metrics, unigram_metrics)
        evidence = {
            "schema_version": 1, "evaluation_id": evaluation_id,
            "kind": "phase9_sealed_test", "status": "completed",
            "recorded_at_utc": _now(), "authority": sealed_authority,
            "access_marker_sha256": publication.reference.sha256,
            "development_evidence_sha256": development_digest,
            "model_test": _metric_mapping(model_metrics),
            "uniform_test": _metric_mapping(uniform_metrics),
            "laplace_unigram_test": _metric_mapping(unigram_metrics),
            "sealed_test_access": "one_shot_completed", "failure": None,
            "limitations": list(LIMITATIONS),
        }
        _validate_sealed_evidence_mapping(
            evidence,
            evaluation_id=evaluation_id,
            marker_sha256=publication.reference.sha256,
            development_evidence_sha256=development_digest,
            authority=sealed_authority,
            completed=True,
        )
        stage = "sealed_evidence_publication"
        completed = _publish_terminal(
            root, evaluation_id, filename="phase9-sealed-test-evidence.json",
            opposite_filename="phase9-sealed-test-failure.json",
            kind="phase9_sealed_test", status="completed",
            content=_canonical_json(evidence),
        )
        if completed.state != KNOWN_PRESENT or completed.reference is None:
            raise Phase9PublicationError("phase9.evidence.publication", operation=completed.operation, publication_state=completed.state)
        return completed.reference
    except _KNOWN_EXECUTION_ERRORS as error:
        finished_at = _now()
        recorded_at = _now()
        failure_value = {
            "schema_version": 1, "evaluation_id": evaluation_id,
            "kind": "phase9_sealed_test_failure", "status": "failed",
            "recorded_at_utc": recorded_at, "started_at_utc": marker["created_at_utc"],
            "finished_at_utc": finished_at, "authority": sealed_authority,
            "access_marker_sha256": publication.reference.sha256,
            "development_evidence_sha256": marker["development_evidence_sha256"],
            "model_test": _metric_mapping(model_metrics) if model_metrics is not None else None,
            "uniform_test": _metric_mapping(uniform_metrics) if uniform_metrics is not None else None,
            "laplace_unigram_test": _metric_mapping(unigram_metrics) if unigram_metrics is not None else None,
            "sealed_test_access": "one_shot_consumed_failed",
            "failure": _failure(stage, error),
            "limitations": list(LIMITATIONS),
        }
        _validate_sealed_evidence_mapping(
            failure_value,
            evaluation_id=evaluation_id,
            marker_sha256=publication.reference.sha256,
            development_evidence_sha256=development_digest,
            authority=sealed_authority,
            completed=False,
        )
        failed = _publish_terminal(root, evaluation_id, filename="phase9-sealed-test-failure.json", opposite_filename="phase9-sealed-test-evidence.json", kind="phase9_sealed_test_failure", status="failed", content=_canonical_json(failure_value))
        if failed.state == KNOWN_PRESENT and failed.reference is not None:
            return failed.reference
        raise


__all__ = [
    "run_fixed_phase9_development_evaluation",
    "run_fixed_phase9_sealed_test_evaluation",
]
