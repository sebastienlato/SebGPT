"""Exact Phase 9 records and content-safe exceptions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch

from sebgpt.model.embeddings import TokenPositionEmbedding
from sebgpt.model.mini_gpt import LanguageModelHead, MiniGPT
from sebgpt.model.self_attention import (
    MultiHeadCausalSelfAttention,
    _AttentionHead,
)
from sebgpt.model.transformer_block import (
    ExplicitDropout,
    ExplicitLayerNorm,
    PositionwiseFeedForward,
    TransformerBlock,
)
from sebgpt.tokenization.code_point import CodePointTokenizer
from sebgpt.tokenization.vocabulary_artifact import VocabularyBinding


class _Phase9ErrorMixin:
    def _initialize(self, invariant: str, safe_facts: Mapping[str, object]) -> str:
        if type(invariant) is not str or not invariant.startswith("phase9."):
            raise TypeError("invalid Phase 9 invariant")
        allowed_keys = {
            "field",
            "position",
            "operation",
            "publication_state",
            "sample_id",
            "split",
            "completed_token_count",
            "attempt_marker_present",
            "final_evidence_present",
        }
        if any(key not in allowed_keys for key in safe_facts):
            raise TypeError("unsafe Phase 9 error fact")
        if any(type(value) not in (str, int, bool, type(None)) for value in safe_facts.values()):
            raise TypeError("unsafe Phase 9 error value")
        safe_fields = {
            "repository_root", "evaluation_id", "branch", "synchronization", "clean",
            "ancestor", "protected_path", "record", "configuration", "runtime",
            "catalog", "reference", "checkpoint", "payload", "authority", "vocabulary",
            "model", "rng", "manifest", "dataset", "tokenizer", "bundle", "windows",
            "work_count", "work_identity", "path", "bytes",
            "utf8", "normalization", "documents", "vocabulary_size", "context_length",
            "stride", "token", "counts", "probabilities", "split", "window_sha256",
            "window_count", "target_count", "prompt", "generated_token_count", "mode",
            "temperature", "top_k", "seed", "logits", "output", "mutation", "evidence",
            "rank", "shape", "dtype", "device", "finite", "baseline", "predictor",
            "metric", "perplexity", "work", "run", "optimizer", "progress", "lineage",
        }
        if "field" in safe_facts and safe_facts["field"] not in safe_fields:
            raise TypeError("unsafe Phase 9 field")
        if "position" in safe_facts and (
            safe_facts["position"] is not None
            and (type(safe_facts["position"]) is not int or safe_facts["position"] < 0)
        ):
            raise TypeError("unsafe Phase 9 position")
        if "operation" in safe_facts and safe_facts["operation"] not in {
            "open_parent", "create_temporary", "write", "fsync_file",
            "publish_no_clobber", "fsync_parent", "revalidate_identity",
        }:
            raise TypeError("unsafe Phase 9 operation")
        if "publication_state" in safe_facts and safe_facts["publication_state"] not in {
            "not_published", "known_absent", "known_present", "unknown",
        }:
            raise TypeError("unsafe Phase 9 publication state")
        if "sample_id" in safe_facts and safe_facts["sample_id"] not in {
            "romeo-greedy", "romeo-focused", "romeo-full", "to-be-greedy",
            "to-be-focused", "to-be-full",
        }:
            raise TypeError("unsafe Phase 9 sample ID")
        if "split" in safe_facts and safe_facts["split"] not in {"train", "validation", "test"}:
            raise TypeError("unsafe Phase 9 split")
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        return json.dumps(details, ensure_ascii=True, sort_keys=True)

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class Phase9TypeError(_Phase9ErrorMixin, TypeError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize(invariant, safe_facts))


class Phase9ContractError(_Phase9ErrorMixin, ValueError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize(invariant, safe_facts))


class Phase9GovernanceError(_Phase9ErrorMixin, ValueError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize(invariant, safe_facts))


class Phase9CheckpointError(_Phase9ErrorMixin, ValueError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize(invariant, safe_facts))


class Phase9NumericalError(_Phase9ErrorMixin, ArithmeticError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize(invariant, safe_facts))


class Phase9PublicationError(_Phase9ErrorMixin, RuntimeError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize(invariant, safe_facts))


def _model_parameter_oracle() -> tuple[tuple[str, tuple[int, ...]], ...]:
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
                result.append(
                    (f"{prefix}.attention.heads.{head}.{projection}_weight", (32, 8))
                )
        result.extend((
            (f"{prefix}.norm2.gamma", (32,)),
            (f"{prefix}.norm2.beta", (32,)),
            (f"{prefix}.feed_forward.first_weight", (32, 128)),
            (f"{prefix}.feed_forward.first_bias", (128,)),
            (f"{prefix}.feed_forward.second_weight", (128, 32)),
            (f"{prefix}.feed_forward.second_bias", (32,)),
        ))
    result.extend((
        ("final_norm.gamma", (32,)),
        ("final_norm.beta", (32,)),
        ("head.output_weight", (32, 81)),
        ("head.output_bias", (81,)),
    ))
    return tuple(result)


_MODEL_PARAMETER_ORACLE = _model_parameter_oracle()
_MODEL_MODULE_NAMES = (
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
_MODEL_CONFIGURATION = {
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
    "device": torch.device("cpu"),
    "dtype": torch.float32,
}


def _model_class_oracle() -> tuple[tuple[str, type[torch.nn.Module]], ...]:
    result: list[tuple[str, type[torch.nn.Module]]] = [
        ("", MiniGPT),
        ("representation", TokenPositionEmbedding),
        ("blocks", torch.nn.ModuleList),
    ]
    for block in range(4):
        prefix = f"blocks.{block}"
        result.extend((
            (prefix, TransformerBlock),
            (f"{prefix}.norm1", ExplicitLayerNorm),
            (f"{prefix}.attention", MultiHeadCausalSelfAttention),
            (f"{prefix}.attention.heads", torch.nn.ModuleList),
        ))
        result.extend(
            (f"{prefix}.attention.heads.{head}", _AttentionHead)
            for head in range(4)
        )
        result.extend((
            (f"{prefix}.attention_dropout", ExplicitDropout),
            (f"{prefix}.norm2", ExplicitLayerNorm),
            (f"{prefix}.feed_forward", PositionwiseFeedForward),
            (f"{prefix}.feed_forward_dropout", ExplicitDropout),
        ))
    result.extend((
        ("final_norm", ExplicitLayerNorm),
        ("head", LanguageModelHead),
    ))
    return tuple(result)


_MODEL_CLASS_ORACLE = _model_class_oracle()


def _validate_phase9_minigpt_structure(model: object) -> MiniGPT:
    if type(model) is not MiniGPT:
        raise Phase9TypeError("phase9.type.argument", field="model")
    try:
        parameters = tuple(model.named_parameters())
        modules = tuple(model.named_modules())
        buffers = tuple(model.named_buffers())
        configuration = dict(model._configuration)
    except (AttributeError, TypeError, RuntimeError):
        raise Phase9ContractError("phase9.contract.bundle", field="model") from None
    if (
        configuration != _MODEL_CONFIGURATION
        or tuple(name for name, _ in parameters)
        != tuple(name for name, _ in _MODEL_PARAMETER_ORACLE)
        or tuple(name for name, _ in modules) != _MODEL_MODULE_NAMES
        or tuple(name for name, _ in modules)
        != tuple(name for name, _ in _MODEL_CLASS_ORACLE)
        or any(
            type(module) is not expected_type
            for (_, module), (_, expected_type) in zip(
                modules, _MODEL_CLASS_ORACLE, strict=True
            )
        )
        or buffers
        or len(parameters) != 90
        or sum(parameter.numel() for _, parameter in parameters) != 63_825
        or len({id(parameter) for _, parameter in parameters}) != 90
        or len({parameter.data_ptr() for _, parameter in parameters}) != 90
        or model.training
        or any(module.training for _, module in modules)
    ):
        raise Phase9ContractError("phase9.contract.bundle", field="model")
    for (expected_name, expected_shape), (name, parameter) in zip(
        _MODEL_PARAMETER_ORACLE, parameters, strict=True
    ):
        if (
            name != expected_name
            or type(parameter) is not torch.nn.Parameter
            or tuple(parameter.shape) != expected_shape
            or parameter.device != torch.device("cpu")
            or parameter.dtype is not torch.float32
            or parameter.requires_grad is not True
            or parameter.grad is not None
            or not bool(torch.all(torch.isfinite(parameter)).item())
        ):
            raise Phase9ContractError("phase9.contract.bundle", field="model")
    dropout_generators = tuple(
        generator
        for block in model.blocks
        for generator in (
            block.attention_dropout._generator,
            block.feed_forward_dropout._generator,
        )
    )
    expected_seeds = (6007, 6008) * 4
    if len(dropout_generators) != 8:
        raise Phase9ContractError("phase9.contract.bundle", field="model")
    for generator, seed in zip(dropout_generators, expected_seeds, strict=True):
        if type(generator) is not torch.Generator or generator.device != torch.device("cpu"):
            raise Phase9ContractError("phase9.contract.bundle", field="model")
        expected_state = torch.Generator(device="cpu").manual_seed(seed).get_state()
        if generator.initial_seed() != seed or not torch.equal(generator.get_state(), expected_state):
            raise Phase9ContractError("phase9.contract.bundle", field="model")
    return model


@dataclass(frozen=True)
class Phase9CheckpointIdentity:
    run_id: str
    role: str
    logical_id: str
    epoch: int
    validation_loss_hex: str
    checkpoint_sha256: str
    checkpoint_relative_path: str
    catalog_sha256: str


@dataclass(frozen=True, init=False)
class Phase9InferenceBundle:
    model: MiniGPT = field(repr=False)
    tokenizer: CodePointTokenizer = field(repr=False)
    vocabulary: VocabularyBinding = field(repr=False)
    checkpoint: Phase9CheckpointIdentity

    def __new__(cls, *args: object, **kwargs: object) -> Phase9InferenceBundle:
        raise Phase9ContractError("phase9.contract.bundle")


@dataclass(frozen=True)
class Phase9Generation:
    prompt: str = field(repr=False)
    generated_text: str = field(repr=False)
    full_text: str = field(repr=False)
    prompt_token_ids: tuple[int, ...] = field(repr=False)
    generated_token_ids: tuple[int, ...] = field(repr=False)
    mode: str
    generated_token_count: int
    temperature: float | None
    top_k: int | None
    seed: int | None
    context_capacity: int
    local_generator_final_state_sha256: str | None


@dataclass(frozen=True)
class Phase9Window:
    document_id: str
    document_order: int
    split: str
    start_index: int
    input_ids: tuple[int, ...] = field(repr=False)
    target_ids: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class Phase9TokenDocument:
    document_id: str
    document_order: int
    split: str
    token_ids: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class Phase9Metrics:
    predictor: str
    split: str
    window_sha256: str
    nll: float
    perplexity: float
    correct_count: int
    target_count: int
    top1_accuracy: float
    window_count: int


@dataclass(frozen=True, init=False)
class Phase9LaplaceUnigramBaseline:
    vocabulary_size: int
    training_target_count: int
    raw_counts: tuple[int, ...] = field(repr=False)
    smoothed_counts: tuple[int, ...] = field(repr=False)
    probabilities: tuple[float, ...] = field(repr=False)
    top_token_id: int

    def __new__(cls, *args: object, **kwargs: object) -> Phase9LaplaceUnigramBaseline:
        raise Phase9ContractError("phase9.contract.generation", field="baseline")


@dataclass(frozen=True)
class Phase9EvidenceReference:
    evaluation_id: str
    kind: str
    status: str
    relative_path: str
    byte_count: int
    sha256: str


_FACTORY_MARKER = object()


def _make_inference_bundle(
    *,
    model: MiniGPT,
    tokenizer: CodePointTokenizer,
    vocabulary: VocabularyBinding,
    checkpoint: Phase9CheckpointIdentity,
) -> Phase9InferenceBundle:
    result = object.__new__(Phase9InferenceBundle)
    object.__setattr__(result, "model", model)
    object.__setattr__(result, "tokenizer", tokenizer)
    object.__setattr__(result, "vocabulary", vocabulary)
    object.__setattr__(result, "checkpoint", checkpoint)
    object.__setattr__(result, "_phase9_factory_marker", _FACTORY_MARKER)
    return result


def _is_inference_bundle(value: object) -> bool:
    return (
        type(value) is Phase9InferenceBundle
        and getattr(value, "_phase9_factory_marker", None) is _FACTORY_MARKER
    )


def _make_laplace_unigram(
    *,
    vocabulary_size: int,
    training_target_count: int,
    raw_counts: tuple[int, ...],
    smoothed_counts: tuple[int, ...],
    probabilities: tuple[float, ...],
    top_token_id: int,
) -> Phase9LaplaceUnigramBaseline:
    result = object.__new__(Phase9LaplaceUnigramBaseline)
    object.__setattr__(result, "vocabulary_size", vocabulary_size)
    object.__setattr__(result, "training_target_count", training_target_count)
    object.__setattr__(result, "raw_counts", raw_counts)
    object.__setattr__(result, "smoothed_counts", smoothed_counts)
    object.__setattr__(result, "probabilities", probabilities)
    object.__setattr__(result, "top_token_id", top_token_id)
    object.__setattr__(result, "_phase9_factory_marker", _FACTORY_MARKER)
    return result


def _is_laplace_unigram(value: object) -> bool:
    return (
        type(value) is Phase9LaplaceUnigramBaseline
        and getattr(value, "_phase9_factory_marker", None) is _FACTORY_MARKER
    )


__all__ = [
    "Phase9TypeError",
    "Phase9ContractError",
    "Phase9GovernanceError",
    "Phase9CheckpointError",
    "Phase9NumericalError",
    "Phase9PublicationError",
    "Phase9CheckpointIdentity",
    "Phase9InferenceBundle",
    "Phase9Generation",
    "Phase9Window",
    "Phase9TokenDocument",
    "Phase9Metrics",
    "Phase9LaplaceUnigramBaseline",
    "Phase9EvidenceReference",
]
