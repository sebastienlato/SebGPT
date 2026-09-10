"""Transparent Phase 5 single-sequence causal self-attention."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Mapping

import torch
from torch import nn


MODEL_WIDTH: Final = 32
MAX_SEQUENCE_LENGTH: Final = 256
SINGLE_HEAD_WIDTH: Final = 32
NUMBER_OF_HEADS: Final = 4
MULTI_HEAD_WIDTH: Final = 8
ATTENTION_INITIALIZATION_SEED: Final = 5005
PARAMETER_DEVICE: Final = torch.device("cpu")
PARAMETER_DTYPE: Final = torch.float32

_SINGLE_PARAMETER_NAMES: Final = (
    "query_weight",
    "key_weight",
    "value_weight",
)
_MULTI_PARAMETER_NAMES: Final = (
    "output_weight",
    "heads.0.query_weight",
    "heads.0.key_weight",
    "heads.0.value_weight",
    "heads.1.query_weight",
    "heads.1.key_weight",
    "heads.1.value_weight",
    "heads.2.query_weight",
    "heads.2.key_weight",
    "heads.2.value_weight",
    "heads.3.query_weight",
    "heads.3.key_weight",
    "heads.3.value_weight",
)


class AttentionTypeError(TypeError):
    """A deterministic type failure containing only structural facts."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class AttentionContractError(ValueError):
    """A deterministic contract failure containing only structural facts."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class AttentionNumericalError(ArithmeticError):
    """A deterministic nonfinite-arithmetic failure with safe diagnostics."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


@dataclass(frozen=True)
class AttentionInspection:
    """The exact output and post-softmax weights from one attention call."""

    output: torch.Tensor = field(repr=False)
    attention_weights: torch.Tensor = field(repr=False)


def _all_true(value: torch.Tensor) -> bool:
    return bool(torch.all(value).item())


def _require_contract(
    condition: bool,
    invariant: str,
    **safe_facts: object,
) -> None:
    if not condition:
        raise AttentionContractError(invariant, **safe_facts)


def _require_finite(
    value: torch.Tensor,
    invariant: str,
    *,
    permitted: torch.Tensor | None = None,
) -> None:
    selected = value if permitted is None else value[permitted]
    if not _all_true(torch.isfinite(selected)):
        raise AttentionNumericalError(invariant, shape=tuple(value.shape))


def _require_positive(value: torch.Tensor, invariant: str) -> None:
    if not _all_true(value > 0):
        raise AttentionNumericalError(invariant, shape=tuple(value.shape))


def _validate_constructor_types(
    prefix: str,
    arguments: tuple[tuple[str, object, type], ...],
) -> None:
    for name, value, expected_type in arguments:
        if expected_type is int:
            valid = type(value) is int
        elif expected_type is torch.device:
            valid = type(value) is torch.device
        else:
            valid = isinstance(value, torch.dtype)
        if not valid:
            raise AttentionTypeError(f"{prefix}.constructor.{name}.type")


def _validate_constructor_values(
    prefix: str,
    arguments: tuple[tuple[str, object, object], ...],
) -> None:
    for name, observed, expected in arguments:
        _require_contract(
            observed == expected,
            f"{prefix}.constructor.{name}.value",
            expected=str(expected),
        )


def _initialize_weight(
    shape: tuple[int, int],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    value = torch.empty(shape, device=PARAMETER_DEVICE, dtype=PARAMETER_DTYPE)
    nn.init.normal_(
        value,
        mean=0.0,
        std=1.0 / math.sqrt(MODEL_WIDTH),
        generator=generator,
    )
    return value


def _validate_representations(representations: torch.Tensor) -> None:
    if not isinstance(representations, torch.Tensor):
        raise AttentionTypeError("attention.input.tensor")
    _require_contract(
        representations.dim() == 2,
        "attention.input.rank",
        expected=2,
        observed=representations.dim(),
    )
    _require_contract(
        representations.dtype is PARAMETER_DTYPE,
        "attention.input.dtype",
        expected=str(PARAMETER_DTYPE),
        observed=str(representations.dtype),
    )
    _require_contract(
        representations.device == PARAMETER_DEVICE,
        "attention.input.device",
        expected=str(PARAMETER_DEVICE),
        observed=str(representations.device),
    )
    _require_contract(
        representations.shape[1] == MODEL_WIDTH,
        "attention.input.model_width",
        expected=MODEL_WIDTH,
        observed=representations.shape[1],
    )
    sequence_length = representations.shape[0]
    _require_contract(
        1 <= sequence_length <= MAX_SEQUENCE_LENGTH,
        "attention.input.sequence_length",
        lower_bound=1,
        upper_bound_inclusive=MAX_SEQUENCE_LENGTH,
        observed=sequence_length,
    )
    _require_finite(representations, "attention.input.finite")


def _validate_semantic_parameters(
    items: tuple[tuple[str, object, tuple[int, int], str], ...],
) -> None:
    for _, parameter, _, invariant_base in items:
        if not isinstance(parameter, nn.Parameter):
            raise AttentionTypeError(f"{invariant_base}.type")
    for _, parameter, expected_shape, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_contract(
            tuple(parameter.shape) == expected_shape,
            f"{invariant_base}.shape",
            expected=expected_shape,
            observed=tuple(parameter.shape),
        )
    for _, parameter, _, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_contract(
            parameter.device == PARAMETER_DEVICE,
            f"{invariant_base}.device",
            expected=str(PARAMETER_DEVICE),
            observed=str(parameter.device),
        )
    for _, parameter, _, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_contract(
            parameter.dtype is PARAMETER_DTYPE,
            f"{invariant_base}.dtype",
            expected=str(PARAMETER_DTYPE),
            observed=str(parameter.dtype),
        )
    for _, parameter, _, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_finite(parameter, f"{invariant_base}.finite")


def _causal_visibility(
    sequence_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(sequence_length, dtype=torch.long, device=device)
    return positions[None, :] <= positions[:, None]


def _stable_masked_softmax(
    scaled_scores: torch.Tensor,
    allowed: torch.Tensor,
    *,
    prefix: str,
) -> torch.Tensor:
    masked_scores = scaled_scores.masked_fill(~allowed, -torch.inf)
    _require_finite(
        masked_scores,
        f"{prefix}.masked_scores.permitted_finite",
        permitted=allowed,
    )
    row_max = masked_scores.max(dim=-1, keepdim=True).values
    _require_finite(row_max, f"{prefix}.row_max.finite")
    shifted_scores = masked_scores - row_max
    _require_finite(
        shifted_scores,
        f"{prefix}.shifted_scores.permitted_finite",
        permitted=allowed,
    )
    exponentials = torch.exp(shifted_scores)
    _require_finite(exponentials, f"{prefix}.exponentials.finite")
    denominator = exponentials.sum(dim=-1, keepdim=True)
    _require_finite(denominator, f"{prefix}.denominator.finite")
    _require_positive(denominator, f"{prefix}.denominator.positive")
    attention_weights = exponentials / denominator
    _require_finite(attention_weights, f"{prefix}.weights.finite")
    return attention_weights


def _compute_head(
    representations: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    allowed: torch.Tensor,
    *,
    head_width: int,
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = representations @ query_weight
    _require_finite(query, f"{prefix}.projection.query.finite")
    key = representations @ key_weight
    _require_finite(key, f"{prefix}.projection.key.finite")
    value = representations @ value_weight
    _require_finite(value, f"{prefix}.projection.value.finite")
    scores = query @ key.transpose(0, 1)
    _require_finite(scores, f"{prefix}.scores.finite")
    scaled_scores = scores / math.sqrt(head_width)
    _require_finite(scaled_scores, f"{prefix}.scaled_scores.finite")
    attention_weights = _stable_masked_softmax(
        scaled_scores,
        allowed,
        prefix=prefix,
    )
    output = attention_weights @ value
    _require_finite(output, f"{prefix}.output.finite")
    return output, attention_weights


class _AttentionHead(nn.Module):
    """Visible holder for one multi-head Q/K/V parameter set."""

    def __init__(
        self,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        value_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.query_weight = nn.Parameter(query_weight)
        self.key_weight = nn.Parameter(key_weight)
        self.value_weight = nn.Parameter(value_weight)


class SingleHeadCausalSelfAttention(nn.Module):
    """One explicit full-width causal self-attention head."""

    def __init__(
        self,
        *,
        model_width: int,
        max_sequence_length: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        type_arguments = (
            ("model_width", model_width, int),
            ("max_sequence_length", max_sequence_length, int),
            ("seed", seed, int),
            ("device", device, torch.device),
            ("dtype", dtype, torch.dtype),
        )
        _validate_constructor_types("single", type_arguments)
        _validate_constructor_values(
            "single",
            (
                ("model_width", model_width, MODEL_WIDTH),
                ("max_sequence_length", max_sequence_length, MAX_SEQUENCE_LENGTH),
                ("seed", seed, ATTENTION_INITIALIZATION_SEED),
                ("device", device, PARAMETER_DEVICE),
                ("dtype", dtype, PARAMETER_DTYPE),
            ),
        )
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.query_weight = nn.Parameter(
            _initialize_weight((MODEL_WIDTH, SINGLE_HEAD_WIDTH), generator=generator)
        )
        self.key_weight = nn.Parameter(
            _initialize_weight((MODEL_WIDTH, SINGLE_HEAD_WIDTH), generator=generator)
        )
        self.value_weight = nn.Parameter(
            _initialize_weight((MODEL_WIDTH, SINGLE_HEAD_WIDTH), generator=generator)
        )

    def _semantic_parameter_items(
        self,
    ) -> tuple[tuple[str, object, tuple[int, int], str], ...]:
        return (
            (
                "query_weight",
                getattr(self, "query_weight", None),
                (32, 32),
                "single.parameter.query_weight",
            ),
            (
                "key_weight",
                getattr(self, "key_weight", None),
                (32, 32),
                "single.parameter.key_weight",
            ),
            (
                "value_weight",
                getattr(self, "value_weight", None),
                (32, 32),
                "single.parameter.value_weight",
            ),
        )

    def _validate_parameters(self) -> tuple[tuple[str, object, tuple[int, int], str], ...]:
        items = self._semantic_parameter_items()
        names = tuple(name for name, _ in self.named_parameters())
        modules = tuple(name for name, _ in self.named_modules())
        _require_contract(
            names == _SINGLE_PARAMETER_NAMES
            and modules == ("",)
            and tuple(self.named_buffers()) == (),
            "single.parameters.structure",
        )
        _validate_semantic_parameters(items)
        return items

    def _compute_attention(
        self,
        representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_representations(representations)
        self._validate_parameters()
        allowed = _causal_visibility(
            representations.shape[0],
            device=representations.device,
        )
        return _compute_head(
            representations,
            self.query_weight,
            self.key_weight,
            self.value_weight,
            allowed,
            head_width=SINGLE_HEAD_WIDTH,
            prefix="single",
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        output, _ = self._compute_attention(representations)
        return output

    def inspect(self, representations: torch.Tensor) -> AttentionInspection:
        output, attention_weights = self._compute_attention(representations)
        return AttentionInspection(
            output=output,
            attention_weights=attention_weights,
        )


class MultiHeadCausalSelfAttention(nn.Module):
    """Four visible width-eight causal heads plus an explicit output weight."""

    def __init__(
        self,
        *,
        model_width: int,
        number_of_heads: int,
        head_width: int,
        max_sequence_length: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        type_arguments = (
            ("model_width", model_width, int),
            ("number_of_heads", number_of_heads, int),
            ("head_width", head_width, int),
            ("max_sequence_length", max_sequence_length, int),
            ("seed", seed, int),
            ("device", device, torch.device),
            ("dtype", dtype, torch.dtype),
        )
        _validate_constructor_types("multi", type_arguments)
        _validate_constructor_values(
            "multi",
            (
                ("model_width", model_width, MODEL_WIDTH),
                ("number_of_heads", number_of_heads, NUMBER_OF_HEADS),
                ("head_width", head_width, MULTI_HEAD_WIDTH),
                ("max_sequence_length", max_sequence_length, MAX_SEQUENCE_LENGTH),
                ("seed", seed, ATTENTION_INITIALIZATION_SEED),
                ("device", device, PARAMETER_DEVICE),
                ("dtype", dtype, PARAMETER_DTYPE),
            ),
        )
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        head_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(NUMBER_OF_HEADS):
            head_tensors.append(
                (
                    _initialize_weight(
                        (MODEL_WIDTH, MULTI_HEAD_WIDTH), generator=generator
                    ),
                    _initialize_weight(
                        (MODEL_WIDTH, MULTI_HEAD_WIDTH), generator=generator
                    ),
                    _initialize_weight(
                        (MODEL_WIDTH, MULTI_HEAD_WIDTH), generator=generator
                    ),
                )
            )
        output_tensor = _initialize_weight(
            (MODEL_WIDTH, MODEL_WIDTH),
            generator=generator,
        )
        self.heads = nn.ModuleList(
            _AttentionHead(query, key, value)
            for query, key, value in head_tensors
        )
        self.output_weight = nn.Parameter(output_tensor)

    def _semantic_parameter_items(
        self,
    ) -> tuple[tuple[str, object, tuple[int, int], str], ...]:
        items: list[tuple[str, object, tuple[int, int], str]] = []
        heads = getattr(self, "heads", ())
        if isinstance(heads, nn.ModuleList):
            for index, head in enumerate(heads):
                for name in ("query_weight", "key_weight", "value_weight"):
                    items.append(
                        (
                            f"heads.{index}.{name}",
                            getattr(head, name, None),
                            (32, 8),
                            f"multi.parameter.heads.{index}.{name}",
                        )
                    )
        items.append(
            (
                "output_weight",
                getattr(self, "output_weight", None),
                (32, 32),
                "multi.parameter.output_weight",
            )
        )
        return tuple(items)

    def _validate_parameters(self) -> tuple[tuple[str, object, tuple[int, int], str], ...]:
        heads = getattr(self, "heads", None)
        structure = (
            type(heads) is nn.ModuleList
            and len(heads) == NUMBER_OF_HEADS
            and all(type(head) is _AttentionHead for head in heads)
            and tuple(name for name, _ in self.named_parameters())
            == _MULTI_PARAMETER_NAMES
            and tuple(name for name, _ in self.named_modules())
            == ("", "heads", "heads.0", "heads.1", "heads.2", "heads.3")
            and tuple(self.named_buffers()) == ()
        )
        _require_contract(structure, "multi.parameters.structure")
        items = self._semantic_parameter_items()
        _validate_semantic_parameters(items)
        return items

    def _compute_attention(
        self,
        representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_representations(representations)
        self._validate_parameters()
        allowed = _causal_visibility(
            representations.shape[0],
            device=representations.device,
        )
        head_outputs: list[torch.Tensor] = []
        head_weights: list[torch.Tensor] = []
        for index, head in enumerate(self.heads):
            output, weights = _compute_head(
                representations,
                head.query_weight,
                head.key_weight,
                head.value_weight,
                allowed,
                head_width=MULTI_HEAD_WIDTH,
                prefix=f"multi.head.{index}",
            )
            head_outputs.append(output)
            head_weights.append(weights)
        concatenated = torch.cat(tuple(head_outputs), dim=-1)
        _require_finite(concatenated, "multi.concatenated.finite")
        output = concatenated @ self.output_weight
        _require_finite(output, "multi.output_projection.finite")
        inspection_weights = torch.stack(tuple(head_weights), dim=0)
        return output, inspection_weights

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        output, _ = self._compute_attention(representations)
        return output

    def inspect(self, representations: torch.Tensor) -> AttentionInspection:
        output, attention_weights = self._compute_attention(representations)
        return AttentionInspection(
            output=output,
            attention_weights=attention_weights,
        )


__all__ = [
    "AttentionContractError",
    "AttentionInspection",
    "AttentionNumericalError",
    "AttentionTypeError",
    "MultiHeadCausalSelfAttention",
    "SingleHeadCausalSelfAttention",
]
