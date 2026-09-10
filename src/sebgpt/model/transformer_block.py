"""Explicit Phase 6 pre-norm Transformer block components."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Final, Mapping, TypeVar

import torch
from torch import nn

from sebgpt.model.self_attention import MultiHeadCausalSelfAttention


MODEL_WIDTH: Final = 32
NUMBER_OF_HEADS: Final = 4
HEAD_WIDTH: Final = 8
HIDDEN_WIDTH: Final = 128
MAX_SEQUENCE_LENGTH: Final = 256
LAYER_NORM_EPSILON: Final = 1e-5
DROPOUT_PROBABILITY: Final = 0.1
ATTENTION_INITIALIZATION_SEED: Final = 5005
PHASE6_PARAMETER_INITIALIZATION_SEED: Final = 6006
ATTENTION_DROPOUT_SEED: Final = 6007
FEED_FORWARD_DROPOUT_SEED: Final = 6008
PARAMETER_DEVICE: Final = torch.device("cpu")
PARAMETER_DTYPE: Final = torch.float32

class TransformerBlockTypeError(TypeError):
    """A deterministic Phase 6 type failure with structural facts only."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class TransformerBlockContractError(ValueError):
    """A deterministic Phase 6 contract failure with structural facts only."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class TransformerBlockNumericalError(ArithmeticError):
    """A deterministic Phase 6 numerical failure with structural facts only."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class TransformerBlockTransactionError(RuntimeError):
    """A deterministic local-dropout-state transaction failure."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


@dataclass(frozen=True)
class LayerNormInspection:
    output: torch.Tensor = field(repr=False)
    mean: torch.Tensor = field(repr=False)
    variance: torch.Tensor = field(repr=False)
    normalized: torch.Tensor = field(repr=False)


@dataclass(frozen=True)
class DropoutInspection:
    output: torch.Tensor = field(repr=False)
    keep_mask: torch.Tensor = field(repr=False)


@dataclass(frozen=True)
class FeedForwardInspection:
    pre_activation: torch.Tensor = field(repr=False)
    hidden_activation: torch.Tensor = field(repr=False)
    output: torch.Tensor = field(repr=False)


@dataclass(frozen=True)
class TransformerBlockInspection:
    normalized_attention_input: torch.Tensor = field(repr=False)
    attention_branch_output_before_dropout: torch.Tensor = field(repr=False)
    attention_weights: torch.Tensor = field(repr=False)
    attention_dropout_keep_mask: torch.Tensor = field(repr=False)
    attention_branch_output_after_dropout: torch.Tensor = field(repr=False)
    first_residual: torch.Tensor = field(repr=False)
    normalized_feed_forward_input: torch.Tensor = field(repr=False)
    feed_forward_pre_activation: torch.Tensor = field(repr=False)
    feed_forward_hidden_activation: torch.Tensor = field(repr=False)
    feed_forward_branch_output_before_dropout: torch.Tensor = field(repr=False)
    feed_forward_dropout_keep_mask: torch.Tensor = field(repr=False)
    feed_forward_branch_output_after_dropout: torch.Tensor = field(repr=False)
    output: torch.Tensor = field(repr=False)


def _all_true(value: torch.Tensor) -> bool:
    return bool(torch.all(value).item())


def _require_contract(
    condition: bool,
    invariant: str,
    **safe_facts: object,
) -> None:
    if not condition:
        raise TransformerBlockContractError(invariant, **safe_facts)


def _require_finite(value: torch.Tensor, invariant: str) -> None:
    if not _all_true(torch.isfinite(value)):
        raise TransformerBlockNumericalError(invariant, shape=tuple(value.shape))


def _require_positive(value: torch.Tensor, invariant: str) -> None:
    if not _all_true(value > 0):
        raise TransformerBlockNumericalError(invariant, shape=tuple(value.shape))


def _validate_constructor_types(
    prefix: str,
    arguments: tuple[tuple[str, object, type], ...],
) -> None:
    for name, value, expected_type in arguments:
        if expected_type is int:
            valid = type(value) is int
        elif expected_type is float:
            valid = type(value) is float
        elif expected_type is torch.device:
            valid = type(value) is torch.device
        else:
            valid = isinstance(value, torch.dtype)
        if not valid:
            raise TransformerBlockTypeError(
                f"{prefix}.constructor.{name}.type"
            )


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


def _validate_representation_input(value: object, prefix: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TransformerBlockTypeError(f"{prefix}.input.tensor")
    _require_contract(
        value.dim() == 2,
        f"{prefix}.input.rank",
        expected=2,
        observed=value.dim(),
    )
    _require_contract(
        value.dtype is PARAMETER_DTYPE,
        f"{prefix}.input.dtype",
        expected=str(PARAMETER_DTYPE),
        observed=str(value.dtype),
    )
    _require_contract(
        value.device == PARAMETER_DEVICE,
        f"{prefix}.input.device",
        expected=str(PARAMETER_DEVICE),
        observed=str(value.device),
    )
    _require_contract(
        value.shape[1] == MODEL_WIDTH,
        f"{prefix}.input.model_width",
        expected=MODEL_WIDTH,
        observed=value.shape[1],
    )
    length = value.shape[0]
    _require_contract(
        1 <= length <= MAX_SEQUENCE_LENGTH,
        f"{prefix}.input.sequence_length",
        lower_bound=1,
        upper_bound_inclusive=MAX_SEQUENCE_LENGTH,
        observed=length,
    )
    _require_finite(value, f"{prefix}.input.finite")
    return value


def _validate_parameters(
    items: tuple[tuple[object, tuple[int, ...], str], ...],
) -> None:
    for parameter, _, invariant_base in items:
        if not isinstance(parameter, nn.Parameter):
            raise TransformerBlockTypeError(f"{invariant_base}.type")
    for parameter, shape, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_contract(
            tuple(parameter.shape) == shape,
            f"{invariant_base}.shape",
            expected=shape,
            observed=tuple(parameter.shape),
        )
    for parameter, _, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_contract(
            parameter.device == PARAMETER_DEVICE,
            f"{invariant_base}.device",
            expected=str(PARAMETER_DEVICE),
            observed=str(parameter.device),
        )
    for parameter, _, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_contract(
            parameter.dtype is PARAMETER_DTYPE,
            f"{invariant_base}.dtype",
            expected=str(PARAMETER_DTYPE),
            observed=str(parameter.dtype),
        )
    for parameter, _, invariant_base in items:
        assert isinstance(parameter, nn.Parameter)
        _require_finite(parameter, f"{invariant_base}.finite")


class ExplicitLayerNorm(nn.Module):
    """Normalize each position across its width using explicit arithmetic."""

    def __init__(
        self,
        *,
        model_width: int,
        epsilon: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        arguments = (
            ("model_width", model_width, int),
            ("epsilon", epsilon, float),
            ("device", device, torch.device),
            ("dtype", dtype, torch.dtype),
        )
        _validate_constructor_types("normalization", arguments)
        _validate_constructor_values(
            "normalization",
            (
                ("model_width", model_width, MODEL_WIDTH),
                ("epsilon", epsilon, LAYER_NORM_EPSILON),
                ("device", device, PARAMETER_DEVICE),
                ("dtype", dtype, PARAMETER_DTYPE),
            ),
        )
        super().__init__()
        self.gamma = nn.Parameter(
            torch.ones((MODEL_WIDTH,), device=device, dtype=dtype)
        )
        self.beta = nn.Parameter(
            torch.zeros((MODEL_WIDTH,), device=device, dtype=dtype)
        )

    def _parameter_items(self, prefix: str) -> tuple[tuple[object, tuple[int, ...], str], ...]:
        return (
            (getattr(self, "gamma", None), (32,), f"{prefix}.parameter.gamma"),
            (getattr(self, "beta", None), (32,), f"{prefix}.parameter.beta"),
        )

    def _validate_parameters(self, prefix: str = "normalization") -> None:
        _require_contract(
            tuple(name for name, _ in self.named_parameters()) == ("gamma", "beta")
            and tuple(self.named_buffers()) == ()
            and tuple(name for name, _ in self.named_modules()) == ("",),
            f"{prefix}.parameters.structure",
        )
        _validate_parameters(self._parameter_items(prefix))

    def _compute_normalization(
        self,
        representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        value = _validate_representation_input(representations, "normalization")
        self._validate_parameters()
        mean = value.mean(dim=-1, keepdim=True)
        _require_finite(mean, "normalization.mean.finite")
        centered = value - mean
        _require_finite(centered, "normalization.centered.finite")
        squared = centered * centered
        _require_finite(squared, "normalization.squared.finite")
        variance = squared.mean(dim=-1, keepdim=True)
        _require_finite(variance, "normalization.variance.finite")
        variance_plus_epsilon = variance + LAYER_NORM_EPSILON
        _require_finite(
            variance_plus_epsilon,
            "normalization.variance_plus_epsilon.finite",
        )
        _require_positive(
            variance_plus_epsilon,
            "normalization.variance_plus_epsilon.positive",
        )
        inverse_std = torch.rsqrt(variance_plus_epsilon)
        _require_finite(inverse_std, "normalization.inverse_std.finite")
        normalized = centered * inverse_std
        _require_finite(normalized, "normalization.normalized.finite")
        gamma_product = normalized * self.gamma
        _require_finite(gamma_product, "normalization.gamma_product.finite")
        output = gamma_product + self.beta
        _require_finite(output, "normalization.output.finite")
        return output, mean, variance, normalized

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        output, _, _, _ = self._compute_normalization(representations)
        return output

    def inspect(self, representations: torch.Tensor) -> LayerNormInspection:
        output, mean, variance, normalized = self._compute_normalization(
            representations
        )
        return LayerNormInspection(
            output=output,
            mean=mean,
            variance=variance,
            normalized=normalized,
        )


def explicit_gelu(values: torch.Tensor) -> torch.Tensor:
    """Apply the exact error-function GELU using visible primitive operations."""

    if not isinstance(values, torch.Tensor):
        raise TransformerBlockTypeError("gelu.input.tensor")
    _require_contract(
        values.dtype is PARAMETER_DTYPE,
        "gelu.input.dtype",
        expected=str(PARAMETER_DTYPE),
        observed=str(values.dtype),
    )
    _require_contract(
        values.device == PARAMETER_DEVICE,
        "gelu.input.device",
        expected=str(PARAMETER_DEVICE),
        observed=str(values.device),
    )
    _require_finite(values, "gelu.input.finite")
    scaled = values / math.sqrt(2.0)
    _require_finite(scaled, "gelu.scaled.finite")
    erf_value = torch.erf(scaled)
    _require_finite(erf_value, "gelu.erf.finite")
    one_plus_erf = 1.0 + erf_value
    _require_finite(one_plus_erf, "gelu.one_plus_erf.finite")
    half_input = 0.5 * values
    _require_finite(half_input, "gelu.half_input.finite")
    output = half_input * one_plus_erf
    _require_finite(output, "gelu.output.finite")
    return output


class PositionwiseFeedForward(nn.Module):
    """Explicit width-32 to width-128 to width-32 positionwise FFN."""

    def __init__(
        self,
        *,
        model_width: int,
        hidden_width: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        arguments = (
            ("model_width", model_width, int),
            ("hidden_width", hidden_width, int),
            ("seed", seed, int),
            ("device", device, torch.device),
            ("dtype", dtype, torch.dtype),
        )
        _validate_constructor_types("ffn", arguments)
        _validate_constructor_values(
            "ffn",
            (
                ("model_width", model_width, MODEL_WIDTH),
                ("hidden_width", hidden_width, HIDDEN_WIDTH),
                ("seed", seed, PHASE6_PARAMETER_INITIALIZATION_SEED),
                ("device", device, PARAMETER_DEVICE),
                ("dtype", dtype, PARAMETER_DTYPE),
            ),
        )
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        first_weight = torch.empty((32, 128), device=device, dtype=dtype)
        nn.init.normal_(
            first_weight,
            mean=0.0,
            std=1.0 / math.sqrt(32),
            generator=generator,
        )
        self.first_weight = nn.Parameter(first_weight)
        self.first_bias = nn.Parameter(
            torch.zeros((128,), device=device, dtype=dtype)
        )

        second_weight = torch.empty((128, 32), device=device, dtype=dtype)
        nn.init.normal_(
            second_weight,
            mean=0.0,
            std=1.0 / math.sqrt(128),
            generator=generator,
        )
        self.second_weight = nn.Parameter(second_weight)
        self.second_bias = nn.Parameter(
            torch.zeros((32,), device=device, dtype=dtype)
        )

    def _parameter_items(self, prefix: str = "ffn") -> tuple[tuple[object, tuple[int, ...], str], ...]:
        return (
            (getattr(self, "first_weight", None), (32, 128), f"{prefix}.parameter.first_weight"),
            (getattr(self, "first_bias", None), (128,), f"{prefix}.parameter.first_bias"),
            (getattr(self, "second_weight", None), (128, 32), f"{prefix}.parameter.second_weight"),
            (getattr(self, "second_bias", None), (32,), f"{prefix}.parameter.second_bias"),
        )

    def _validate_parameters(self, prefix: str = "ffn") -> None:
        _require_contract(
            tuple(name for name, _ in self.named_parameters())
            == ("first_weight", "first_bias", "second_weight", "second_bias")
            and tuple(self.named_buffers()) == ()
            and tuple(name for name, _ in self.named_modules()) == ("",),
            f"{prefix}.parameters.structure",
        )
        _validate_parameters(self._parameter_items(prefix))

    def _compute_feed_forward(
        self,
        representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = _validate_representation_input(representations, "ffn")
        self._validate_parameters()
        pre_activation = value @ self.first_weight + self.first_bias
        _require_finite(pre_activation, "ffn.pre_activation.finite")
        hidden_activation = explicit_gelu(pre_activation)
        _require_finite(hidden_activation, "ffn.hidden_activation.finite")
        output = hidden_activation @ self.second_weight + self.second_bias
        _require_finite(output, "ffn.output.finite")
        return pre_activation, hidden_activation, output

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        _, _, output = self._compute_feed_forward(representations)
        return output

    def inspect(self, representations: torch.Tensor) -> FeedForwardInspection:
        pre_activation, hidden_activation, output = self._compute_feed_forward(
            representations
        )
        return FeedForwardInspection(
            pre_activation=pre_activation,
            hidden_activation=hidden_activation,
            output=output,
        )


def _read_generator_state(generator: torch.Generator) -> torch.Tensor:
    return generator.get_state()


def _snapshot_generator_state(
    generator: torch.Generator,
    invariant: str,
) -> torch.Tensor:
    try:
        return _read_generator_state(generator).clone()
    except BaseException as error:
        raise TransformerBlockTransactionError(invariant) from error


def _write_generator_state(
    generator: torch.Generator,
    state: torch.Tensor,
) -> None:
    generator.set_state(state)


def _restore_generator_state(
    generator: torch.Generator,
    state: torch.Tensor,
) -> None:
    _write_generator_state(generator, state)


_Result = TypeVar("_Result")


class ExplicitDropout(nn.Module):
    """Explicit inverted dropout driven by one module-local CPU generator."""

    def __init__(
        self,
        *,
        probability: float,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        arguments = (
            ("probability", probability, float),
            ("seed", seed, int),
            ("device", device, torch.device),
            ("dtype", dtype, torch.dtype),
        )
        _validate_constructor_types("dropout", arguments)
        if probability != DROPOUT_PROBABILITY:
            raise TransformerBlockContractError(
                "dropout.constructor.probability.value",
                expected=str(DROPOUT_PROBABILITY),
            )
        _require_contract(
            seed in (ATTENTION_DROPOUT_SEED, FEED_FORWARD_DROPOUT_SEED),
            "dropout.constructor.seed.value",
            expected=(ATTENTION_DROPOUT_SEED, FEED_FORWARD_DROPOUT_SEED),
        )
        _validate_constructor_values(
            "dropout",
            (
                ("device", device, PARAMETER_DEVICE),
                ("dtype", dtype, PARAMETER_DTYPE),
            ),
        )
        super().__init__()
        self.probability = probability
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def _validate_structure(self, *, expected_seed: int | None = None) -> None:
        if type(getattr(self, "training", None)) is not bool:
            raise TransformerBlockTypeError("dropout.mode.type")
        generator = getattr(self, "_generator", None)
        if type(generator) is not torch.Generator:
            raise TransformerBlockTypeError("dropout.structure")
        observed_seed = getattr(self, "seed", None)
        expected = observed_seed if expected_seed is None else expected_seed
        structure = (
            type(getattr(self, "probability", None)) is float
            and self.probability == DROPOUT_PROBABILITY
            and type(observed_seed) is int
            and observed_seed == expected
            and type(getattr(self, "device", None)) is torch.device
            and self.device == PARAMETER_DEVICE
            and isinstance(getattr(self, "dtype", None), torch.dtype)
            and self.dtype is PARAMETER_DTYPE
            and generator.device == PARAMETER_DEVICE
            and generator.initial_seed() == observed_seed
            and tuple(self.named_parameters()) == ()
            and tuple(self.named_buffers()) == ()
            and tuple(name for name, _ in self.named_modules()) == ("",)
        )
        _require_contract(structure, "dropout.structure")

    def _compute_dropout(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            uniform = torch.rand(
                values.shape,
                generator=self._generator,
                device=PARAMETER_DEVICE,
                dtype=PARAMETER_DTYPE,
            )
            _require_finite(uniform, "dropout.uniform.finite")
            _require_contract(
                _all_true((uniform >= 0.0) & (uniform < 1.0)),
                "dropout.uniform.range",
            )
            keep_mask = uniform >= DROPOUT_PROBABILITY
            _require_contract(
                keep_mask.dtype is torch.bool
                and keep_mask.device == PARAMETER_DEVICE
                and tuple(keep_mask.shape) == tuple(values.shape),
                "dropout.keep_mask.structure",
            )
            output = values * keep_mask.to(PARAMETER_DTYPE) / (
                1.0 - DROPOUT_PROBABILITY
            )
        else:
            keep_mask = torch.ones(
                values.shape,
                device=PARAMETER_DEVICE,
                dtype=torch.bool,
            )
            output = values
        _require_finite(output, "dropout.output.finite")
        return output, keep_mask

    def _transaction(
        self,
        values: torch.Tensor,
        package: Callable[[torch.Tensor, torch.Tensor], _Result],
    ) -> _Result:
        value = _validate_representation_input(values, "dropout")
        self._validate_structure()
        snapshot = _snapshot_generator_state(
            self._generator,
            "dropout.transaction.snapshot",
        )
        try:
            output, keep_mask = self._compute_dropout(value)
            return package(output, keep_mask)
        except BaseException as original:
            try:
                _restore_generator_state(self._generator, snapshot)
            except BaseException:
                raise TransformerBlockTransactionError(
                    "dropout.transaction.rollback",
                    failed_streams=("dropout",),
                ) from original
            raise

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self._transaction(values, lambda output, _: output)

    def inspect(self, values: torch.Tensor) -> DropoutInspection:
        return self._transaction(
            values,
            lambda output, keep_mask: DropoutInspection(
                output=output,
                keep_mask=keep_mask,
            ),
        )


class TransformerBlock(nn.Module):
    """One explicit width-32 pre-norm causal Transformer block."""

    def __init__(
        self,
        *,
        model_width: int,
        number_of_heads: int,
        head_width: int,
        hidden_width: int,
        max_sequence_length: int,
        layer_norm_epsilon: float,
        dropout_probability: float,
        attention_seed: int,
        parameter_seed: int,
        attention_dropout_seed: int,
        feed_forward_dropout_seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        arguments = (
            ("model_width", model_width, int),
            ("number_of_heads", number_of_heads, int),
            ("head_width", head_width, int),
            ("hidden_width", hidden_width, int),
            ("max_sequence_length", max_sequence_length, int),
            ("layer_norm_epsilon", layer_norm_epsilon, float),
            ("dropout_probability", dropout_probability, float),
            ("attention_seed", attention_seed, int),
            ("parameter_seed", parameter_seed, int),
            ("attention_dropout_seed", attention_dropout_seed, int),
            ("feed_forward_dropout_seed", feed_forward_dropout_seed, int),
            ("device", device, torch.device),
            ("dtype", dtype, torch.dtype),
        )
        _validate_constructor_types("block", arguments)
        _validate_constructor_values(
            "block",
            (
                ("model_width", model_width, MODEL_WIDTH),
                ("number_of_heads", number_of_heads, NUMBER_OF_HEADS),
                ("head_width", head_width, HEAD_WIDTH),
                ("hidden_width", hidden_width, HIDDEN_WIDTH),
                ("max_sequence_length", max_sequence_length, MAX_SEQUENCE_LENGTH),
                ("layer_norm_epsilon", layer_norm_epsilon, LAYER_NORM_EPSILON),
                ("dropout_probability", dropout_probability, DROPOUT_PROBABILITY),
                ("attention_seed", attention_seed, ATTENTION_INITIALIZATION_SEED),
                ("parameter_seed", parameter_seed, PHASE6_PARAMETER_INITIALIZATION_SEED),
                ("attention_dropout_seed", attention_dropout_seed, ATTENTION_DROPOUT_SEED),
                ("feed_forward_dropout_seed", feed_forward_dropout_seed, FEED_FORWARD_DROPOUT_SEED),
                ("device", device, PARAMETER_DEVICE),
                ("dtype", dtype, PARAMETER_DTYPE),
            ),
        )
        super().__init__()
        self.norm1 = ExplicitLayerNorm(
            model_width=model_width,
            epsilon=layer_norm_epsilon,
            device=device,
            dtype=dtype,
        )
        self.attention = MultiHeadCausalSelfAttention(
            model_width=model_width,
            number_of_heads=number_of_heads,
            head_width=head_width,
            max_sequence_length=max_sequence_length,
            seed=attention_seed,
            device=device,
            dtype=dtype,
        )
        self.attention_dropout = ExplicitDropout(
            probability=dropout_probability,
            seed=attention_dropout_seed,
            device=device,
            dtype=dtype,
        )
        self.norm2 = ExplicitLayerNorm(
            model_width=model_width,
            epsilon=layer_norm_epsilon,
            device=device,
            dtype=dtype,
        )
        self.feed_forward = PositionwiseFeedForward(
            model_width=model_width,
            hidden_width=hidden_width,
            seed=parameter_seed,
            device=device,
            dtype=dtype,
        )
        self.feed_forward_dropout = ExplicitDropout(
            probability=dropout_probability,
            seed=feed_forward_dropout_seed,
            device=device,
            dtype=dtype,
        )

    def _validate_direct_children(self) -> None:
        children = tuple(self.named_children())
        expected_names = (
            "norm1",
            "attention",
            "attention_dropout",
            "norm2",
            "feed_forward",
            "feed_forward_dropout",
        )
        expected_types = (
            ExplicitLayerNorm,
            MultiHeadCausalSelfAttention,
            ExplicitDropout,
            ExplicitLayerNorm,
            PositionwiseFeedForward,
            ExplicitDropout,
        )
        _require_contract(
            tuple(name for name, _ in children) == expected_names
            and all(
                type(child) is expected_type
                for (_, child), expected_type in zip(
                    children, expected_types, strict=True
                )
            ),
            "block.submodules.structure",
        )

    def _validate_modes(self) -> None:
        if type(getattr(self, "training", None)) is not bool:
            raise TransformerBlockTypeError("dropout.mode.type")
        if type(getattr(self.attention_dropout, "training", None)) is not bool:
            raise TransformerBlockTypeError("dropout.mode.type")
        if type(getattr(self.feed_forward_dropout, "training", None)) is not bool:
            raise TransformerBlockTypeError("dropout.mode.type")
        _require_contract(
            self.attention_dropout.training == self.training
            and self.feed_forward_dropout.training == self.training,
            "block.dropout_modes.consistent",
        )

    def _validate_dropout_components(self) -> None:
        self.attention_dropout._validate_structure(
            expected_seed=ATTENTION_DROPOUT_SEED
        )
        self.feed_forward_dropout._validate_structure(
            expected_seed=FEED_FORWARD_DROPOUT_SEED
        )

    def _validate_new_parameters(self) -> None:
        structure = (
            tuple(self._parameters) == ()
            and tuple(name for name, _ in self.norm1.named_parameters())
            == ("gamma", "beta")
            and tuple(name for name, _ in self.norm2.named_parameters())
            == ("gamma", "beta")
            and tuple(name for name, _ in self.feed_forward.named_parameters())
            == ("first_weight", "first_bias", "second_weight", "second_bias")
            and tuple(name for name, _ in self.norm1.named_modules()) == ("",)
            and tuple(name for name, _ in self.norm2.named_modules()) == ("",)
            and tuple(name for name, _ in self.feed_forward.named_modules())
            == ("",)
            and tuple(self.named_buffers()) == ()
        )
        _require_contract(structure, "block.parameters.structure")
        items = (
            (getattr(self.norm1, "gamma", None), (32,), "block.parameter.norm1.gamma"),
            (getattr(self.norm1, "beta", None), (32,), "block.parameter.norm1.beta"),
            (getattr(self.norm2, "gamma", None), (32,), "block.parameter.norm2.gamma"),
            (getattr(self.norm2, "beta", None), (32,), "block.parameter.norm2.beta"),
            (
                getattr(self.feed_forward, "first_weight", None),
                (32, 128),
                "block.parameter.feed_forward.first_weight",
            ),
            (
                getattr(self.feed_forward, "first_bias", None),
                (128,),
                "block.parameter.feed_forward.first_bias",
            ),
            (
                getattr(self.feed_forward, "second_weight", None),
                (128, 32),
                "block.parameter.feed_forward.second_weight",
            ),
            (
                getattr(self.feed_forward, "second_bias", None),
                (32,),
                "block.parameter.feed_forward.second_bias",
            ),
        )
        _validate_parameters(items)

    def _validate_before_snapshot(
        self,
        representations: torch.Tensor,
    ) -> torch.Tensor:
        value = _validate_representation_input(representations, "block")
        self._validate_direct_children()
        self._validate_modes()
        self._validate_dropout_components()
        _require_contract(
            self.attention_dropout._generator
            is not self.feed_forward_dropout._generator,
            "block.submodules.structure",
        )
        self._validate_new_parameters()
        return value

    def _snapshot_stream(
        self,
        dropout: ExplicitDropout,
        invariant: str,
    ) -> torch.Tensor:
        return _snapshot_generator_state(dropout._generator, invariant)

    def _restore_both(
        self,
        attention_state: torch.Tensor,
        feed_forward_state: torch.Tensor,
        original: BaseException,
    ) -> None:
        failed: list[str] = []
        try:
            _restore_generator_state(
                self.attention_dropout._generator,
                attention_state,
            )
        except BaseException:
            failed.append("attention")
        try:
            _restore_generator_state(
                self.feed_forward_dropout._generator,
                feed_forward_state,
            )
        except BaseException:
            failed.append("feed_forward")
        if failed:
            raise TransformerBlockTransactionError(
                "block.dropout_transaction.rollback",
                failed_streams=tuple(failed),
            ) from original

    def _compute_forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized_attention_input = self.norm1(value)
        attention_output = self.attention(normalized_attention_input)
        attention_after_dropout = self.attention_dropout(attention_output)
        _require_finite(
            attention_after_dropout,
            "block.attention_dropout_output.finite",
        )
        first_residual = value + attention_after_dropout
        _require_finite(first_residual, "block.first_residual.finite")
        normalized_feed_forward_input = self.norm2(first_residual)
        feed_forward_output = self.feed_forward(normalized_feed_forward_input)
        feed_forward_after_dropout = self.feed_forward_dropout(feed_forward_output)
        _require_finite(
            feed_forward_after_dropout,
            "block.feed_forward_dropout_output.finite",
        )
        output = first_residual + feed_forward_after_dropout
        _require_finite(output, "block.output.finite")
        return output

    def _compute_inspection(
        self,
        value: torch.Tensor,
    ) -> TransformerBlockInspection:
        norm1 = self.norm1.inspect(value)
        attention = self.attention.inspect(norm1.output)
        attention_dropout = self.attention_dropout.inspect(attention.output)
        _require_finite(
            attention_dropout.output,
            "block.attention_dropout_output.finite",
        )
        first_residual = value + attention_dropout.output
        _require_finite(first_residual, "block.first_residual.finite")
        norm2 = self.norm2.inspect(first_residual)
        feed_forward = self.feed_forward.inspect(norm2.output)
        feed_forward_dropout = self.feed_forward_dropout.inspect(
            feed_forward.output
        )
        _require_finite(
            feed_forward_dropout.output,
            "block.feed_forward_dropout_output.finite",
        )
        output = first_residual + feed_forward_dropout.output
        _require_finite(output, "block.output.finite")
        return TransformerBlockInspection(
            normalized_attention_input=norm1.output,
            attention_branch_output_before_dropout=attention.output,
            attention_weights=attention.attention_weights,
            attention_dropout_keep_mask=attention_dropout.keep_mask,
            attention_branch_output_after_dropout=attention_dropout.output,
            first_residual=first_residual,
            normalized_feed_forward_input=norm2.output,
            feed_forward_pre_activation=feed_forward.pre_activation,
            feed_forward_hidden_activation=feed_forward.hidden_activation,
            feed_forward_branch_output_before_dropout=feed_forward.output,
            feed_forward_dropout_keep_mask=feed_forward_dropout.keep_mask,
            feed_forward_branch_output_after_dropout=feed_forward_dropout.output,
            output=output,
        )

    def _transaction(
        self,
        representations: torch.Tensor,
        operation: Callable[[torch.Tensor], _Result],
    ) -> _Result:
        value = self._validate_before_snapshot(representations)
        attention_state = self._snapshot_stream(
            self.attention_dropout,
            "block.dropout_transaction.snapshot.attention",
        )
        feed_forward_state = self._snapshot_stream(
            self.feed_forward_dropout,
            "block.dropout_transaction.snapshot.feed_forward",
        )
        try:
            return operation(value)
        except BaseException as original:
            self._restore_both(attention_state, feed_forward_state, original)
            raise

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self._transaction(representations, self._compute_forward)

    def inspect(self, representations: torch.Tensor) -> TransformerBlockInspection:
        return self._transaction(representations, self._compute_inspection)


__all__ = [
    "DropoutInspection",
    "ExplicitDropout",
    "ExplicitLayerNorm",
    "FeedForwardInspection",
    "LayerNormInspection",
    "PositionwiseFeedForward",
    "TransformerBlock",
    "TransformerBlockContractError",
    "TransformerBlockInspection",
    "TransformerBlockNumericalError",
    "TransformerBlockTransactionError",
    "TransformerBlockTypeError",
    "explicit_gelu",
]
