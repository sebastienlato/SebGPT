"""Explicit Phase 7 four-block decoder-only Mini-GPT."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Mapping

import torch
from torch import nn

from sebgpt.model.embeddings import TokenPositionEmbedding
from sebgpt.model.self_attention import (
    MultiHeadCausalSelfAttention,
    _AttentionHead,
)
from sebgpt.model.transformer_block import (
    ExplicitDropout,
    ExplicitLayerNorm,
    LayerNormInspection,
    PositionwiseFeedForward,
    TransformerBlock,
    TransformerBlockInspection,
)
from sebgpt.tokenization.vocabulary_artifact import (
    VocabularyBinding,
    _is_verified_vocabulary_binding,
)


VOCABULARY_SIZE: Final = 81
MODEL_WIDTH: Final = 32
MAX_SEQUENCE_LENGTH: Final = 256
NUMBER_OF_BLOCKS: Final = 4
NUMBER_OF_HEADS: Final = 4
HEAD_WIDTH: Final = 8
HIDDEN_WIDTH: Final = 128
LAYER_NORM_EPSILON: Final = 1e-5
DROPOUT_PROBABILITY: Final = 0.1
EMBEDDING_INITIALIZATION_SEED: Final = 1337
PHASE5_ATTENTION_CONSTRUCTION_SEED: Final = 5005
PHASE6_PARAMETER_CONSTRUCTION_SEED: Final = 6006
ATTENTION_DROPOUT_SEED: Final = 6007
FEED_FORWARD_DROPOUT_SEED: Final = 6008
BLOCK_PARAMETER_INITIALIZATION_SEEDS: Final = (7001, 7002, 7003, 7004)
OUTPUT_HEAD_INITIALIZATION_SEED: Final = 7005
PARAMETER_DEVICE: Final = torch.device("cpu")
PARAMETER_DTYPE: Final = torch.float32

_BLOCK_PARAMETER_SUFFIXES: Final = (
    "norm1.gamma",
    "norm1.beta",
    "attention.output_weight",
    "attention.heads.0.query_weight",
    "attention.heads.0.key_weight",
    "attention.heads.0.value_weight",
    "attention.heads.1.query_weight",
    "attention.heads.1.key_weight",
    "attention.heads.1.value_weight",
    "attention.heads.2.query_weight",
    "attention.heads.2.key_weight",
    "attention.heads.2.value_weight",
    "attention.heads.3.query_weight",
    "attention.heads.3.key_weight",
    "attention.heads.3.value_weight",
    "norm2.gamma",
    "norm2.beta",
    "feed_forward.first_weight",
    "feed_forward.first_bias",
    "feed_forward.second_weight",
    "feed_forward.second_bias",
)

_BLOCK_PARAMETER_SHAPES: Final = (
    (32,),
    (32,),
    (32, 32),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32, 8),
    (32,),
    (32,),
    (32, 128),
    (128,),
    (128, 32),
    (32,),
)


class MiniGPTTypeError(TypeError):
    """A deterministic Phase 7 type failure with structural facts only."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class MiniGPTContractError(ValueError):
    """A deterministic Phase 7 contract failure with structural facts only."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class MiniGPTNumericalError(ArithmeticError):
    """A deterministic Phase 7 numerical failure with structural facts only."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: object) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


@dataclass(frozen=True)
class MiniGPTInspection:
    """Same-call complete-model intermediates and logits."""

    embedding_representation: torch.Tensor = field(repr=False)
    block_inspections: tuple[TransformerBlockInspection, ...] = field(repr=False)
    final_normalization: LayerNormInspection = field(repr=False)
    logits: torch.Tensor = field(repr=False)


def _all_true(value: torch.Tensor) -> bool:
    return bool(torch.all(value).item())


def _require_contract(
    condition: bool,
    invariant: str,
    **safe_facts: object,
) -> None:
    if not condition:
        raise MiniGPTContractError(invariant, **safe_facts)


def _require_finite(value: torch.Tensor, invariant: str) -> None:
    if not _all_true(torch.isfinite(value)):
        raise MiniGPTNumericalError(invariant, shape=tuple(value.shape))


def _validate_head_constructor(
    *,
    model_width: object,
    vocabulary_size: object,
    seed: object,
    device: object,
    dtype: object,
) -> None:
    type_checks = (
        (type(model_width) is int, "lm_head.constructor.model_width.exact_int"),
        (
            type(vocabulary_size) is int,
            "lm_head.constructor.vocabulary_size.exact_int",
        ),
        (type(seed) is int, "lm_head.constructor.seed.exact_int"),
        (
            type(device) is torch.device,
            "lm_head.constructor.device.exact_torch_device",
        ),
        (isinstance(dtype, torch.dtype), "lm_head.constructor.dtype.torch_dtype"),
    )
    for valid, invariant in type_checks:
        if not valid:
            raise MiniGPTTypeError(invariant)

    value_checks = (
        (model_width == MODEL_WIDTH, "model_width", MODEL_WIDTH),
        (vocabulary_size == VOCABULARY_SIZE, "vocabulary_size", VOCABULARY_SIZE),
        (seed == OUTPUT_HEAD_INITIALIZATION_SEED, "seed", OUTPUT_HEAD_INITIALIZATION_SEED),
        (device == PARAMETER_DEVICE, "device", str(PARAMETER_DEVICE)),
        (dtype is PARAMETER_DTYPE, "dtype", str(PARAMETER_DTYPE)),
    )
    for valid, name, expected in value_checks:
        _require_contract(
            valid,
            f"lm_head.constructor.{name}.value",
            expected=expected,
        )


class LanguageModelHead(nn.Module):
    """One explicit untied affine map from width 32 to 81 logits."""

    def __init__(
        self,
        *,
        model_width: int,
        vocabulary_size: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        _validate_head_constructor(
            model_width=model_width,
            vocabulary_size=vocabulary_size,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        output_weight = torch.empty(
            (MODEL_WIDTH, VOCABULARY_SIZE),
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(
            output_weight,
            mean=0.0,
            std=1.0 / math.sqrt(MODEL_WIDTH),
            generator=generator,
        )
        self.output_weight = nn.Parameter(output_weight)
        self.output_bias = nn.Parameter(
            torch.zeros((VOCABULARY_SIZE,), device=device, dtype=dtype)
        )

    def _parameter_items(
        self,
    ) -> tuple[tuple[str, object, tuple[int, ...]], ...]:
        return (
            (
                "output_weight",
                getattr(self, "output_weight", None),
                (32, 81),
            ),
            (
                "output_bias",
                getattr(self, "output_bias", None),
                (81,),
            ),
        )

    def _validate_parameters(self) -> None:
        items = self._parameter_items()
        names = tuple(name for name, _ in self.named_parameters(remove_duplicate=False))
        _require_contract(
            names == ("output_weight", "output_bias")
            and tuple(self.named_buffers()) == ()
            and tuple(name for name, _ in self.named_modules()) == ("",),
            "lm_head.parameters.structure",
        )
        for name, parameter, _ in items:
            if not isinstance(parameter, nn.Parameter):
                raise MiniGPTTypeError(f"lm_head.parameter.{name}.type")
        for name, parameter, shape in items:
            assert isinstance(parameter, nn.Parameter)
            _require_contract(
                tuple(parameter.shape) == shape,
                f"lm_head.parameter.{name}.shape",
                expected=shape,
                observed=tuple(parameter.shape),
            )
        for name, parameter, _ in items:
            assert isinstance(parameter, nn.Parameter)
            _require_contract(
                parameter.device == PARAMETER_DEVICE,
                f"lm_head.parameter.{name}.device",
                expected=str(PARAMETER_DEVICE),
                observed=str(parameter.device),
            )
        for name, parameter, _ in items:
            assert isinstance(parameter, nn.Parameter)
            _require_contract(
                parameter.dtype is PARAMETER_DTYPE,
                f"lm_head.parameter.{name}.dtype",
                expected=str(PARAMETER_DTYPE),
                observed=str(parameter.dtype),
            )
        for name, parameter, _ in items:
            assert isinstance(parameter, nn.Parameter)
            _require_finite(parameter, f"lm_head.parameter.{name}.finite")

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        if not isinstance(representations, torch.Tensor):
            raise MiniGPTTypeError("lm_head.input.tensor")
        _require_contract(
            representations.dim() == 2,
            "lm_head.input.rank",
            expected=2,
            observed=representations.dim(),
        )
        _require_contract(
            representations.dtype is PARAMETER_DTYPE,
            "lm_head.input.dtype",
            expected=str(PARAMETER_DTYPE),
            observed=str(representations.dtype),
        )
        _require_contract(
            representations.device == PARAMETER_DEVICE,
            "lm_head.input.device",
            expected=str(PARAMETER_DEVICE),
            observed=str(representations.device),
        )
        _require_contract(
            representations.shape[1] == MODEL_WIDTH,
            "lm_head.input.model_width",
            expected=MODEL_WIDTH,
            observed=representations.shape[1],
        )
        sequence_length = representations.shape[0]
        _require_contract(
            1 <= sequence_length <= MAX_SEQUENCE_LENGTH,
            "lm_head.input.sequence_length",
            lower_bound=1,
            upper_bound_inclusive=MAX_SEQUENCE_LENGTH,
            observed=sequence_length,
        )
        _require_finite(representations, "lm_head.input.finite")
        self._validate_parameters()
        logits = representations @ self.output_weight + self.output_bias
        _require_finite(logits, "lm_head.output.finite")
        return logits


def _validate_model_constructor(
    vocabulary: object,
    *,
    model_width: object,
    max_sequence_length: object,
    number_of_blocks: object,
    number_of_heads: object,
    head_width: object,
    hidden_width: object,
    layer_norm_epsilon: object,
    dropout_probability: object,
    embedding_seed: object,
    block_parameter_seeds: object,
    output_head_seed: object,
    device: object,
    dtype: object,
) -> None:
    if type(vocabulary) is not VocabularyBinding:
        raise MiniGPTTypeError("mini_gpt.constructor.vocabulary.exact_type")

    integer_types = (
        ("model_width", model_width),
        ("max_sequence_length", max_sequence_length),
        ("number_of_blocks", number_of_blocks),
        ("number_of_heads", number_of_heads),
        ("head_width", head_width),
        ("hidden_width", hidden_width),
        ("embedding_seed", embedding_seed),
    )
    for name, value in integer_types:
        if type(value) is not int:
            raise MiniGPTTypeError(f"mini_gpt.constructor.{name}.exact_int")

    float_types = (
        ("layer_norm_epsilon", layer_norm_epsilon),
        ("dropout_probability", dropout_probability),
    )
    for name, value in float_types:
        if type(value) is not float:
            raise MiniGPTTypeError(f"mini_gpt.constructor.{name}.exact_float")

    if type(block_parameter_seeds) is not tuple:
        raise MiniGPTTypeError(
            "mini_gpt.constructor.block_parameter_seeds.exact_tuple"
        )
    _require_contract(
        len(block_parameter_seeds) == NUMBER_OF_BLOCKS,
        "mini_gpt.constructor.block_parameter_seeds.length",
        expected=NUMBER_OF_BLOCKS,
        observed=len(block_parameter_seeds),
    )
    for index, seed in enumerate(block_parameter_seeds):
        if type(seed) is not int:
            raise MiniGPTTypeError(
                f"mini_gpt.constructor.block_parameter_seeds.{index}.exact_int"
            )

    if type(output_head_seed) is not int:
        raise MiniGPTTypeError("mini_gpt.constructor.output_head_seed.exact_int")
    if type(device) is not torch.device:
        raise MiniGPTTypeError("mini_gpt.constructor.device.exact_torch_device")
    if not isinstance(dtype, torch.dtype):
        raise MiniGPTTypeError("mini_gpt.constructor.dtype.torch_dtype")

    _require_contract(
        _is_verified_vocabulary_binding(vocabulary),
        "mini_gpt.constructor.vocabulary.verified",
    )

    value_checks = (
        ("model_width", model_width, MODEL_WIDTH),
        ("max_sequence_length", max_sequence_length, MAX_SEQUENCE_LENGTH),
        ("number_of_blocks", number_of_blocks, NUMBER_OF_BLOCKS),
        ("number_of_heads", number_of_heads, NUMBER_OF_HEADS),
        ("head_width", head_width, HEAD_WIDTH),
        ("hidden_width", hidden_width, HIDDEN_WIDTH),
        ("layer_norm_epsilon", layer_norm_epsilon, LAYER_NORM_EPSILON),
        ("dropout_probability", dropout_probability, DROPOUT_PROBABILITY),
        ("embedding_seed", embedding_seed, EMBEDDING_INITIALIZATION_SEED),
        (
            "block_parameter_seeds",
            block_parameter_seeds,
            BLOCK_PARAMETER_INITIALIZATION_SEEDS,
        ),
        ("output_head_seed", output_head_seed, OUTPUT_HEAD_INITIALIZATION_SEED),
        ("device", device, PARAMETER_DEVICE),
        ("dtype", dtype, PARAMETER_DTYPE),
    )
    for name, observed, expected in value_checks:
        valid = observed is expected if name == "dtype" else observed == expected
        _require_contract(
            valid,
            f"mini_gpt.constructor.{name}.value",
            expected=str(expected),
        )

    _require_contract(
        len(set(block_parameter_seeds)) == NUMBER_OF_BLOCKS,
        "mini_gpt.constructor.block_parameter_seeds.distinct",
    )
    inherited_parameter_seeds = {
        EMBEDDING_INITIALIZATION_SEED,
        PHASE5_ATTENTION_CONSTRUCTION_SEED,
        PHASE6_PARAMETER_CONSTRUCTION_SEED,
        OUTPUT_HEAD_INITIALIZATION_SEED,
    }
    _require_contract(
        set(block_parameter_seeds).isdisjoint(inherited_parameter_seeds),
        "mini_gpt.constructor.block_parameter_seeds.disjoint",
    )


def _block_parameter_specs(
    block: TransformerBlock,
) -> tuple[tuple[str, object, tuple[int, ...]], ...]:
    heads = getattr(getattr(block, "attention", None), "heads", ())
    head_parameters: list[object] = []
    if isinstance(heads, nn.ModuleList):
        for head in heads:
            head_parameters.extend(
                (
                    getattr(head, "query_weight", None),
                    getattr(head, "key_weight", None),
                    getattr(head, "value_weight", None),
                )
            )
    objects = (
        getattr(getattr(block, "norm1", None), "gamma", None),
        getattr(getattr(block, "norm1", None), "beta", None),
        getattr(getattr(block, "attention", None), "output_weight", None),
        *head_parameters,
        getattr(getattr(block, "norm2", None), "gamma", None),
        getattr(getattr(block, "norm2", None), "beta", None),
        getattr(getattr(block, "feed_forward", None), "first_weight", None),
        getattr(getattr(block, "feed_forward", None), "first_bias", None),
        getattr(getattr(block, "feed_forward", None), "second_weight", None),
        getattr(getattr(block, "feed_forward", None), "second_bias", None),
    )
    return tuple(
        (name, parameter, shape)
        for name, parameter, shape in zip(
            _BLOCK_PARAMETER_SUFFIXES,
            objects,
            _BLOCK_PARAMETER_SHAPES,
            strict=True,
        )
    )


def _block_module_structure_is_exact(block: object) -> bool:
    if type(block) is not TransformerBlock:
        return False
    try:
        expected_children = (
            ("norm1", ExplicitLayerNorm),
            ("attention", MultiHeadCausalSelfAttention),
            ("attention_dropout", ExplicitDropout),
            ("norm2", ExplicitLayerNorm),
            ("feed_forward", PositionwiseFeedForward),
            ("feed_forward_dropout", ExplicitDropout),
        )
        children = tuple(block.named_children())
        if tuple(name for name, _ in children) != tuple(
            name for name, _ in expected_children
        ):
            return False
        if not all(
            type(child) is expected_type
            for (_, child), (_, expected_type) in zip(
                children, expected_children, strict=True
            )
        ):
            return False
        heads = getattr(block.attention, "heads", None)
        if type(heads) is not nn.ModuleList or len(heads) != NUMBER_OF_HEADS:
            return False
        return (
            tuple(name for name, _ in block.named_modules())
            == (
                "",
                "norm1",
                "attention",
                "attention.heads",
                "attention.heads.0",
                "attention.heads.1",
                "attention.heads.2",
                "attention.heads.3",
                "attention_dropout",
                "norm2",
                "feed_forward",
                "feed_forward_dropout",
            )
            and tuple(block.named_buffers()) == ()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _block_initialization_structure_is_exact(block: object) -> bool:
    if not _block_module_structure_is_exact(block):
        return False
    assert isinstance(block, TransformerBlock)
    try:
        specs = _block_parameter_specs(block)
        parameters = tuple(block.named_parameters(remove_duplicate=False))
        return (
            tuple(name for name, _ in parameters) == _BLOCK_PARAMETER_SUFFIXES
            and len(specs) == len(_BLOCK_PARAMETER_SUFFIXES)
            and all(
                isinstance(parameter, nn.Parameter)
                and tuple(parameter.shape) == shape
                and parameter.device == PARAMETER_DEVICE
                and parameter.dtype is PARAMETER_DTYPE
                for _, parameter, shape in specs
            )
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _randomized_block_parameters(
    block: TransformerBlock,
) -> tuple[nn.Parameter, ...]:
    parameters: list[nn.Parameter] = []
    for head in block.attention.heads:
        parameters.extend(
            (head.query_weight, head.key_weight, head.value_weight)
        )
    parameters.extend(
        (
            block.attention.output_weight,
            block.feed_forward.first_weight,
            block.feed_forward.second_weight,
        )
    )
    return tuple(parameters)


def _normal_tensor(
    shape: tuple[int, ...],
    *,
    standard_deviation: float,
    generator: torch.Generator,
) -> torch.Tensor:
    value = torch.empty(shape, device=PARAMETER_DEVICE, dtype=PARAMETER_DTYPE)
    nn.init.normal_(
        value,
        mean=0.0,
        std=standard_deviation,
        generator=generator,
    )
    return value


def _initialize_block_for_phase7(
    block: TransformerBlock,
    *,
    block_index: int,
    seed: int,
) -> None:
    _require_contract(
        _block_initialization_structure_is_exact(block),
        f"mini_gpt.initialization.block.{block_index}.structure",
    )
    children_before = tuple(child for _, child in block.named_children())
    heads_before = tuple(block.attention.heads)
    parameters_before = tuple(
        parameter
        for _, parameter in block.named_parameters(remove_duplicate=False)
    )
    attention_dropout_before = block.attention_dropout
    feed_forward_dropout_before = block.feed_forward_dropout
    attention_generator_before = block.attention_dropout._generator
    feed_forward_generator_before = block.feed_forward_dropout._generator
    attention_state_before = attention_generator_before.get_state().clone()
    feed_forward_state_before = feed_forward_generator_before.get_state().clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_parameters = _randomized_block_parameters(block)
    random_shapes_and_scales = (
        *((tuple(parameter.shape), 1.0 / math.sqrt(32)) for parameter in random_parameters[:14]),
        ((128, 32), 1.0 / math.sqrt(128)),
    )
    _require_contract(
        len(random_parameters) == 15,
        f"mini_gpt.initialization.block.{block_index}.structure",
    )
    expected_random_values: list[torch.Tensor] = []
    with torch.no_grad():
        for parameter, (shape, scale) in zip(
            random_parameters,
            random_shapes_and_scales,
            strict=True,
        ):
            expected = _normal_tensor(
                shape,
                standard_deviation=scale,
                generator=generator,
            )
            expected_random_values.append(expected)
            parameter.copy_(expected)
        block.norm1.gamma.fill_(1.0)
        block.norm1.beta.zero_()
        block.norm2.gamma.fill_(1.0)
        block.norm2.beta.zero_()
        block.feed_forward.first_bias.zero_()
        block.feed_forward.second_bias.zero_()

    children_after = tuple(child for _, child in block.named_children())
    heads_after = tuple(block.attention.heads)
    parameters_after = tuple(
        parameter
        for _, parameter in block.named_parameters(remove_duplicate=False)
    )
    _require_contract(
        children_after == children_before
        and heads_after == heads_before
        and block.attention_dropout is attention_dropout_before
        and block.feed_forward_dropout is feed_forward_dropout_before,
        f"mini_gpt.initialization.block.{block_index}.structure",
    )
    _require_contract(
        all(
            after is before
            for after, before in zip(
                parameters_after, parameters_before, strict=True
            )
        ),
        f"mini_gpt.initialization.block.{block_index}.parameter_identity",
    )
    deterministic_values_are_exact = (
        torch.equal(block.norm1.gamma, torch.ones_like(block.norm1.gamma))
        and torch.equal(block.norm1.beta, torch.zeros_like(block.norm1.beta))
        and torch.equal(block.norm2.gamma, torch.ones_like(block.norm2.gamma))
        and torch.equal(block.norm2.beta, torch.zeros_like(block.norm2.beta))
        and torch.equal(
            block.feed_forward.first_bias,
            torch.zeros_like(block.feed_forward.first_bias),
        )
        and torch.equal(
            block.feed_forward.second_bias,
            torch.zeros_like(block.feed_forward.second_bias),
        )
        and all(
            torch.equal(parameter, expected)
            for parameter, expected in zip(
                random_parameters,
                expected_random_values,
                strict=True,
            )
        )
        and all(_all_true(torch.isfinite(parameter)) for parameter in parameters_after)
    )
    _require_contract(
        deterministic_values_are_exact,
        f"mini_gpt.initialization.block.{block_index}.deterministic_values",
    )
    _require_contract(
        block.attention_dropout._generator is attention_generator_before
        and block.feed_forward_dropout._generator is feed_forward_generator_before
        and torch.equal(
            block.attention_dropout._generator.get_state(),
            attention_state_before,
        )
        and torch.equal(
            block.feed_forward_dropout._generator.get_state(),
            feed_forward_state_before,
        ),
        f"mini_gpt.initialization.block.{block_index}.dropout_state",
    )


def _blocks_are_pairwise_nonidentical(
    blocks: tuple[TransformerBlock, ...],
) -> bool:
    for left_index in range(len(blocks)):
        left = _randomized_block_parameters(blocks[left_index])
        for right_index in range(left_index + 1, len(blocks)):
            right = _randomized_block_parameters(blocks[right_index])
            if all(
                torch.equal(left_parameter, right_parameter)
                for left_parameter, right_parameter in zip(
                    left, right, strict=True
                )
            ):
                return False
    return True


class MiniGPT(nn.Module):
    """Four explicit causal Transformer blocks from token IDs to logits."""

    def __init__(
        self,
        vocabulary: VocabularyBinding,
        *,
        model_width: int,
        max_sequence_length: int,
        number_of_blocks: int,
        number_of_heads: int,
        head_width: int,
        hidden_width: int,
        layer_norm_epsilon: float,
        dropout_probability: float,
        embedding_seed: int,
        block_parameter_seeds: tuple[int, int, int, int],
        output_head_seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        _validate_model_constructor(
            vocabulary,
            model_width=model_width,
            max_sequence_length=max_sequence_length,
            number_of_blocks=number_of_blocks,
            number_of_heads=number_of_heads,
            head_width=head_width,
            hidden_width=hidden_width,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout_probability=dropout_probability,
            embedding_seed=embedding_seed,
            block_parameter_seeds=block_parameter_seeds,
            output_head_seed=output_head_seed,
            device=device,
            dtype=dtype,
        )
        global_rng_before = torch.get_rng_state().clone()
        super().__init__()
        self._configuration = MappingProxyType(
            {
                "model_width": model_width,
                "max_sequence_length": max_sequence_length,
                "number_of_blocks": number_of_blocks,
                "number_of_heads": number_of_heads,
                "head_width": head_width,
                "hidden_width": hidden_width,
                "layer_norm_epsilon": layer_norm_epsilon,
                "dropout_probability": dropout_probability,
                "embedding_seed": embedding_seed,
                "block_parameter_seeds": block_parameter_seeds,
                "output_head_seed": output_head_seed,
                "device": device,
                "dtype": dtype,
            }
        )
        self.representation = TokenPositionEmbedding(
            vocabulary,
            embedding_dim=model_width,
            max_positions=max_sequence_length,
            seed=embedding_seed,
            device=device,
            dtype=dtype,
        )

        blocks: list[TransformerBlock] = []
        for block_index, parameter_seed in enumerate(block_parameter_seeds):
            block = TransformerBlock(
                model_width=model_width,
                number_of_heads=number_of_heads,
                head_width=head_width,
                hidden_width=hidden_width,
                max_sequence_length=max_sequence_length,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout_probability=dropout_probability,
                attention_seed=PHASE5_ATTENTION_CONSTRUCTION_SEED,
                parameter_seed=PHASE6_PARAMETER_CONSTRUCTION_SEED,
                attention_dropout_seed=ATTENTION_DROPOUT_SEED,
                feed_forward_dropout_seed=FEED_FORWARD_DROPOUT_SEED,
                device=device,
                dtype=dtype,
            )
            _initialize_block_for_phase7(
                block,
                block_index=block_index,
                seed=parameter_seed,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        _require_contract(
            _blocks_are_pairwise_nonidentical(tuple(self.blocks)),
            "mini_gpt.initialization.blocks.nonidentical",
        )
        self.final_norm = ExplicitLayerNorm(
            model_width=model_width,
            epsilon=layer_norm_epsilon,
            device=device,
            dtype=dtype,
        )
        self.head = LanguageModelHead(
            model_width=model_width,
            vocabulary_size=VOCABULARY_SIZE,
            seed=output_head_seed,
            device=device,
            dtype=dtype,
        )
        self._validate_direct_children()
        self._validate_blocks()
        self._validate_modes()
        self._validate_parameters()
        if not torch.equal(torch.get_rng_state(), global_rng_before):
            torch.set_rng_state(global_rng_before)
            raise MiniGPTContractError("mini_gpt.initialization.global_rng")

    def _expected_parameter_specs(
        self,
    ) -> tuple[tuple[str, object, tuple[int, ...]], ...]:
        specs: list[tuple[str, object, tuple[int, ...]]] = [
            (
                "representation.token_embeddings",
                getattr(self.representation, "token_embeddings", None),
                (81, 32),
            ),
            (
                "representation.position_embeddings",
                getattr(self.representation, "position_embeddings", None),
                (256, 32),
            ),
        ]
        for block_index, block in enumerate(self.blocks):
            specs.extend(
                (
                    f"blocks.{block_index}.{suffix}",
                    parameter,
                    shape,
                )
                for suffix, parameter, shape in _block_parameter_specs(block)
            )
        specs.extend(
            (
                (
                    "final_norm.gamma",
                    getattr(self.final_norm, "gamma", None),
                    (32,),
                ),
                (
                    "final_norm.beta",
                    getattr(self.final_norm, "beta", None),
                    (32,),
                ),
                (
                    "head.output_weight",
                    getattr(self.head, "output_weight", None),
                    (32, 81),
                ),
                (
                    "head.output_bias",
                    getattr(self.head, "output_bias", None),
                    (81,),
                ),
            )
        )
        return tuple(specs)

    def _validate_direct_children(self) -> None:
        children = tuple(self._modules.items())
        expected = (
            ("representation", TokenPositionEmbedding),
            ("blocks", nn.ModuleList),
            ("final_norm", ExplicitLayerNorm),
            ("head", LanguageModelHead),
        )
        _require_contract(
            tuple(name for name, _ in children)
            == tuple(name for name, _ in expected)
            and tuple(self._buffers) == (),
            "mini_gpt.submodules.structure",
        )
        for (name, child), (_, expected_type) in zip(
            children, expected, strict=True
        ):
            if type(child) is not expected_type:
                raise MiniGPTTypeError(
                    "mini_gpt.submodules.structure",
                    child=name,
                )

    def _validate_blocks(self) -> None:
        _require_contract(
            type(self.blocks) is nn.ModuleList
            and len(self.blocks) == NUMBER_OF_BLOCKS
            and tuple(self.blocks._modules)
            == tuple(str(index) for index in range(NUMBER_OF_BLOCKS)),
            "mini_gpt.blocks.structure",
        )
        for block_index, block in enumerate(self.blocks):
            if type(block) is not TransformerBlock:
                raise MiniGPTTypeError(
                    "mini_gpt.blocks.structure",
                    block_index=block_index,
                )
            expected_children = (
                ("norm1", ExplicitLayerNorm),
                ("attention", MultiHeadCausalSelfAttention),
                ("attention_dropout", ExplicitDropout),
                ("norm2", ExplicitLayerNorm),
                ("feed_forward", PositionwiseFeedForward),
                ("feed_forward_dropout", ExplicitDropout),
            )
            children = tuple(block._modules.items())
            _require_contract(
                tuple(name for name, _ in children)
                == tuple(name for name, _ in expected_children),
                "mini_gpt.blocks.structure",
                block_index=block_index,
            )
            for (name, child), (_, expected_type) in zip(
                children, expected_children, strict=True
            ):
                if type(child) is not expected_type:
                    raise MiniGPTTypeError(
                        "mini_gpt.blocks.structure",
                        block_index=block_index,
                        child=name,
                    )
            _require_contract(
                tuple(block.attention._modules) == ("heads",),
                "mini_gpt.blocks.structure",
                block_index=block_index,
            )
            heads = block.attention._modules["heads"]
            if type(heads) is not nn.ModuleList:
                raise MiniGPTTypeError(
                    "mini_gpt.blocks.structure",
                    block_index=block_index,
                    child="attention.heads",
                )
            _require_contract(
                len(heads) == NUMBER_OF_HEADS
                and tuple(heads._modules)
                == tuple(str(index) for index in range(NUMBER_OF_HEADS)),
                "mini_gpt.blocks.structure",
                block_index=block_index,
            )
            for head_index, head in enumerate(heads):
                if type(head) is not _AttentionHead:
                    raise MiniGPTTypeError(
                        "mini_gpt.blocks.structure",
                        block_index=block_index,
                        child=f"attention.heads.{head_index}",
                    )
            _require_contract(
                tuple(name for name, _ in block.named_modules())
                == (
                    "",
                    "norm1",
                    "attention",
                    "attention.heads",
                    "attention.heads.0",
                    "attention.heads.1",
                    "attention.heads.2",
                    "attention.heads.3",
                    "attention_dropout",
                    "norm2",
                    "feed_forward",
                    "feed_forward_dropout",
                ),
                "mini_gpt.blocks.structure",
                block_index=block_index,
            )
        _require_contract(
            tuple(self.named_buffers()) == (),
            "mini_gpt.submodules.structure",
        )

    def _validate_modes(self) -> None:
        if type(getattr(self, "training", None)) is not bool:
            raise MiniGPTTypeError("mini_gpt.modes.type")
        for block_index, block in enumerate(self.blocks):
            states = (
                getattr(block, "training", None),
                getattr(block.attention_dropout, "training", None),
                getattr(block.feed_forward_dropout, "training", None),
            )
            if any(type(state) is not bool for state in states):
                raise MiniGPTTypeError(
                    "mini_gpt.modes.type",
                    block_index=block_index,
                )
            _require_contract(
                all(state == self.training for state in states),
                "mini_gpt.modes.consistent",
                block_index=block_index,
            )

    def _validate_parameters(self) -> None:
        expected = self._expected_parameter_specs()
        observed = tuple(self.named_parameters(remove_duplicate=False))
        _require_contract(
            tuple(name for name, _ in observed)
            == tuple(name for name, _, _ in expected),
            "mini_gpt.parameters.enumeration",
        )
        for (name, parameter), (_, expected_parameter, _) in zip(
            observed, expected, strict=True
        ):
            _require_contract(
                parameter is expected_parameter,
                f"mini_gpt.parameter.{name}.ownership",
            )
        for name, parameter in observed:
            if not isinstance(parameter, nn.Parameter):
                raise MiniGPTTypeError(f"mini_gpt.parameter.{name}.type")
        for (name, parameter), (_, _, shape) in zip(
            observed, expected, strict=True
        ):
            assert isinstance(parameter, nn.Parameter)
            _require_contract(
                tuple(parameter.shape) == shape,
                f"mini_gpt.parameter.{name}.shape",
                expected=shape,
                observed=tuple(parameter.shape),
            )
        for name, parameter in observed:
            assert isinstance(parameter, nn.Parameter)
            _require_contract(
                parameter.device == PARAMETER_DEVICE,
                f"mini_gpt.parameter.{name}.device",
                expected=str(PARAMETER_DEVICE),
                observed=str(parameter.device),
            )
        for name, parameter in observed:
            assert isinstance(parameter, nn.Parameter)
            _require_contract(
                parameter.dtype is PARAMETER_DTYPE,
                f"mini_gpt.parameter.{name}.dtype",
                expected=str(PARAMETER_DTYPE),
                observed=str(parameter.dtype),
            )
        for name, parameter in observed:
            assert isinstance(parameter, nn.Parameter)
            _require_finite(parameter, f"mini_gpt.parameter.{name}.finite")
        identities = tuple(id(parameter) for _, parameter in observed)
        _require_contract(
            len(set(identities)) == len(identities),
            "mini_gpt.parameters.unique",
        )
        token_weight = self.representation.token_embeddings
        output_weight = self.head.output_weight
        _require_contract(
            output_weight is not token_weight
            and output_weight.untyped_storage().data_ptr()
            != token_weight.untyped_storage().data_ptr(),
            "mini_gpt.head.untied",
        )

    def _validate_input(self, token_ids: object) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor):
            raise MiniGPTTypeError("mini_gpt.input.tensor")
        _require_contract(
            token_ids.dim() == 1,
            "mini_gpt.input.rank",
            expected=1,
            observed=token_ids.dim(),
        )
        _require_contract(
            token_ids.dtype is torch.long,
            "mini_gpt.input.dtype",
            expected=str(torch.long),
            observed=str(token_ids.dtype),
        )
        _require_contract(
            token_ids.device == PARAMETER_DEVICE,
            "mini_gpt.input.device",
            expected=str(PARAMETER_DEVICE),
            observed=str(token_ids.device),
        )
        sequence_length = token_ids.shape[0]
        _require_contract(
            1 <= sequence_length <= MAX_SEQUENCE_LENGTH,
            "mini_gpt.input.sequence_length",
            lower_bound=1,
            upper_bound_inclusive=MAX_SEQUENCE_LENGTH,
            observed=sequence_length,
        )
        _require_contract(
            _all_true((token_ids >= 0) & (token_ids < VOCABULARY_SIZE)),
            "mini_gpt.input.token_id_range",
            lower_bound=0,
            upper_bound_exclusive=VOCABULARY_SIZE,
            shape=tuple(token_ids.shape),
        )
        return token_ids

    def _validate_before_arithmetic(self, token_ids: object) -> torch.Tensor:
        value = self._validate_input(token_ids)
        self._validate_direct_children()
        self._validate_blocks()
        self._validate_modes()
        self._validate_parameters()
        return value

    def _representation(self, token_ids: torch.Tensor) -> torch.Tensor:
        representation = self.representation(token_ids)
        _require_finite(representation, "mini_gpt.embedding_output.finite")
        return representation

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        value = self._validate_before_arithmetic(token_ids)
        hidden = self._representation(value)
        for block_index, block in enumerate(self.blocks):
            hidden = block(hidden)
            _require_finite(hidden, f"mini_gpt.block_output.{block_index}.finite")
        hidden = self.final_norm(hidden)
        _require_finite(hidden, "mini_gpt.final_normalization_output.finite")
        logits = self.head(hidden)
        _require_finite(logits, "mini_gpt.output.finite")
        return logits

    def inspect(self, token_ids: torch.Tensor) -> MiniGPTInspection:
        value = self._validate_before_arithmetic(token_ids)
        embedding_representation = self._representation(value)
        hidden = embedding_representation
        block_inspections: list[TransformerBlockInspection] = []
        for block_index, block in enumerate(self.blocks):
            inspection = block.inspect(hidden)
            _require_contract(
                type(inspection) is TransformerBlockInspection,
                "mini_gpt.inspection.handoff_identity",
                block_index=block_index,
            )
            hidden = inspection.output
            _require_finite(hidden, f"mini_gpt.block_output.{block_index}.finite")
            block_inspections.append(inspection)
        _require_contract(
            len(block_inspections) == NUMBER_OF_BLOCKS,
            "mini_gpt.inspection.block_count",
            expected=NUMBER_OF_BLOCKS,
            observed=len(block_inspections),
        )
        final_normalization = self.final_norm.inspect(hidden)
        _require_contract(
            type(final_normalization) is LayerNormInspection,
            "mini_gpt.inspection.final_normalization_identity",
        )
        _require_finite(
            final_normalization.output,
            "mini_gpt.final_normalization_output.finite",
        )
        logits = self.head(final_normalization.output)
        _require_finite(logits, "mini_gpt.output.finite")
        captured_blocks = tuple(block_inspections)
        result = MiniGPTInspection(
            embedding_representation=embedding_representation,
            block_inspections=captured_blocks,
            final_normalization=final_normalization,
            logits=logits,
        )
        observed_blocks = getattr(result, "block_inspections", None)
        _require_contract(
            type(observed_blocks) is tuple
            and len(observed_blocks) == NUMBER_OF_BLOCKS,
            "mini_gpt.inspection.block_count",
            expected=NUMBER_OF_BLOCKS,
        )
        assert isinstance(observed_blocks, tuple)
        _require_contract(
            getattr(result, "embedding_representation", None)
            is embedding_representation
            and all(
                observed is expected
                for observed, expected in zip(
                    observed_blocks,
                    captured_blocks,
                    strict=True,
                )
            ),
            "mini_gpt.inspection.handoff_identity",
        )
        _require_contract(
            getattr(result, "final_normalization", None) is final_normalization,
            "mini_gpt.inspection.final_normalization_identity",
        )
        _require_contract(
            getattr(result, "logits", None) is logits,
            "mini_gpt.inspection.logits_identity",
        )
        return result


__all__ = [
    "LanguageModelHead",
    "MiniGPT",
    "MiniGPTContractError",
    "MiniGPTInspection",
    "MiniGPTNumericalError",
    "MiniGPTTypeError",
]
