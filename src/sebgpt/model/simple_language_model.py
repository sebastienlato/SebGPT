"""Transparent Phase 4 non-attention language-model mechanics."""

from __future__ import annotations

import hashlib
import math
from typing import Final

import torch
from torch import nn

from sebgpt.data.shifted_examples import (
    Phase4ContractError,
    Phase4GovernanceError,
    Phase4TypeError,
)
from sebgpt.model.embeddings import TokenPositionEmbedding
from sebgpt.tokenization.vocabulary_artifact import (
    VocabularyBinding,
    _is_verified_vocabulary_binding,
)


VOCABULARY_SIZE: Final = 81
EMBEDDING_DIM: Final = 32
MAX_POSITIONS: Final = 256
CONTEXT_LENGTH: Final = 64
EMBEDDING_SEED: Final = 1337
OUTPUT_HEAD_SEED: Final = 4004
PARAMETER_DEVICE: Final = torch.device("cpu")
PARAMETER_DTYPE: Final = torch.float32

_PARAMETER_NAMES: Final = (
    "output_weight",
    "output_bias",
    "representation.token_embeddings",
    "representation.position_embeddings",
)
_PARAMETER_SHAPES: Final = (
    (32, 81),
    (81,),
    (81, 32),
    (256, 32),
)


def _require(condition: bool, invariant: str, **safe_facts: object) -> None:
    if not condition:
        raise Phase4ContractError(invariant, **safe_facts)


def _all_true(value: torch.Tensor) -> bool:
    return bool(torch.all(value).item())


def _require_exact_constructor_types(
    vocabulary: VocabularyBinding,
    *,
    embedding_dim: int,
    max_positions: int,
    embedding_seed: int,
    output_head_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if type(vocabulary) is not VocabularyBinding:
        raise Phase4TypeError("model.vocabulary.binding_type")
    integer_arguments = (
        ("embedding_dim", embedding_dim),
        ("max_positions", max_positions),
        ("embedding_seed", embedding_seed),
        ("output_head_seed", output_head_seed),
    )
    for name, value in integer_arguments:
        if type(value) is not int:
            raise Phase4TypeError(f"model.{name}.exact_int")
    if type(device) is not torch.device:
        raise Phase4TypeError("model.device.exact_torch_device")
    if not isinstance(dtype, torch.dtype):
        raise Phase4TypeError("model.dtype.torch_dtype")


def _require_exact_constructor_values(
    vocabulary: VocabularyBinding,
    *,
    embedding_dim: int,
    max_positions: int,
    embedding_seed: int,
    output_head_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _require(
        _is_verified_vocabulary_binding(vocabulary),
        "model.vocabulary.verified",
    )
    values = (
        ("embedding_dim", embedding_dim, EMBEDDING_DIM),
        ("max_positions", max_positions, MAX_POSITIONS),
        ("embedding_seed", embedding_seed, EMBEDDING_SEED),
        ("output_head_seed", output_head_seed, OUTPUT_HEAD_SEED),
        ("device", device, PARAMETER_DEVICE),
        ("dtype", dtype, PARAMETER_DTYPE),
    )
    for name, observed, expected in values:
        _require(observed == expected, f"model.{name}.value", expected=str(expected))


class SimpleNeuralLanguageModel(nn.Module):
    """A positionwise linear head over the accepted Phase 3 representation."""

    def __init__(
        self,
        vocabulary: VocabularyBinding,
        *,
        embedding_dim: int,
        max_positions: int,
        embedding_seed: int,
        output_head_seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        _require_exact_constructor_types(
            vocabulary,
            embedding_dim=embedding_dim,
            max_positions=max_positions,
            embedding_seed=embedding_seed,
            output_head_seed=output_head_seed,
            device=device,
            dtype=dtype,
        )
        _require_exact_constructor_values(
            vocabulary,
            embedding_dim=embedding_dim,
            max_positions=max_positions,
            embedding_seed=embedding_seed,
            output_head_seed=output_head_seed,
            device=device,
            dtype=dtype,
        )
        super().__init__()

        self.representation = TokenPositionEmbedding(
            vocabulary,
            embedding_dim=embedding_dim,
            max_positions=max_positions,
            seed=embedding_seed,
            device=device,
            dtype=dtype,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(output_head_seed)
        output_weights = torch.empty(
            (embedding_dim, vocabulary.vocabulary_size),
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(
            output_weights,
            mean=0.0,
            std=1.0 / math.sqrt(embedding_dim),
            generator=generator,
        )
        self.output_weight = nn.Parameter(output_weights)
        self.output_bias = nn.Parameter(
            torch.zeros(
                (vocabulary.vocabulary_size,),
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return one row of next-token logits per input position."""

        if not isinstance(token_ids, torch.Tensor):
            raise Phase4TypeError("model.input.tensor")
        _require(token_ids.dim() == 1, "model.input.rank", expected=1)
        _require(
            token_ids.dtype is torch.long,
            "model.input.dtype",
            expected=str(torch.long),
        )
        parameter_devices = tuple(
            parameter.device for _, parameter in self.named_parameters()
        )
        _require(
            token_ids.device == PARAMETER_DEVICE
            and all(device == PARAMETER_DEVICE for device in parameter_devices),
            "model.input.device",
            expected=str(PARAMETER_DEVICE),
        )
        length = token_ids.shape[0]
        _require(
            0 <= length <= CONTEXT_LENGTH,
            "model.input.length",
            lower_bound=0,
            upper_bound_inclusive=CONTEXT_LENGTH,
        )
        if token_ids.numel() > 0:
            _require(
                _all_true((token_ids >= 0) & (token_ids < VOCABULARY_SIZE)),
                "model.input.token_id_range",
                lower_bound=0,
                upper_bound_exclusive=VOCABULARY_SIZE,
                shape=tuple(token_ids.shape),
            )

        representations = self.representation(token_ids)
        return representations @ self.output_weight + self.output_bias


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return stable row-wise probabilities for educational inspection."""

    if not isinstance(logits, torch.Tensor):
        raise Phase4TypeError("probabilities.logits.tensor")
    _require(logits.dim() == 2, "probabilities.logits.rank", expected=2)
    _require(
        logits.dtype is torch.float32,
        "probabilities.logits.dtype",
        expected=str(torch.float32),
    )
    _require(
        logits.device == PARAMETER_DEVICE,
        "probabilities.logits.device",
        expected=str(PARAMETER_DEVICE),
    )
    _require(
        logits.shape[1] == VOCABULARY_SIZE,
        "probabilities.logits.class_dimension",
        expected=VOCABULARY_SIZE,
    )
    length = logits.shape[0]
    _require(length >= 0, "probabilities.logits.length", lower_bound=0)
    _require(
        _all_true(torch.isfinite(logits)),
        "probabilities.logits.finite",
    )
    if length == 0:
        return logits.clone()

    row_max = logits.max(dim=1, keepdim=True).values
    shifted = logits - row_max
    unnormalized = torch.exp(shifted)
    return unnormalized / unnormalized.sum(dim=1, keepdim=True)


def explicit_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute cancellation-safe mean cross-entropy from primitive operations."""

    if not isinstance(logits, torch.Tensor):
        raise Phase4TypeError("loss.logits.tensor")
    if not isinstance(targets, torch.Tensor):
        raise Phase4TypeError("loss.targets.tensor")
    _require(logits.dim() == 2, "loss.logits.rank", expected=2)
    _require(targets.dim() == 1, "loss.targets.rank", expected=1)
    _require(
        logits.dtype is torch.float32,
        "loss.logits.dtype",
        expected=str(torch.float32),
    )
    _require(
        targets.dtype is torch.long,
        "loss.targets.dtype",
        expected=str(torch.long),
    )
    _require(
        logits.device == PARAMETER_DEVICE
        and targets.device == PARAMETER_DEVICE,
        "loss.input.device",
        expected=str(PARAMETER_DEVICE),
    )
    _require(
        logits.shape[1] == VOCABULARY_SIZE,
        "loss.logits.class_dimension",
        expected=VOCABULARY_SIZE,
    )
    _require(
        logits.shape[0] == targets.shape[0],
        "loss.input.length_match",
    )
    length = targets.shape[0]
    _require(length > 0, "loss.input.nonempty")
    _require(_all_true(torch.isfinite(logits)), "loss.logits.finite")
    _require(
        _all_true((targets >= 0) & (targets < VOCABULARY_SIZE)),
        "loss.targets.range",
        lower_bound=0,
        upper_bound_exclusive=VOCABULARY_SIZE,
        shape=tuple(targets.shape),
    )

    row_max = logits.max(dim=1, keepdim=True).values
    shifted = logits - row_max
    _require(
        _all_true(torch.isfinite(shifted) | torch.isneginf(shifted)),
        "loss.unrepresentable.shifted",
    )
    shifted_exp = torch.exp(shifted)
    _require(
        _all_true(torch.isfinite(shifted_exp) & (shifted_exp >= 0)),
        "loss.unrepresentable.shifted_exp",
    )
    shifted_sum = shifted_exp.sum(dim=1)
    _require(
        _all_true(torch.isfinite(shifted_sum) & (shifted_sum > 0)),
        "loss.unrepresentable.shifted_sum",
    )
    log_shifted_sum = torch.log(shifted_sum)
    _require(
        _all_true(torch.isfinite(log_shifted_sum)),
        "loss.unrepresentable.log_shifted_sum",
    )
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_margin = row_max.squeeze(1) - target_logits
    _require(
        _all_true(torch.isfinite(target_margin)),
        "loss.unrepresentable.target_margin",
    )
    negative_log_likelihood = log_shifted_sum + target_margin
    _require(
        _all_true(torch.isfinite(negative_log_likelihood)),
        "loss.unrepresentable.row",
    )
    scaled_rows = negative_log_likelihood / length
    _require(
        _all_true(torch.isfinite(scaled_rows)),
        "loss.unrepresentable.scaled_row",
    )
    loss = scaled_rows.sum()
    _require(torch.isfinite(loss).item(), "loss.unrepresentable.reduction")
    return loss


def _expected_parameter_objects(
    model: SimpleNeuralLanguageModel,
) -> tuple[nn.Parameter, ...]:
    return (
        model.output_weight,
        model.output_bias,
        model.representation.token_embeddings,
        model.representation.position_embeddings,
    )


def _validated_parameter_items(
    model: SimpleNeuralLanguageModel,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if type(model) is not SimpleNeuralLanguageModel:
        raise Phase4TypeError("parameters.model.exact_type")
    items = tuple(model.named_parameters())
    _require(
        tuple(name for name, _ in items) == _PARAMETER_NAMES,
        "parameters.enumeration",
    )
    expected_objects = _expected_parameter_objects(model)
    for position, ((_, parameter), expected_object) in enumerate(
        zip(items, expected_objects, strict=True)
    ):
        if not isinstance(parameter, nn.Parameter):
            raise Phase4TypeError("parameters.parameter.type", position=position)
        _require(
            parameter is expected_object,
            "parameters.parameter.identity",
            position=position,
        )
        _require(
            tuple(parameter.shape) == _PARAMETER_SHAPES[position],
            "parameters.parameter.shape",
            position=position,
        )
        _require(
            parameter.device == PARAMETER_DEVICE,
            "parameters.parameter.device",
            position=position,
        )
        _require(
            parameter.dtype is PARAMETER_DTYPE,
            "parameters.parameter.dtype",
            position=position,
        )
        _require(
            parameter.requires_grad,
            "parameters.parameter.requires_grad",
            position=position,
        )
        _require(
            _all_true(torch.isfinite(parameter)),
            "parameters.parameter.finite",
            position=position,
        )
    return items


def _validated_parameter_items_for_clear_gradients(
    model: SimpleNeuralLanguageModel,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Validate parameters by category for the gradient-clearing contract."""

    if type(model) is not SimpleNeuralLanguageModel:
        raise Phase4TypeError("parameters.model.exact_type")
    items = tuple(model.named_parameters())
    _require(
        tuple(name for name, _ in items) == _PARAMETER_NAMES,
        "parameters.enumeration",
    )
    expected_objects = _expected_parameter_objects(model)
    for position, (_, parameter) in enumerate(items):
        if not isinstance(parameter, nn.Parameter):
            raise Phase4TypeError("parameters.parameter.type", position=position)
    for position, ((_, parameter), expected_object) in enumerate(
        zip(items, expected_objects, strict=True)
    ):
        _require(
            parameter is expected_object,
            "parameters.parameter.identity",
            position=position,
        )
    for position, (_, parameter) in enumerate(items):
        _require(
            tuple(parameter.shape) == _PARAMETER_SHAPES[position],
            "parameters.parameter.shape",
            position=position,
        )
    for position, (_, parameter) in enumerate(items):
        _require(
            parameter.device == PARAMETER_DEVICE,
            "parameters.parameter.device",
            position=position,
        )
    for position, (_, parameter) in enumerate(items):
        _require(
            parameter.dtype is PARAMETER_DTYPE,
            "parameters.parameter.dtype",
            position=position,
        )
    for position, (_, parameter) in enumerate(items):
        _require(
            parameter.requires_grad,
            "parameters.parameter.requires_grad",
            position=position,
        )
    for position, (_, parameter) in enumerate(items):
        _require(
            _all_true(torch.isfinite(parameter)),
            "parameters.parameter.finite",
            position=position,
        )
    return items


def _gradient_for_parameter(parameter: nn.Parameter) -> torch.Tensor | None:
    return parameter.grad


def manual_sgd_step(
    model: SimpleNeuralLanguageModel,
    *,
    learning_rate: float,
) -> None:
    """Atomically preflight and update all four trainable parameters."""

    if type(learning_rate) is not float:
        raise Phase4TypeError("sgd.learning_rate.exact_float")
    _require(
        math.isfinite(learning_rate) and learning_rate > 0.0,
        "sgd.learning_rate.value",
    )
    if type(model) is not SimpleNeuralLanguageModel:
        raise Phase4TypeError("sgd.model.exact_type")
    items = _validated_parameter_items(model)
    gradients = tuple(_gradient_for_parameter(parameter) for _, parameter in items)
    for position, gradient in enumerate(gradients):
        _require(gradient is not None, "sgd.gradient.present", position=position)
    typed_gradients = tuple(gradient for gradient in gradients if gradient is not None)
    for position, gradient in enumerate(typed_gradients):
        if not isinstance(gradient, torch.Tensor):
            raise Phase4TypeError("sgd.gradient.tensor", position=position)
    for position, ((_, parameter), gradient) in enumerate(
        zip(items, typed_gradients, strict=True)
    ):
        _require(
            tuple(gradient.shape) == tuple(parameter.shape),
            "sgd.gradient.shape",
            position=position,
        )
    for position, ((_, parameter), gradient) in enumerate(
        zip(items, typed_gradients, strict=True)
    ):
        _require(
            gradient.dtype is parameter.dtype,
            "sgd.gradient.dtype",
            position=position,
        )
    for position, ((_, parameter), gradient) in enumerate(
        zip(items, typed_gradients, strict=True)
    ):
        _require(
            gradient.device == parameter.device,
            "sgd.gradient.device",
            position=position,
        )
    for position, gradient in enumerate(typed_gradients):
        _require(
            _all_true(torch.isfinite(gradient)),
            "sgd.gradient.finite",
            position=position,
        )

    with torch.no_grad():
        for (_, parameter), gradient in zip(items, typed_gradients, strict=True):
            parameter -= learning_rate * gradient


def clear_gradients(model: SimpleNeuralLanguageModel) -> None:
    """Validate completely, then set every model gradient to ``None``."""

    if type(model) is not SimpleNeuralLanguageModel:
        raise Phase4TypeError("clear_gradients.model.exact_type")
    items = _validated_parameter_items_for_clear_gradients(model)
    gradients = tuple(_gradient_for_parameter(parameter) for _, parameter in items)
    present = tuple(
        (position, parameter, gradient)
        for position, ((_, parameter), gradient) in enumerate(
            zip(items, gradients, strict=True)
        )
        if gradient is not None
    )
    for position, _, gradient in present:
        if not isinstance(gradient, torch.Tensor):
            raise Phase4TypeError("clear_gradients.gradient.tensor", position=position)
    for position, parameter, gradient in present:
        _require(
            tuple(gradient.shape) == tuple(parameter.shape),
            "clear_gradients.gradient.shape",
            position=position,
        )
    for position, parameter, gradient in present:
        _require(
            gradient.dtype is parameter.dtype,
            "clear_gradients.gradient.dtype",
            position=position,
        )
    for position, parameter, gradient in present:
        _require(
            gradient.device == parameter.device,
            "clear_gradients.gradient.device",
            position=position,
        )
    for position, _, gradient in present:
        _require(
            _all_true(torch.isfinite(gradient)),
            "clear_gradients.gradient.finite",
            position=position,
        )

    for _, parameter in items:
        parameter.grad = None


def scale_backward_loss(
    mean_loss: torch.Tensor,
    *,
    example_length: int,
    context_length: int,
) -> torch.Tensor:
    """Scale one mean loss so each target contributes ``1 / 64``."""

    if not isinstance(mean_loss, torch.Tensor):
        raise Phase4TypeError("scale.mean_loss.tensor")
    if type(example_length) is not int:
        raise Phase4TypeError("scale.example_length.exact_int")
    if type(context_length) is not int:
        raise Phase4TypeError("scale.context_length.exact_int")
    _require(mean_loss.dim() == 0, "scale.mean_loss.scalar")
    _require(mean_loss.dtype is torch.float32, "scale.mean_loss.dtype")
    _require(mean_loss.device == PARAMETER_DEVICE, "scale.mean_loss.device")
    _require(torch.isfinite(mean_loss).item(), "scale.mean_loss.finite")
    _require(
        1 <= example_length <= CONTEXT_LENGTH,
        "scale.example_length.value",
        lower_bound=1,
        upper_bound_inclusive=CONTEXT_LENGTH,
    )
    _require(
        context_length == CONTEXT_LENGTH,
        "scale.context_length.value",
        expected=CONTEXT_LENGTH,
    )
    return mean_loss * (example_length / context_length)


def _digest_part(digest: object, content: bytes) -> None:
    assert isinstance(digest, type(hashlib.sha256()))
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


def model_parameter_digest(model: SimpleNeuralLanguageModel) -> str:
    """Hash canonical names, metadata, and raw parameter bytes in order."""

    items = _validated_parameter_items(model)
    digest = hashlib.sha256()
    for name, parameter in items:
        _digest_part(digest, name.encode("utf-8"))
        shape_bytes = b"".join(
            dimension.to_bytes(8, "big", signed=True)
            for dimension in parameter.shape
        )
        _digest_part(digest, shape_bytes)
        _digest_part(digest, str(parameter.dtype).encode("ascii"))
        _digest_part(digest, str(parameter.device).encode("ascii"))
        raw_bytes = bytes(
            parameter.detach()
            .contiguous()
            .view(torch.uint8)
            .reshape(-1)
            .tolist()
        )
        _digest_part(digest, raw_bytes)
    return digest.hexdigest()


def measure_example_loss(
    model: SimpleNeuralLanguageModel,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Measure one example under no-grad while proving model immutability."""

    items = _validated_parameter_items(model)
    _require(
        all(parameter.grad is None for _, parameter in items),
        "measurement.gradients.none_before",
    )
    identities = tuple(id(parameter) for _, parameter in items)
    before_digest = model_parameter_digest(model)
    with torch.no_grad():
        loss = explicit_cross_entropy(model(token_ids), targets)
    after_items = _validated_parameter_items(model)
    _require(
        tuple(id(parameter) for _, parameter in after_items) == identities,
        "measurement.parameter_identity",
    )
    _require(
        model_parameter_digest(model) == before_digest,
        "measurement.parameter_digest",
    )
    _require(
        all(parameter.grad is None for _, parameter in after_items),
        "measurement.gradients.none_after",
    )
    return loss


__all__ = [
    "CONTEXT_LENGTH",
    "EMBEDDING_DIM",
    "EMBEDDING_SEED",
    "MAX_POSITIONS",
    "OUTPUT_HEAD_SEED",
    "PARAMETER_DEVICE",
    "PARAMETER_DTYPE",
    "Phase4ContractError",
    "Phase4GovernanceError",
    "Phase4TypeError",
    "SimpleNeuralLanguageModel",
    "VOCABULARY_SIZE",
    "clear_gradients",
    "explicit_cross_entropy",
    "manual_sgd_step",
    "measure_example_loss",
    "model_parameter_digest",
    "probabilities_from_logits",
    "scale_backward_loss",
]
