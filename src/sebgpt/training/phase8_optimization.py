"""Explicit Phase 8 AdamW updates and no-mutation evaluation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch.optim import AdamW

from sebgpt.model.mini_gpt import MiniGPT
from sebgpt.model.simple_language_model import explicit_cross_entropy
from sebgpt.training.phase8_types import (
    Phase8BatchUpdate,
    Phase8ContractError,
    Phase8Evaluation,
    Phase8LogicalBatch,
    Phase8NumericalError,
    Phase8Progress,
    Phase8TypeError,
    Phase8Window,
    _validate_runtime,
)


LEARNING_RATE = 3e-4
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WEIGHT_DECAY = 0.01
EXPECTED_PARAMETER_COUNT = 63_825
EXPECTED_TENSOR_COUNT = 90


@dataclass(frozen=True)
class _Phase8BatchEvidence:
    batch: Phase8LogicalBatch
    window_identities: tuple[tuple[str, int, str, int, int], ...]
    logits: tuple[torch.Tensor, ...] = field(repr=False)
    losses: tuple[torch.Tensor, ...] = field(repr=False)
    scaled_losses: tuple[torch.Tensor, ...] = field(repr=False)
    per_sequence_accumulated_gradients: tuple[
        tuple[torch.Tensor, ...], ...
    ] = field(repr=False)
    accumulated_gradients: tuple[torch.Tensor, ...] = field(repr=False)
    pre_clip_global_norm: torch.Tensor = field(repr=False)
    clipped_gradients: tuple[torch.Tensor, ...] = field(repr=False)
    parameters_after: tuple[torch.Tensor, ...] = field(repr=False)
    optimizer_state_after: dict[str, object] = field(repr=False)


def _all_finite(value: torch.Tensor) -> bool:
    return bool(torch.all(torch.isfinite(value)).item())


def _model_parameters(model: MiniGPT) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    if type(model) is not MiniGPT:
        raise Phase8TypeError("phase8.type.model")
    model._validate_direct_children()
    model._validate_blocks()
    model._validate_modes()
    model._validate_parameters()
    items = tuple(model.named_parameters())
    identities = tuple(id(parameter) for _, parameter in items)
    if (
        len(items) != EXPECTED_TENSOR_COUNT
        or len(set(identities)) != EXPECTED_TENSOR_COUNT
        or sum(parameter.numel() for _, parameter in items)
        != EXPECTED_PARAMETER_COUNT
    ):
        raise Phase8ContractError("phase8.contract.optimizer", field="parameters")
    for position, (_, parameter) in enumerate(items):
        if not isinstance(parameter, torch.nn.Parameter):
            raise Phase8TypeError(
                "phase8.type.tensor",
                position=position,
            )
        if (
            parameter.device != torch.device("cpu")
            or parameter.dtype is not torch.float32
            or not parameter.requires_grad
            or not _all_finite(parameter)
        ):
            raise Phase8ContractError(
                "phase8.contract.optimizer",
                field="parameter",
                position=position,
            )
    return items


def _validate_optimizer(
    model: MiniGPT,
    optimizer: AdamW,
    *,
    require_state: bool,
    expected_updates: int | None = None,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    items = _model_parameters(model)
    if type(optimizer) is not AdamW:
        raise Phase8TypeError("phase8.type.optimizer")
    if len(optimizer.param_groups) != 1:
        raise Phase8ContractError("phase8.contract.optimizer", field="groups")
    group = optimizer.param_groups[0]
    expected_values = {
        "lr": LEARNING_RATE,
        "betas": BETAS,
        "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
    }
    if tuple(group["params"]) != tuple(parameter for _, parameter in items):
        raise Phase8ContractError(
            "phase8.contract.optimizer",
            field="parameter_order",
        )
    for name, expected in expected_values.items():
        if (
            name not in group
            or type(group[name]) is not type(expected)
            or group[name] != expected
        ):
            raise Phase8ContractError(
                "phase8.contract.optimizer",
                field=name,
            )
    state = optimizer.state_dict()
    expected_group_names = (
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
    )
    if tuple(state) != ("state", "param_groups") or len(state["param_groups"]) != 1:
        raise Phase8ContractError("phase8.contract.optimizer", field="state_dict")
    serialized_group = state["param_groups"][0]
    if tuple(serialized_group) != expected_group_names:
        raise Phase8ContractError(
            "phase8.contract.optimizer",
            field="serialized_group",
        )
    if (
        type(serialized_group["params"]) is not list
        or serialized_group["params"] != list(range(EXPECTED_TENSOR_COUNT))
        or any(type(value) is not int for value in serialized_group["params"])
    ):
        raise Phase8ContractError(
            "phase8.contract.optimizer",
            field="serialized_parameter_order",
        )
    if not require_state:
        if state["state"] != {}:
            raise Phase8ContractError(
                "phase8.contract.optimizer",
                field="unexpected_state",
            )
        return items
    if (
        type(state["state"]) is not dict
        or tuple(state["state"]) != tuple(range(EXPECTED_TENSOR_COUNT))
        or any(type(value) is not int for value in state["state"])
    ):
        raise Phase8ContractError("phase8.contract.optimizer", field="state_keys")
    observed_steps: list[int] = []
    for position, (_, parameter) in enumerate(items):
        parameter_state = state["state"][position]
        if tuple(parameter_state) != ("step", "exp_avg", "exp_avg_sq"):
            raise Phase8ContractError(
                "phase8.contract.optimizer",
                field="parameter_state",
                position=position,
            )
        step = parameter_state["step"]
        if type(step) is not torch.Tensor:
            raise Phase8TypeError(
                "phase8.type.tensor",
                field="step",
                position=position,
            )
        numeric_step = float(step.item())
        valid_step = (
            step.shape == torch.Size([])
            and step.device == torch.device("cpu")
            and step.dtype is torch.float32
            and math.isfinite(numeric_step)
            and numeric_step.is_integer()
        )
        if expected_updates is not None:
            valid_step = valid_step and numeric_step == expected_updates
        observed_steps.append(int(numeric_step))
        for name in ("exp_avg", "exp_avg_sq"):
            value = parameter_state[name]
            valid_step = (
                valid_step
                and type(value) is torch.Tensor
                and value.shape == parameter.shape
                and value.device == torch.device("cpu")
                and value.dtype is torch.float32
                and _all_finite(value)
            )
        if not valid_step:
            raise Phase8ContractError(
                "phase8.contract.optimizer",
                field="state_value",
                position=position,
            )
    if len(set(observed_steps)) != 1:
        raise Phase8ContractError(
            "phase8.contract.optimizer",
            field="step_consistency",
        )
    return items


def create_phase8_optimizer(model: MiniGPT) -> torch.optim.AdamW:
    _validate_runtime()
    items = _model_parameters(model)
    optimizer = AdamW(
        tuple(parameter for _, parameter in items),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    _validate_optimizer(model, optimizer, require_state=False)
    return optimizer


def _validate_windows_and_batch(
    windows: tuple[Phase8Window, ...],
    batch: Phase8LogicalBatch,
) -> tuple[Phase8Window, ...]:
    if type(windows) is not tuple:
        raise Phase8TypeError("phase8.type.argument", field="windows")
    if type(batch) is not Phase8LogicalBatch:
        raise Phase8TypeError("phase8.type.record", field="batch")
    if type(batch.epoch) is not int or not 1 <= batch.epoch <= 10:
        raise Phase8ContractError("phase8.contract.batch", field="epoch")
    if type(batch.batch_index) is not int or batch.batch_index < 0:
        raise Phase8ContractError("phase8.contract.batch", field="batch_index")
    if type(batch.window_indices) is not tuple:
        raise Phase8TypeError("phase8.type.record", field="window_indices")
    if type(batch.target_count) is not int or batch.target_count <= 0:
        raise Phase8ContractError("phase8.contract.batch", field="target_count")
    if not 1 <= len(batch.window_indices) <= 8:
        raise Phase8ContractError("phase8.contract.batch", field="capacity")
    if batch.batch_index >= math.ceil(len(windows) / 8):
        raise Phase8ContractError("phase8.contract.batch", field="batch_index")
    expected_batch_size = min(8, len(windows) - batch.batch_index * 8)
    if len(batch.window_indices) != expected_batch_size:
        raise Phase8ContractError("phase8.contract.batch", field="batch_size")
    if any(type(index) is not int for index in batch.window_indices):
        raise Phase8TypeError("phase8.type.record", field="window_index")
    if len(set(batch.window_indices)) != len(batch.window_indices):
        raise Phase8ContractError("phase8.contract.batch", field="duplicate_index")
    permitted_works = {
        "hamlet": 1,
        "romeo-and-juliet": 2,
        "macbeth": 3,
        "a-midsummer-nights-dream": 4,
        "much-ado-about-nothing": 5,
        "henry-v": 6,
    }
    seen_identities: set[tuple[str, int, str, int, int]] = set()
    for position, window in enumerate(windows):
        if type(window) is not Phase8Window:
            raise Phase8TypeError(
                "phase8.type.record",
                field="window",
                position=position,
            )
        identity = (
            window.split,
            window.manifest_order,
            window.work_id,
            window.start_index,
            window.length,
        )
        if identity in seen_identities:
            raise Phase8ContractError(
                "phase8.contract.window",
                field="duplicate_identity",
                position=position,
            )
        seen_identities.add(identity)
    selected: list[Phase8Window] = []
    for position, index in enumerate(batch.window_indices):
        if type(index) is not int:
            raise Phase8TypeError(
                "phase8.type.record",
                field="window_index",
                position=position,
            )
        if not 0 <= index < len(windows):
            raise Phase8ContractError(
                "phase8.contract.batch",
                field="window_index",
                position=position,
            )
        window = windows[index]
        if (
            type(window.work_id) is not str
            or window.work_id not in permitted_works
            or type(window.manifest_order) is not int
            or window.manifest_order != permitted_works[window.work_id]
            or type(window.split) is not str
            or window.split != "train"
            or type(window.start_index) is not int
            or window.start_index < 0
            or window.start_index % 256 != 0
            or type(window.input_ids) is not tuple
            or type(window.target_ids) is not tuple
            or not 1 <= window.length <= 256
            or len(window.target_ids) != window.length
            or any(type(token) is not int or not 0 <= token < 81 for token in window.input_ids)
            or any(type(token) is not int or not 0 <= token < 81 for token in window.target_ids)
            or window.input_ids[1:] != window.target_ids[:-1]
        ):
            raise Phase8ContractError(
                "phase8.contract.window",
                position=index,
            )
        selected.append(window)
    if sum(window.length for window in selected) != batch.target_count:
        raise Phase8ContractError("phase8.contract.batch", field="target_count")
    return tuple(selected)


def _run_phase8_logical_batch(
    model: MiniGPT,
    optimizer: torch.optim.AdamW,
    windows: tuple[Phase8Window, ...],
    batch: Phase8LogicalBatch,
    *,
    capture_evidence: bool,
    capture_sequence_gradients: bool = False,
) -> tuple[Phase8BatchUpdate, _Phase8BatchEvidence | None]:
    _validate_runtime()
    selected = _validate_windows_and_batch(windows, batch)
    items = _validate_optimizer(
        model,
        optimizer,
        require_state=False if not optimizer.state else True,
    )
    if not model.training:
        raise Phase8ContractError("phase8.contract.mode")
    if any(parameter.grad is not None for _, parameter in items):
        raise Phase8ContractError("phase8.contract.gradient", field="entry")
    optimizer.zero_grad(set_to_none=True)
    target_total = batch.target_count
    weighted_values: list[float] = []
    captured_logits: list[torch.Tensor] = []
    captured_losses: list[torch.Tensor] = []
    captured_scaled_losses: list[torch.Tensor] = []
    captured_sequence_gradients: list[tuple[torch.Tensor, ...]] = []
    for window in selected:
        input_ids = torch.tensor(window.input_ids, dtype=torch.long, device="cpu")
        targets = torch.tensor(window.target_ids, dtype=torch.long, device="cpu")
        logits = model(input_ids)
        mean_loss = explicit_cross_entropy(logits, targets)
        if not bool(torch.isfinite(mean_loss).item()):
            raise Phase8NumericalError("phase8.numerical.loss")
        scaled_loss = mean_loss * (window.length / target_total)
        if not bool(torch.isfinite(scaled_loss).item()):
            raise Phase8NumericalError("phase8.numerical.loss", field="scaled")
        scaled_loss.backward()
        weighted_values.append(float(mean_loss.detach().item()) * window.length)
        if capture_evidence:
            captured_logits.append(logits.detach().clone())
            captured_losses.append(mean_loss.detach().clone())
            captured_scaled_losses.append(scaled_loss.detach().clone())
            if capture_sequence_gradients:
                captured_sequence_gradients.append(
                    tuple(
                        parameter.grad.detach().clone()  # type: ignore[union-attr]
                        for _, parameter in items
                    )
                )
    for position, (_, parameter) in enumerate(items):
        gradient = parameter.grad
        if type(gradient) is not torch.Tensor:
            raise Phase8ContractError(
                "phase8.contract.gradient",
                field="presence",
                position=position,
            )
        if (
            gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or gradient.dtype is not parameter.dtype
        ):
            raise Phase8ContractError(
                "phase8.contract.gradient",
                field="metadata",
                position=position,
            )
        if not _all_finite(gradient):
            raise Phase8NumericalError(
                "phase8.numerical.gradient",
                position=position,
            )
    gradient_identities = tuple(id(parameter.grad) for _, parameter in items)
    accumulated_gradients = (
        tuple(
            parameter.grad.detach().clone()  # type: ignore[union-attr]
            for _, parameter in items
        )
        if capture_evidence
        else ()
    )
    parameters_before_clip = tuple(
        parameter.detach().clone() for _, parameter in items
    )
    optimizer_before_clip = copy.deepcopy(optimizer.state_dict())
    try:
        pre_clip = torch.nn.utils.clip_grad_norm_(
            tuple(parameter for _, parameter in items),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True,
            foreach=False,
        )
    except RuntimeError as error:
        with torch.no_grad():
            for before, (_, parameter) in zip(
                parameters_before_clip,
                items,
                strict=True,
            ):
                parameter.copy_(before)
        if not _objects_equal(optimizer.state_dict(), optimizer_before_clip):
            optimizer.load_state_dict(optimizer_before_clip)
        optimizer.zero_grad(set_to_none=True)
        raise Phase8NumericalError("phase8.numerical.global_norm") from error
    if (
        type(pre_clip) is not torch.Tensor
        or pre_clip.shape != torch.Size([])
        or pre_clip.device != torch.device("cpu")
        or pre_clip.dtype is not torch.float32
        or not bool(torch.isfinite(pre_clip).item())
    ):
        with torch.no_grad():
            for before, (_, parameter) in zip(
                parameters_before_clip,
                items,
                strict=True,
            ):
                parameter.copy_(before)
        if not _objects_equal(optimizer.state_dict(), optimizer_before_clip):
            optimizer.load_state_dict(optimizer_before_clip)
        optimizer.zero_grad(set_to_none=True)
        raise Phase8NumericalError("phase8.numerical.global_norm")
    invalid_post_clip: tuple[str, int] | None = None
    for position, (_, parameter) in enumerate(items):
        gradient = parameter.grad
        if type(gradient) is not torch.Tensor:
            invalid_post_clip = ("presence", position)
            break
        if (
            id(gradient) != gradient_identities[position]
            or gradient.shape != parameter.shape
            or gradient.device != parameter.device
            or gradient.dtype is not parameter.dtype
        ):
            invalid_post_clip = ("metadata", position)
            break
        if not _all_finite(gradient):
            invalid_post_clip = ("finiteness", position)
            break
    if invalid_post_clip is not None:
        with torch.no_grad():
            for before, (_, parameter) in zip(
                parameters_before_clip,
                items,
                strict=True,
            ):
                parameter.copy_(before)
        if not _objects_equal(optimizer.state_dict(), optimizer_before_clip):
            optimizer.load_state_dict(optimizer_before_clip)
        optimizer.zero_grad(set_to_none=True)
        field, position = invalid_post_clip
        if field == "finiteness":
            raise Phase8NumericalError(
                "phase8.numerical.gradient",
                field="post_clip",
                position=position,
            )
        raise Phase8ContractError(
            "phase8.contract.gradient",
            field=f"post_clip_{field}",
            position=position,
        )
    clipped_gradients = (
        tuple(
            parameter.grad.detach().clone()  # type: ignore[union-attr]
            for _, parameter in items
        )
        if capture_evidence
        else ()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for _, parameter in items):
        raise Phase8ContractError("phase8.contract.gradient", field="exit")
    _validate_optimizer(model, optimizer, require_state=True)
    update = Phase8BatchUpdate(
        batch_loss=math.fsum(weighted_values) / target_total,
        target_count=target_total,
        sequence_count=len(selected),
        pre_clip_global_norm=float(pre_clip.item()),
    )
    evidence = None
    if capture_evidence:
        evidence = _Phase8BatchEvidence(
            batch=batch,
            window_identities=tuple(
                (
                    window.split,
                    window.manifest_order,
                    window.work_id,
                    window.start_index,
                    window.length,
                )
                for window in selected
            ),
            logits=tuple(captured_logits),
            losses=tuple(captured_losses),
            scaled_losses=tuple(captured_scaled_losses),
            per_sequence_accumulated_gradients=tuple(
                captured_sequence_gradients
            ),
            accumulated_gradients=accumulated_gradients,
            pre_clip_global_norm=pre_clip.detach().clone(),
            clipped_gradients=clipped_gradients,
            parameters_after=tuple(
                parameter.detach().clone() for _, parameter in items
            ),
            optimizer_state_after=copy.deepcopy(optimizer.state_dict()),
        )
    return update, evidence


def run_phase8_logical_batch(
    model: MiniGPT,
    optimizer: torch.optim.AdamW,
    windows: tuple[Phase8Window, ...],
    batch: Phase8LogicalBatch,
) -> Phase8BatchUpdate:
    update, _ = _run_phase8_logical_batch(
        model,
        optimizer,
        windows,
        batch,
        capture_evidence=False,
    )
    return update


def _dropout_states(model: MiniGPT) -> tuple[torch.Tensor, ...]:
    states: list[torch.Tensor] = []
    for block in model.blocks:
        states.append(block.attention_dropout._generator.get_state().clone())
        states.append(block.feed_forward_dropout._generator.get_state().clone())
    return tuple(states)


def _objects_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return tuple(left) == tuple(right) and all(
            _objects_equal(left[key], right[key]) for key in left
        )
    if type(left) is list and type(right) is list:
        return len(left) == len(right) and all(
            _objects_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is tuple and type(right) is tuple:
        return len(left) == len(right) and all(
            _objects_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def evaluate_phase8_split(
    model: MiniGPT,
    optimizer: torch.optim.AdamW,
    windows: tuple[Phase8Window, ...],
    order_generator: torch.Generator,
    progress: Phase8Progress,
    *,
    split: str,
) -> Phase8Evaluation:
    _validate_runtime()
    items = _validate_optimizer(model, optimizer, require_state=bool(optimizer.state))
    if type(windows) is not tuple:
        raise Phase8TypeError("phase8.type.argument", field="windows")
    if type(order_generator) is not torch.Generator:
        raise Phase8TypeError("phase8.type.generator")
    if type(progress) is not Phase8Progress:
        raise Phase8TypeError("phase8.type.record", field="progress")
    progress_values = (
        progress.completed_epochs,
        progress.next_epoch,
        progress.next_example_offset,
        progress.optimizer_updates,
        progress.examples_processed,
        progress.targets_processed,
    )
    if any(type(value) is not int for value in progress_values):
        raise Phase8ContractError("phase8.contract.progress", field="type")
    if (
        not 0 <= progress.completed_epochs <= 10
        or progress.next_epoch != progress.completed_epochs + 1
        or progress.next_example_offset != 0
        or any(value < 0 for value in progress_values[3:])
        or (
            progress.completed_epochs == 0
            and progress_values[3:] != (0, 0, 0)
        )
        or (
            progress.completed_epochs > 0
            and any(value <= 0 for value in progress_values[3:])
        )
    ):
        raise Phase8ContractError("phase8.contract.progress")
    if type(split) is not str:
        raise Phase8TypeError("phase8.type.argument", field="split")
    if split not in ("train", "validation"):
        raise Phase8ContractError("phase8.contract.metric", field="split")
    if not model.training or any(parameter.grad is not None for _, parameter in items):
        raise Phase8ContractError("phase8.contract.mode")
    previous_key: tuple[int, int] | None = None
    expected_works = (
        {
            "hamlet": 1,
            "romeo-and-juliet": 2,
            "macbeth": 3,
            "a-midsummer-nights-dream": 4,
            "much-ado-about-nothing": 5,
            "henry-v": 6,
        }
        if split == "train"
        else {"the-tempest": 7}
    )
    for position, window in enumerate(windows):
        if type(window) is not Phase8Window:
            raise Phase8TypeError("phase8.type.record", field="window", position=position)
        key = (window.manifest_order, window.start_index)
        if (
            type(window.work_id) is not str
            or window.work_id not in expected_works
            or type(window.manifest_order) is not int
            or window.manifest_order != expected_works[window.work_id]
            or type(window.split) is not str
            or window.split != split
            or type(window.start_index) is not int
            or window.start_index < 0
            or window.start_index % 256 != 0
            or type(window.input_ids) is not tuple
            or type(window.target_ids) is not tuple
            or not 1 <= window.length <= 256
            or len(window.target_ids) != window.length
            or any(
                type(token) is not int or not 0 <= token < 81
                for token in window.input_ids
            )
            or any(
                type(token) is not int or not 0 <= token < 81
                for token in window.target_ids
            )
            or window.input_ids[1:] != window.target_ids[:-1]
            or (previous_key is not None and key <= previous_key)
        ):
            raise Phase8ContractError("phase8.contract.metric", field="window_order")
        previous_key = key
    if not windows:
        raise Phase8ContractError("phase8.contract.metric", field="windows")

    parameter_bytes = tuple(
        bytes(parameter.detach().contiguous().view(torch.uint8).reshape(-1).tolist())
        for _, parameter in items
    )
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    dropout_before = _dropout_states(model)
    order_before = order_generator.get_state().clone()
    global_before = torch.get_rng_state().clone()
    modes_before = tuple((name, module.training) for name, module in model.named_modules())
    weighted: list[float] = []
    target_count = 0
    try:
        model.eval()
        with torch.no_grad():
            for window in windows:
                input_ids = torch.tensor(window.input_ids, dtype=torch.long, device="cpu")
                targets = torch.tensor(window.target_ids, dtype=torch.long, device="cpu")
                mean_loss = explicit_cross_entropy(model(input_ids), targets)
                if not bool(torch.isfinite(mean_loss).item()):
                    raise Phase8NumericalError("phase8.numerical.metric")
                weighted.append(float(mean_loss.item()) * window.length)
                target_count += window.length
    finally:
        model.train()
    if tuple((name, module.training) for name, module in model.named_modules()) != modes_before:
        raise Phase8ContractError("phase8.contract.mode", field="restoration")
    after_bytes = tuple(
        bytes(parameter.detach().contiguous().view(torch.uint8).reshape(-1).tolist())
        for _, parameter in items
    )
    if (
        after_bytes != parameter_bytes
        or not _objects_equal(optimizer.state_dict(), optimizer_before)
        or any(
            not torch.equal(left, right)
            for left, right in zip(_dropout_states(model), dropout_before, strict=True)
        )
        or not torch.equal(order_generator.get_state(), order_before)
        or not torch.equal(torch.get_rng_state(), global_before)
    ):
        raise Phase8ContractError("phase8.contract.metric", field="mutation")
    loss = math.fsum(weighted) / target_count
    if not math.isfinite(loss):
        raise Phase8NumericalError("phase8.numerical.metric")
    return Phase8Evaluation(
        split=split,
        loss=loss,
        target_count=target_count,
        window_count=len(windows),
    )


__all__ = [
    "create_phase8_optimizer",
    "evaluate_phase8_split",
    "run_phase8_logical_batch",
]
