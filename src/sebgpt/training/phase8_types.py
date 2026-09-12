"""Exact public records and exceptions for Phase 8 training."""

from __future__ import annotations

import json
import hashlib
import platform
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
from torch.optim import AdamW

from sebgpt.model.mini_gpt import MiniGPT


class _Phase8ErrorMixin:
    def _initialize_error(
        self,
        invariant: str,
        safe_facts: Mapping[str, object],
    ) -> str:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        return json.dumps(details, ensure_ascii=True, sort_keys=True)

    @property
    def details(self) -> Mapping[str, object]:
        return self._details


class Phase8TypeError(_Phase8ErrorMixin, TypeError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize_error(invariant, safe_facts))


class Phase8ContractError(_Phase8ErrorMixin, ValueError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize_error(invariant, safe_facts))


class Phase8GovernanceError(_Phase8ErrorMixin, ValueError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize_error(invariant, safe_facts))


class Phase8NumericalError(_Phase8ErrorMixin, ArithmeticError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize_error(invariant, safe_facts))


class Phase8CheckpointError(_Phase8ErrorMixin, ValueError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize_error(invariant, safe_facts))


class Phase8PublicationError(_Phase8ErrorMixin, RuntimeError):
    def __init__(self, invariant: str, **safe_facts: object) -> None:
        super().__init__(self._initialize_error(invariant, safe_facts))


@dataclass(frozen=True)
class Phase8Window:
    work_id: str
    manifest_order: int
    split: str
    start_index: int
    input_ids: tuple[int, ...] = field(repr=False)
    target_ids: tuple[int, ...] = field(repr=False)

    @property
    def length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class Phase8LogicalBatch:
    epoch: int
    batch_index: int
    window_indices: tuple[int, ...]
    target_count: int


@dataclass(frozen=True)
class Phase8BatchUpdate:
    batch_loss: float
    target_count: int
    sequence_count: int
    pre_clip_global_norm: float


@dataclass(frozen=True)
class Phase8Evaluation:
    split: str
    loss: float
    target_count: int
    window_count: int


@dataclass(frozen=True)
class Phase8Progress:
    completed_epochs: int
    next_epoch: int
    next_example_offset: int
    optimizer_updates: int
    examples_processed: int
    targets_processed: int


@dataclass(frozen=True)
class Phase8CheckpointReference:
    run_id: str
    role: str
    logical_id: str
    epoch: int
    validation_loss_hex: str
    sha256: str
    relative_path: str


@dataclass(frozen=True)
class Phase8RunResult:
    run_id: str
    status: str
    stopping_reason: str
    progress: Phase8Progress
    initialized_training: Phase8Evaluation
    initialized_validation: Phase8Evaluation
    epoch_training: tuple[Phase8Evaluation, ...]
    epoch_validation: tuple[Phase8Evaluation, ...]
    latest_checkpoint: Phase8CheckpointReference
    best_checkpoint: Phase8CheckpointReference
    exact_resume_passed: bool


@dataclass(frozen=True)
class Phase8RuntimeIdentity:
    schema_version: int
    python_version: str
    python_implementation: str
    torch_version: str
    operating_system: str
    operating_system_release: str
    machine: str
    processor: str
    device: str
    parameter_dtype: str
    token_dtype: str
    torch_intra_op_threads: int
    torch_inter_op_threads: int
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    float32_matmul_precision: str
    default_dtype: str
    default_device: str
    grad_mode_enabled: bool
    cpu_autocast_enabled: bool
    inference_mode_enabled: bool
    mkldnn_enabled: bool
    torch_build_config_sha256: str
    torch_parallel_info_sha256: str
    requirements_lock_sha256: str


@dataclass(frozen=True, init=False)
class Phase8Configuration:
    schema_version: int
    runtime: Phase8RuntimeIdentity
    context_length: int
    stride: int
    retain_unpadded_tail: bool
    logical_batch_capacity: int
    allow_final_partial_batch: bool
    maximum_epochs: int
    order_seed: int
    order_algorithm: str
    optimizer_name: str
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    amsgrad: bool
    maximize: bool
    foreach: bool
    capturable: bool
    differentiable: bool
    fused: bool
    scheduler_name: None
    warmup_steps: int
    gradient_clip_norm_type: float
    gradient_clip_max_norm: float
    gradient_clip_epsilon: float
    evaluate_initialized_state: bool
    evaluation_interval_epochs: int
    evaluation_split_order: tuple[str, str]
    best_comparison: str
    validation_early_stopping: bool
    parameter_device: str
    parameter_dtype: str
    token_dtype: str
    checkpoint_schema_version: int
    catalog_schema_version: int
    maximum_catalog_bytes: int
    maximum_checkpoint_object_bytes: int

    def __new__(cls, *args: object, **kwargs: object) -> Phase8Configuration:
        raise Phase8ContractError("phase8.contract.configuration")


@dataclass(frozen=True, init=False)
class Phase8MetricState:
    initialized_training: Phase8Evaluation
    initialized_validation: Phase8Evaluation
    epoch_training: tuple[Phase8Evaluation, ...]
    epoch_validation: tuple[Phase8Evaluation, ...]
    current_training: Phase8Evaluation
    current_validation: Phase8Evaluation
    best_validation_loss: float
    best_validation_epoch: int
    best_logical_id: str

    def __new__(cls, *args: object, **kwargs: object) -> Phase8MetricState:
        raise Phase8ContractError("phase8.contract.metric")


@dataclass(frozen=True, init=False)
class Phase8TrainingState:
    run_id: str
    code_commit: str
    configuration: Phase8Configuration
    model: MiniGPT = field(repr=False)
    optimizer: AdamW = field(repr=False)
    order_generator: torch.Generator = field(repr=False)
    progress: Phase8Progress
    metrics: Phase8MetricState
    global_cpu_rng_state: torch.Tensor = field(repr=False)
    resumed_from_checkpoint_sha256: str | None
    resume_checkpoint_sha256s: tuple[str, ...]

    def __new__(cls, *args: object, **kwargs: object) -> Phase8TrainingState:
        raise Phase8ContractError("phase8.contract.lifecycle")


def _factory_record(record_type: type[object], **values: object) -> object:
    expected = tuple(item.name for item in fields(record_type))
    if tuple(values) != expected:
        raise Phase8ContractError("phase8.contract.record")
    result = object.__new__(record_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _make_configuration(**values: object) -> Phase8Configuration:
    return _factory_record(Phase8Configuration, **values)  # type: ignore[return-value]


def _make_metric_state(**values: object) -> Phase8MetricState:
    return _factory_record(Phase8MetricState, **values)  # type: ignore[return-value]


def _make_training_state(**values: object) -> Phase8TrainingState:
    return _factory_record(Phase8TrainingState, **values)  # type: ignore[return-value]


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime_identity() -> Phase8RuntimeIdentity:
    lock_path = _REPOSITORY_ROOT / "requirements.lock"
    try:
        lock_bytes = lock_path.read_bytes()
    except OSError:
        raise Phase8ContractError(
            "phase8.contract.runtime",
            field="requirements_lock",
        ) from None
    return Phase8RuntimeIdentity(
        schema_version=1,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        torch_version=str(torch.__version__),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        device="cpu",
        parameter_dtype=str(torch.float32),
        token_dtype="torch.long",
        torch_intra_op_threads=torch.get_num_threads(),
        torch_inter_op_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        default_dtype=str(torch.get_default_dtype()),
        default_device=str(torch.get_default_device()),
        grad_mode_enabled=torch.is_grad_enabled(),
        cpu_autocast_enabled=torch.is_autocast_enabled("cpu"),
        inference_mode_enabled=torch.is_inference_mode_enabled(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        torch_build_config_sha256=_sha256_text(torch.__config__.show()),
        torch_parallel_info_sha256=_sha256_text(torch.__config__.parallel_info()),
        requirements_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
    )


def _validate_runtime_identity(value: Phase8RuntimeIdentity) -> None:
    if type(value) is not Phase8RuntimeIdentity:
        raise Phase8TypeError("phase8.type.record", field="runtime")
    exact_types = {
        "schema_version": int,
        "python_version": str,
        "python_implementation": str,
        "torch_version": str,
        "operating_system": str,
        "operating_system_release": str,
        "machine": str,
        "processor": str,
        "device": str,
        "parameter_dtype": str,
        "token_dtype": str,
        "torch_intra_op_threads": int,
        "torch_inter_op_threads": int,
        "deterministic_algorithms": bool,
        "deterministic_warn_only": bool,
        "float32_matmul_precision": str,
        "default_dtype": str,
        "default_device": str,
        "grad_mode_enabled": bool,
        "cpu_autocast_enabled": bool,
        "inference_mode_enabled": bool,
        "mkldnn_enabled": bool,
        "torch_build_config_sha256": str,
        "torch_parallel_info_sha256": str,
        "requirements_lock_sha256": str,
    }
    for name, expected_type in exact_types.items():
        if type(getattr(value, name)) is not expected_type:
            raise Phase8ContractError(
                "phase8.contract.runtime",
                field=name,
            )
    expected = {
        "schema_version": 1,
        "python_version": "3.14.4",
        "python_implementation": "CPython",
        "torch_version": "2.14.0",
        "operating_system": "Darwin",
        "machine": "arm64",
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
    }
    for name, required in expected.items():
        if getattr(value, name) != required:
            raise Phase8ContractError(
                "phase8.contract.runtime",
                field=name,
            )
    for name in (
        "torch_build_config_sha256",
        "torch_parallel_info_sha256",
        "requirements_lock_sha256",
    ):
        digest = getattr(value, name)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise Phase8ContractError(
                "phase8.contract.runtime",
                field=name,
            )
    live = _runtime_identity()
    for item in fields(Phase8RuntimeIdentity):
        if getattr(value, item.name) != getattr(live, item.name):
            raise Phase8ContractError(
                "phase8.contract.runtime",
                field=item.name,
            )


def _validate_runtime() -> Phase8RuntimeIdentity:
    value = _runtime_identity()
    _validate_runtime_identity(value)
    return value


def _configure_runtime() -> Phase8RuntimeIdentity:
    try:
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cpu")
        torch.backends.mkldnn.enabled = False
        torch.set_grad_enabled(True)
    except BaseException as error:
        raise Phase8ContractError(
            "phase8.contract.runtime",
            field="configuration",
        ) from error
    return _validate_runtime()


def _fixed_configuration(runtime: Phase8RuntimeIdentity) -> Phase8Configuration:
    _validate_runtime_identity(runtime)
    return _make_configuration(
        schema_version=1,
        runtime=runtime,
        context_length=256,
        stride=256,
        retain_unpadded_tail=True,
        logical_batch_capacity=8,
        allow_final_partial_batch=True,
        maximum_epochs=10,
        order_seed=8001,
        order_algorithm="torch_randperm_local_cpu_generator",
        optimizer_name="AdamW",
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
        scheduler_name=None,
        warmup_steps=0,
        gradient_clip_norm_type=2.0,
        gradient_clip_max_norm=1.0,
        gradient_clip_epsilon=1e-6,
        evaluate_initialized_state=True,
        evaluation_interval_epochs=1,
        evaluation_split_order=("train", "validation"),
        best_comparison="strict_lower",
        validation_early_stopping=False,
        parameter_device="cpu",
        parameter_dtype="torch.float32",
        token_dtype="torch.long",
        checkpoint_schema_version=1,
        catalog_schema_version=1,
        maximum_catalog_bytes=16_384,
        maximum_checkpoint_object_bytes=67_108_864,
    )


__all__ = [
    "Phase8BatchUpdate",
    "Phase8CheckpointError",
    "Phase8CheckpointReference",
    "Phase8Configuration",
    "Phase8ContractError",
    "Phase8Evaluation",
    "Phase8GovernanceError",
    "Phase8LogicalBatch",
    "Phase8MetricState",
    "Phase8NumericalError",
    "Phase8Progress",
    "Phase8PublicationError",
    "Phase8RunResult",
    "Phase8RuntimeIdentity",
    "Phase8TrainingState",
    "Phase8TypeError",
    "Phase8Window",
]
