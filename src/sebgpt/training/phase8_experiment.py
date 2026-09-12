"""Fixed Phase 8 runtime configuration and training lifecycle."""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path

import torch

from sebgpt.model.mini_gpt import (
    BLOCK_PARAMETER_INITIALIZATION_SEEDS,
    DROPOUT_PROBABILITY,
    EMBEDDING_INITIALIZATION_SEED,
    HEAD_WIDTH,
    HIDDEN_WIDTH,
    LAYER_NORM_EPSILON,
    MAX_SEQUENCE_LENGTH,
    MODEL_WIDTH,
    NUMBER_OF_BLOCKS,
    NUMBER_OF_HEADS,
    OUTPUT_HEAD_INITIALIZATION_SEED,
    PARAMETER_DEVICE,
    PARAMETER_DTYPE,
    MiniGPT,
)
from sebgpt.model.phase4_experiment import Phase4PermittedCorpus
from sebgpt.tokenization.vocabulary_artifact import VocabularyBinding
from sebgpt.training.phase8_checkpoint import (
    _authority_mapping,
    _catalog_references,
    _configuration_mapping,
    _load_phase8_checkpoint,
    _payload,
    _publish_payload,
    _validate_live_repository,
    load_phase8_checkpoint,
    save_phase8_checkpoint,
)
from sebgpt.training.phase8_data import (
    build_phase8_windows,
    create_phase8_epoch_order,
    iter_phase8_logical_batches,
)
from sebgpt.training.phase8_optimization import (
    _Phase8BatchEvidence,
    _run_phase8_logical_batch,
    _objects_equal,
    create_phase8_optimizer,
    evaluate_phase8_split,
    run_phase8_logical_batch,
)
from sebgpt.training.phase8_types import (
    Phase8Configuration,
    Phase8CheckpointReference,
    Phase8ContractError,
    Phase8GovernanceError,
    Phase8MetricState,
    Phase8Progress,
    Phase8RunResult,
    Phase8RuntimeIdentity,
    Phase8TrainingState,
    Phase8TypeError,
    _configure_runtime,
    _fixed_configuration,
    _make_metric_state,
    _make_training_state,
    _validate_runtime,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PLANNED_ENTRY = re.compile(
    r"^### (EXP-[0-9]{8}-[0-9]{2}) — Phase 8[^\n]*\n(?P<body>.*?)(?=^### |\Z)",
    re.MULTILINE | re.DOTALL,
)
_PLANNED_FIELD = re.compile(r"^- \*\*(?P<label>[^*]+):\*\* (?P<value>.+)$")
_PLANNED_RECORD_FIELDS = (
    "Experiment ID",
    "Entry kind",
    "Recorded at UTC",
    "Status",
    "Question",
    "Authorization",
    "Code commit",
    "Phase 7 authority",
    "Phase 8 contract authority",
    "Runtime identity",
    "Dataset authority",
    "Tokenizer authority",
    "Model configuration",
    "Training configuration",
    "Checkpoint configuration",
    "Feasibility evidence",
    "Success predicate",
    "Sealed-test policy",
    "Generation policy",
    "Next gate",
)
_RESULT_RECORD_FIELDS = (
    "Experiment ID",
    "Entry kind",
    "Recorded at UTC",
    "Started at UTC",
    "Finished at UTC",
    "Status",
    "Question",
    "Authorization and predecessor",
    "Code commit",
    "Phase 7 authority",
    "Phase 8 contract authority",
    "Runtime identity",
    "Dataset authority",
    "Tokenizer authority",
    "Model configuration",
    "Training configuration",
    "Window counts",
    "Target counts",
    "Update counts",
    "Initialized losses",
    "Epoch training losses",
    "Epoch validation losses",
    "Gradient-norm evidence",
    "Latest checkpoint",
    "Best-validation checkpoint",
    "Checkpoint publication evidence",
    "Resume audit",
    "Resume lineage",
    "Training-loss improvement predicate",
    "Training-loss improvement result",
    "Fitting evidence",
    "Sealed-test access",
    "Generation and samples",
    "Stopping reason",
    "Observations",
    "Conclusions",
    "Limitations",
    "Next gate",
)


def _float_record(value: float) -> dict[str, str]:
    return {"decimal": repr(value), "hex": value.hex()}


def configure_phase8_runtime() -> Phase8RuntimeIdentity:
    return _configure_runtime()


def validate_phase8_runtime() -> Phase8RuntimeIdentity:
    return _validate_runtime()


def load_phase8_configuration() -> Phase8Configuration:
    return _fixed_configuration(_validate_runtime())


def _record_value(value: str) -> object:
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        value = value[1:-1]
    if value.startswith("{") or value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise Phase8ContractError(
                "phase8.contract.lifecycle",
                field="planned_record_json",
            ) from None
        canonical = json.dumps(
            parsed,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if value != canonical:
            raise Phase8ContractError(
                "phase8.contract.lifecycle",
                field="planned_record_canonical",
            )
        return parsed
    if not value or not value.isascii():
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="planned_record_value",
        )
    return value


def _planned_run_record(
    repository_root: Path,
    *,
    code_commit: str | None = None,
    configuration: Phase8Configuration | None = None,
) -> tuple[str, dict[str, object]]:
    try:
        content = (repository_root / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="experiment_log",
        ) from None
    matches = tuple(_PLANNED_ENTRY.finditer(content))
    if len(matches) != 1:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="planned_run_id",
        )
    match = matches[0]
    fields_found: list[tuple[str, object]] = []
    for line in match.group("body").splitlines():
        if not line.startswith("- **"):
            continue
        field_match = _PLANNED_FIELD.fullmatch(line)
        if field_match is None:
            raise Phase8ContractError(
                "phase8.contract.lifecycle",
                field="planned_record_syntax",
            )
        fields_found.append(
            (
                field_match.group("label"),
                _record_value(field_match.group("value")),
            )
        )
    if tuple(label for label, _ in fields_found) != _PLANNED_RECORD_FIELDS:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="planned_record_fields",
        )
    record = dict(fields_found)
    run_id = match.group(1)
    if (
        record["Experiment ID"] != run_id
        or record["Entry kind"] != "planned"
        or record["Status"] != "planned"
        or type(record["Recorded at UTC"]) is not str
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            record["Recorded at UTC"],
        )
        is None
        or record["Success predicate"]
        != "min(training_loss at completed epochs 1..10) < initialized training_loss"
    ):
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="planned_record_identity",
        )
    if code_commit is not None and record["Code commit"] != code_commit:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="planned_record_code_commit",
        )
    if configuration is not None:
        authority = _authority_mapping(configuration, repository_root)
        exact_values = {
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
            "Training configuration": _configuration_mapping(configuration),
            "Checkpoint configuration": {
                "checkpoint_schema_version": configuration.checkpoint_schema_version,
                "catalog_schema_version": configuration.catalog_schema_version,
                "maximum_catalog_bytes": configuration.maximum_catalog_bytes,
                "maximum_checkpoint_object_bytes": configuration.maximum_checkpoint_object_bytes,
            },
            "Sealed-test policy": "none",
            "Generation policy": "none",
        }
        for label, expected in exact_values.items():
            comparable_expected = (
                json.loads(
                    json.dumps(
                        expected,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                if type(expected) in (dict, tuple, list)
                else expected
            )
            if record[label] != comparable_expected:
                raise Phase8ContractError(
                    "phase8.contract.lifecycle",
                    field=f"planned_record_{label.lower().replace(' ', '_')}",
                )
    return run_id, record


def _planned_run_id(repository_root: Path) -> str:
    return _planned_run_record(repository_root)[0]


def _new_model(vocabulary: VocabularyBinding) -> MiniGPT:
    return MiniGPT(
        vocabulary,
        model_width=MODEL_WIDTH,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        number_of_blocks=NUMBER_OF_BLOCKS,
        number_of_heads=NUMBER_OF_HEADS,
        head_width=HEAD_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        layer_norm_epsilon=LAYER_NORM_EPSILON,
        dropout_probability=DROPOUT_PROBABILITY,
        embedding_seed=EMBEDDING_INITIALIZATION_SEED,
        block_parameter_seeds=BLOCK_PARAMETER_INITIALIZATION_SEEDS,
        output_head_seed=OUTPUT_HEAD_INITIALIZATION_SEED,
        device=PARAMETER_DEVICE,
        dtype=PARAMETER_DTYPE,
    )


def _tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(
        bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())
    ).hexdigest()


def _batch_evidence_record(evidence: _Phase8BatchEvidence) -> dict[str, object]:
    return {
        "epoch": evidence.batch.epoch,
        "batch_index": evidence.batch.batch_index,
        "window_identities": evidence.window_identities,
        "target_count": evidence.batch.target_count,
        "logit_sha256s": tuple(_tensor_digest(value) for value in evidence.logits),
        "per_sequence_loss_hex": tuple(
            float(value.item()).hex() for value in evidence.losses
        ),
        "scaled_loss_hex": tuple(
            float(value.item()).hex() for value in evidence.scaled_losses
        ),
        "per_sequence_accumulated_gradient_sha256s": tuple(
            tuple(_tensor_digest(value) for value in gradients)
            for gradients in evidence.per_sequence_accumulated_gradients
        ),
        "pre_clip_global_norm_hex": float(
            evidence.pre_clip_global_norm.item()
        ).hex(),
        "accumulated_gradient_sha256s": tuple(
            _tensor_digest(value) for value in evidence.accumulated_gradients
        ),
        "clipped_gradient_sha256s": tuple(
            _tensor_digest(value) for value in evidence.clipped_gradients
        ),
        "parameter_sha256s_after": tuple(
            _tensor_digest(value) for value in evidence.parameters_after
        ),
        "optimizer_update": int(
            evidence.optimizer_state_after["state"][0]["step"].item()
        ),
    }


def _batch_evidence_equal(
    left: _Phase8BatchEvidence,
    right: _Phase8BatchEvidence,
) -> bool:
    return (
        left.batch == right.batch
        and left.window_identities == right.window_identities
        and _objects_equal(left.logits, right.logits)
        and _objects_equal(left.losses, right.losses)
        and _objects_equal(left.scaled_losses, right.scaled_losses)
        and _objects_equal(
            left.per_sequence_accumulated_gradients,
            right.per_sequence_accumulated_gradients,
        )
        and _objects_equal(left.accumulated_gradients, right.accumulated_gradients)
        and torch.equal(left.pre_clip_global_norm, right.pre_clip_global_norm)
        and _objects_equal(left.clipped_gradients, right.clipped_gradients)
        and _objects_equal(left.parameters_after, right.parameters_after)
        and _objects_equal(
            left.optimizer_state_after,
            right.optimizer_state_after,
        )
    )


def _branch_runtime_equal(
    left: Phase8TrainingState,
    right: Phase8TrainingState,
) -> bool:
    return (
        torch.equal(
            left.order_generator.get_state(),
            right.order_generator.get_state(),
        )
        and all(
            torch.equal(
                left_block.attention_dropout._generator.get_state(),
                right_block.attention_dropout._generator.get_state(),
            )
            and torch.equal(
                left_block.feed_forward_dropout._generator.get_state(),
                right_block.feed_forward_dropout._generator.get_state(),
            )
            for left_block, right_block in zip(
                left.model.blocks,
                right.model.blocks,
                strict=True,
            )
        )
        and torch.equal(left.global_cpu_rng_state, right.global_cpu_rng_state)
        and all(module.training for _, module in left.model.named_modules())
        and all(module.training for _, module in right.model.named_modules())
        and all(parameter.grad is None for parameter in left.model.parameters())
        and all(parameter.grad is None for parameter in right.model.parameters())
    )


def _write_evidence_artifact(
    repository_root: Path,
    run_id: str,
    records: tuple[dict[str, object], ...],
    latest: Phase8CheckpointReference,
    best: Phase8CheckpointReference,
) -> dict[str, object]:
    artifact_root = repository_root / "experiments" / run_id / "artifacts"
    artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    content = (
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "batch_and_resume_evidence": records,
                "latest_checkpoint": {
                    name: getattr(latest, name)
                    for name in latest.__dataclass_fields__
                },
                "best_validation_checkpoint": {
                    name: getattr(best, name)
                    for name in best.__dataclass_fields__
                },
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path = artifact_root / "phase8-training-evidence.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        written = os.write(fd, content)
        if written != len(content):
            raise Phase8ContractError(
                "phase8.contract.lifecycle",
                field="evidence_write",
            )
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "relative_path": path.relative_to(repository_root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _train_epoch(
    model: MiniGPT,
    optimizer: torch.optim.AdamW,
    order_generator: torch.Generator,
    prior_progress: Phase8Progress,
    windows: tuple[object, ...],
    epoch: int,
    *,
    evidence_sink: list[dict[str, object]] | None = None,
) -> Phase8Progress:
    order = create_phase8_epoch_order(
        len(windows),
        epoch=epoch,
        generator=order_generator,
    )
    updates = 0
    examples = 0
    targets = 0
    for batch in iter_phase8_logical_batches(windows, order, epoch=epoch):  # type: ignore[arg-type]
        if evidence_sink is None:
            run_phase8_logical_batch(
                model,
                optimizer,
                windows,  # type: ignore[arg-type]
                batch,
            )
        else:
            _, evidence = _run_phase8_logical_batch(
                model,
                optimizer,
                windows,  # type: ignore[arg-type]
                batch,
                capture_evidence=True,
            )
            if evidence is None:
                raise Phase8ContractError(
                    "phase8.contract.lifecycle",
                    field="batch_evidence",
                )
            evidence_sink.append(_batch_evidence_record(evidence))
        updates += 1
        examples += len(batch.window_indices)
        targets += batch.target_count
    return Phase8Progress(
        completed_epochs=epoch,
        next_epoch=epoch + 1,
        next_example_offset=0,
        optimizer_updates=prior_progress.optimizer_updates + updates,
        examples_processed=prior_progress.examples_processed + examples,
        targets_processed=prior_progress.targets_processed + targets,
    )


def _metrics_after_epoch(
    state: Phase8TrainingState,
    training_windows: tuple[object, ...],
    validation_windows: tuple[object, ...],
) -> Phase8MetricState:
    training = evaluate_phase8_split(
        state.model,
        state.optimizer,
        training_windows,  # type: ignore[arg-type]
        state.order_generator,
        state.progress,
        split="train",
    )
    validation = evaluate_phase8_split(
        state.model,
        state.optimizer,
        validation_windows,  # type: ignore[arg-type]
        state.order_generator,
        state.progress,
        split="validation",
    )
    prior = state.metrics
    epoch_training = (*prior.epoch_training, training)
    epoch_validation = (*prior.epoch_validation, validation)
    if validation.loss < prior.best_validation_loss:
        best_loss = validation.loss
        best_epoch = state.progress.completed_epochs
        best_logical_id = f"epoch-{best_epoch:04d}"
    else:
        best_loss = prior.best_validation_loss
        best_epoch = prior.best_validation_epoch
        best_logical_id = prior.best_logical_id
    return _make_metric_state(
        initialized_training=prior.initialized_training,
        initialized_validation=prior.initialized_validation,
        epoch_training=epoch_training,
        epoch_validation=epoch_validation,
        current_training=training,
        current_validation=validation,
        best_validation_loss=best_loss,
        best_validation_epoch=best_epoch,
        best_logical_id=best_logical_id,
    )


def _state_with(
    state: Phase8TrainingState,
    *,
    progress: Phase8Progress,
    metrics: Phase8MetricState,
) -> Phase8TrainingState:
    return _make_training_state(
        run_id=state.run_id,
        code_commit=state.code_commit,
        configuration=state.configuration,
        model=state.model,
        optimizer=state.optimizer,
        order_generator=state.order_generator,
        progress=progress,
        metrics=metrics,
        global_cpu_rng_state=torch.get_rng_state().clone(),
        resumed_from_checkpoint_sha256=state.resumed_from_checkpoint_sha256,
        resume_checkpoint_sha256s=state.resume_checkpoint_sha256s,
    )


def _semantic_equal(left: Phase8TrainingState, right: Phase8TrainingState) -> bool:
    left_parameters = tuple(left.model.state_dict().items())
    right_parameters = tuple(right.model.state_dict().items())
    return (
        left.run_id == right.run_id
        and left.code_commit == right.code_commit
        and left.configuration == right.configuration
        and left.progress == right.progress
        and left.metrics == right.metrics
        and tuple(name for name, _ in left_parameters)
        == tuple(name for name, _ in right_parameters)
        and all(
            torch.equal(a, b)
            for (_, a), (_, b) in zip(left_parameters, right_parameters, strict=True)
        )
        and _objects_equal(left.optimizer.state_dict(), right.optimizer.state_dict())
        and torch.equal(
            left.order_generator.get_state(),
            right.order_generator.get_state(),
        )
        and all(
            torch.equal(
                a.attention_dropout._generator.get_state(),
                b.attention_dropout._generator.get_state(),
            )
            and torch.equal(
                a.feed_forward_dropout._generator.get_state(),
                b.feed_forward_dropout._generator.get_state(),
            )
            for a, b in zip(left.model.blocks, right.model.blocks, strict=True)
        )
        and torch.equal(left.global_cpu_rng_state, right.global_cpu_rng_state)
    )


def _run_exact_resume_audit(
    control: Phase8TrainingState,
    resumed: Phase8TrainingState,
    training_windows: tuple[object, ...],
    validation_windows: tuple[object, ...],
    repository_root: Path,
) -> tuple[Phase8TrainingState, Phase8TrainingState, tuple[dict[str, object], ...]]:
    control_order = create_phase8_epoch_order(
        len(training_windows),
        epoch=2,
        generator=control.order_generator,
    )
    resumed_order = create_phase8_epoch_order(
        len(training_windows),
        epoch=2,
        generator=resumed.order_generator,
    )
    if control_order != resumed_order:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="exact_resume_permutation",
        )
    control_batches = tuple(
        iter_phase8_logical_batches(
            training_windows,  # type: ignore[arg-type]
            control_order,
            epoch=2,
        )
    )
    resumed_batches = tuple(
        iter_phase8_logical_batches(
            training_windows,  # type: ignore[arg-type]
            resumed_order,
            epoch=2,
        )
    )
    if control_batches != resumed_batches:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="exact_resume_batches",
        )
    records: list[dict[str, object]] = []
    updates = 0
    examples = 0
    targets = 0
    for control_batch, resumed_batch in zip(
        control_batches,
        resumed_batches,
        strict=True,
    ):
        control_update, control_evidence = _run_phase8_logical_batch(
            control.model,
            control.optimizer,
            training_windows,  # type: ignore[arg-type]
            control_batch,
            capture_evidence=True,
            capture_sequence_gradients=True,
        )
        resumed_update, resumed_evidence = _run_phase8_logical_batch(
            resumed.model,
            resumed.optimizer,
            training_windows,  # type: ignore[arg-type]
            resumed_batch,
            capture_evidence=True,
            capture_sequence_gradients=True,
        )
        if (
            control_evidence is None
            or resumed_evidence is None
            or control_update != resumed_update
            or not _batch_evidence_equal(control_evidence, resumed_evidence)
            or not _branch_runtime_equal(control, resumed)
        ):
            raise Phase8ContractError(
                "phase8.contract.lifecycle",
                field="exact_resume_update",
                position=control_batch.batch_index,
            )
        updates += 1
        examples += len(control_batch.window_indices)
        targets += control_batch.target_count
        records.append(
            {
                **_batch_evidence_record(control_evidence),
                "control_resumed_equal": True,
                "cumulative_optimizer_updates": (
                    control.progress.optimizer_updates + updates
                ),
                "cumulative_examples_processed": (
                    control.progress.examples_processed + examples
                ),
                "cumulative_targets_processed": (
                    control.progress.targets_processed + targets
                ),
                "order_rng_sha256": _tensor_digest(
                    control.order_generator.get_state()
                ),
                "dropout_rng_sha256s": tuple(
                    _tensor_digest(value)
                    for block in control.model.blocks
                    for value in (
                        block.attention_dropout._generator.get_state(),
                        block.feed_forward_dropout._generator.get_state(),
                    )
                ),
                "global_rng_sha256": _tensor_digest(
                    control.global_cpu_rng_state
                ),
                "modes_and_gradients_valid": True,
            }
        )
    control_progress = Phase8Progress(
        completed_epochs=2,
        next_epoch=3,
        next_example_offset=0,
        optimizer_updates=control.progress.optimizer_updates + updates,
        examples_processed=control.progress.examples_processed + examples,
        targets_processed=control.progress.targets_processed + targets,
    )
    resumed_progress = Phase8Progress(
        completed_epochs=2,
        next_epoch=3,
        next_example_offset=0,
        optimizer_updates=resumed.progress.optimizer_updates + updates,
        examples_processed=resumed.progress.examples_processed + examples,
        targets_processed=resumed.progress.targets_processed + targets,
    )
    if control_progress != resumed_progress:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="exact_resume_progress",
        )
    control = _state_with(
        control,
        progress=control_progress,
        metrics=control.metrics,
    )
    resumed = _state_with(
        resumed,
        progress=resumed_progress,
        metrics=resumed.metrics,
    )
    control_metrics = _metrics_after_epoch(
        control,
        training_windows,
        validation_windows,
    )
    resumed_metrics = _metrics_after_epoch(
        resumed,
        training_windows,
        validation_windows,
    )
    if control_metrics != resumed_metrics:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="exact_resume_metrics",
        )
    control = _state_with(
        control,
        progress=control_progress,
        metrics=control_metrics,
    )
    resumed = _state_with(
        resumed,
        progress=resumed_progress,
        metrics=resumed_metrics,
    )
    if not _semantic_equal(control, resumed):
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="exact_resume",
        )
    control_payload = _payload(control, repository_root)
    resumed_payload = _payload(resumed, repository_root)
    control_lineage = control_payload.pop("resume_lineage")
    resumed_lineage = resumed_payload.pop("resume_lineage")
    if (
        not _objects_equal(control_payload, resumed_payload)
        or control_lineage == resumed_lineage
    ):
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="exact_resume_payload",
        )
    records.append(
        {
            "epoch": 2,
            "training_evaluation_equal": True,
            "validation_evaluation_equal": True,
            "metric_best_state_equal": True,
            "semantic_payload_equal_except_lineage": True,
        }
    )
    return control, resumed, tuple(records)


def run_fixed_phase8_experiment(
    corpus: Phase4PermittedCorpus,
    vocabulary: VocabularyBinding,
    config: Phase8Configuration,
    *,
    code_commit: str,
) -> Phase8RunResult:
    if type(corpus) is not Phase4PermittedCorpus:
        raise Phase8TypeError("phase8.type.argument", field="corpus")
    if type(vocabulary) is not VocabularyBinding:
        raise Phase8TypeError("phase8.type.argument", field="vocabulary")
    if type(config) is not Phase8Configuration:
        raise Phase8TypeError("phase8.type.argument", field="config")
    if type(code_commit) is not str:
        raise Phase8TypeError("phase8.type.argument", field="code_commit")
    if corpus.sealed_test_supplier is not None:
        raise Phase8GovernanceError("phase8.governance.sealed_test")
    if config != load_phase8_configuration():
        raise Phase8ContractError("phase8.contract.configuration")
    _validate_live_repository(REPOSITORY_ROOT, code_commit)
    run_id, _ = _planned_run_record(
        REPOSITORY_ROOT,
        code_commit=code_commit,
        configuration=config,
    )
    training_windows = build_phase8_windows(
        corpus.training_works,
        vocabulary,
        corpus.phase_1_manifest,
        split="train",
    )
    validation_windows = build_phase8_windows(
        corpus.validation_works,
        vocabulary,
        corpus.phase_1_manifest,
        split="validation",
    )
    model = _new_model(vocabulary).train()
    optimizer = create_phase8_optimizer(model)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(8001)
    progress = Phase8Progress(0, 1, 0, 0, 0, 0)
    training_evidence: list[dict[str, object]] = []
    initial_training = evaluate_phase8_split(
        model,
        optimizer,
        training_windows,
        order_generator,
        progress,
        split="train",
    )
    initial_validation = evaluate_phase8_split(
        model,
        optimizer,
        validation_windows,
        order_generator,
        progress,
        split="validation",
    )
    progress = _train_epoch(
        model,
        optimizer,
        order_generator,
        progress,
        training_windows,
        1,
        evidence_sink=training_evidence,
    )
    epoch_one_training = evaluate_phase8_split(
        model,
        optimizer,
        training_windows,
        order_generator,
        progress,
        split="train",
    )
    epoch_one_validation = evaluate_phase8_split(
        model,
        optimizer,
        validation_windows,
        order_generator,
        progress,
        split="validation",
    )
    metrics = _make_metric_state(
        initialized_training=initial_training,
        initialized_validation=initial_validation,
        epoch_training=(epoch_one_training,),
        epoch_validation=(epoch_one_validation,),
        current_training=epoch_one_training,
        current_validation=epoch_one_validation,
        best_validation_loss=epoch_one_validation.loss,
        best_validation_epoch=1,
        best_logical_id="epoch-0001",
    )
    state = _make_training_state(
        run_id=run_id,
        code_commit=code_commit,
        configuration=config,
        model=model,
        optimizer=optimizer,
        order_generator=order_generator,
        progress=progress,
        metrics=metrics,
        global_cpu_rng_state=torch.get_rng_state().clone(),
        resumed_from_checkpoint_sha256=None,
        resume_checkpoint_sha256s=(),
    )
    save_phase8_checkpoint(state, REPOSITORY_ROOT)
    exact_resume_passed = False
    control = state
    resumed = load_phase8_checkpoint(
        REPOSITORY_ROOT,
        run_id=run_id,
        role="latest",
    )
    audit_seed = load_phase8_checkpoint(
        REPOSITORY_ROOT,
        run_id=run_id,
        role="latest",
    )
    audit_components = (
        "experiments",
        run_id,
        "artifacts",
        "resume-audit-control",
        "objects",
    )
    _publish_payload(
        _payload(audit_seed, REPOSITORY_ROOT),
        REPOSITORY_ROOT,
        components=audit_components,
    )
    control, resumed, audit_records = _run_exact_resume_audit(
        control,
        resumed,
        training_windows,
        validation_windows,
        REPOSITORY_ROOT,
    )
    control_reference = _publish_payload(
        _payload(control, REPOSITORY_ROOT),
        REPOSITORY_ROOT,
        components=audit_components,
    )
    resumed_reference = save_phase8_checkpoint(resumed, REPOSITORY_ROOT)
    reloaded_control = _load_phase8_checkpoint(
        REPOSITORY_ROOT,
        run_id=run_id,
        role="latest",
        components=audit_components,
    )
    reloaded_resumed = load_phase8_checkpoint(
        REPOSITORY_ROOT,
        run_id=run_id,
        role="latest",
    )
    if not _semantic_equal(reloaded_control, reloaded_resumed):
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="exact_resume_public_reload",
        )
    audit_records = (
        *audit_records,
        {
            "control_checkpoint_sha256": control_reference.sha256,
            "control_checkpoint_relative_path": control_reference.relative_path,
            "resumed_checkpoint_sha256": resumed_reference.sha256,
            "resumed_checkpoint_relative_path": resumed_reference.relative_path,
            "both_catalog_graphs_verified": True,
            "both_public_load_paths_reloaded": True,
            "semantic_state_equal_except_lineage_hash_location": True,
        },
    )
    state = reloaded_resumed
    training_evidence.extend(audit_records)
    exact_resume_passed = True
    for epoch in range(3, 11):
        progress = _train_epoch(
            state.model,
            state.optimizer,
            state.order_generator,
            state.progress,
            training_windows,
            epoch,
            evidence_sink=training_evidence,
        )
        state = _state_with(state, progress=progress, metrics=state.metrics)
        metrics = _metrics_after_epoch(state, training_windows, validation_windows)
        state = _state_with(state, progress=progress, metrics=metrics)
        save_phase8_checkpoint(state, REPOSITORY_ROOT)
    latest, best = _catalog_references(REPOSITORY_ROOT, run_id)
    _write_evidence_artifact(
        REPOSITORY_ROOT,
        run_id,
        tuple(training_evidence),
        latest,
        best,
    )
    if min(item.loss for item in state.metrics.epoch_training) >= initial_training.loss:
        raise Phase8ContractError(
            "phase8.contract.lifecycle",
            field="training_improvement",
        )
    return Phase8RunResult(
        run_id=run_id,
        status="completed",
        stopping_reason="maximum_epochs",
        progress=state.progress,
        initialized_training=initial_training,
        initialized_validation=initial_validation,
        epoch_training=state.metrics.epoch_training,
        epoch_validation=state.metrics.epoch_validation,
        latest_checkpoint=latest,
        best_checkpoint=best,
        exact_resume_passed=exact_resume_passed,
    )


__all__ = [
    "configure_phase8_runtime",
    "load_phase8_configuration",
    "run_fixed_phase8_experiment",
    "validate_phase8_runtime",
]
