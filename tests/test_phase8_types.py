"""Independent literal-oracle tests for the Phase 8 public surface."""

from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.training as training  # noqa: E402
import sebgpt.training.phase8_checkpoint as checkpoint  # noqa: E402
import sebgpt.training.phase8_data as data  # noqa: E402
import sebgpt.training.phase8_experiment as experiment  # noqa: E402
import sebgpt.training.phase8_optimization as optimization  # noqa: E402
import sebgpt.training.phase8_types as types_module  # noqa: E402


PACKAGE_EXPORTS = (
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
    "build_phase8_windows",
    "configure_phase8_runtime",
    "create_phase8_epoch_order",
    "create_phase8_optimizer",
    "evaluate_phase8_split",
    "iter_phase8_logical_batches",
    "load_phase8_checkpoint",
    "load_phase8_configuration",
    "run_fixed_phase8_experiment",
    "run_phase8_logical_batch",
    "save_phase8_checkpoint",
    "validate_phase8_runtime",
)

RECORD_ORACLE = {
    "Phase8Window": (
        (
            ("work_id", "str", True),
            ("manifest_order", "int", True),
            ("split", "str", True),
            ("start_index", "int", True),
            ("input_ids", "tuple[int, ...]", False),
            ("target_ids", "tuple[int, ...]", False),
        ),
        False,
    ),
    "Phase8LogicalBatch": (
        (("epoch", "int", True), ("batch_index", "int", True), ("window_indices", "tuple[int, ...]", True), ("target_count", "int", True)),
        False,
    ),
    "Phase8BatchUpdate": (
        (("batch_loss", "float", True), ("target_count", "int", True), ("sequence_count", "int", True), ("pre_clip_global_norm", "float", True)),
        False,
    ),
    "Phase8Evaluation": (
        (("split", "str", True), ("loss", "float", True), ("target_count", "int", True), ("window_count", "int", True)),
        False,
    ),
    "Phase8Progress": (
        (("completed_epochs", "int", True), ("next_epoch", "int", True), ("next_example_offset", "int", True), ("optimizer_updates", "int", True), ("examples_processed", "int", True), ("targets_processed", "int", True)),
        False,
    ),
    "Phase8CheckpointReference": (
        (("run_id", "str", True), ("role", "str", True), ("logical_id", "str", True), ("epoch", "int", True), ("validation_loss_hex", "str", True), ("sha256", "str", True), ("relative_path", "str", True)),
        False,
    ),
    "Phase8RunResult": (
        (("run_id", "str", True), ("status", "str", True), ("stopping_reason", "str", True), ("progress", "Phase8Progress", True), ("initialized_training", "Phase8Evaluation", True), ("initialized_validation", "Phase8Evaluation", True), ("epoch_training", "tuple[Phase8Evaluation, ...]", True), ("epoch_validation", "tuple[Phase8Evaluation, ...]", True), ("latest_checkpoint", "Phase8CheckpointReference", True), ("best_checkpoint", "Phase8CheckpointReference", True), ("exact_resume_passed", "bool", True)),
        False,
    ),
    "Phase8RuntimeIdentity": (
        (("schema_version", "int", True), ("python_version", "str", True), ("python_implementation", "str", True), ("torch_version", "str", True), ("operating_system", "str", True), ("operating_system_release", "str", True), ("machine", "str", True), ("processor", "str", True), ("device", "str", True), ("parameter_dtype", "str", True), ("token_dtype", "str", True), ("torch_intra_op_threads", "int", True), ("torch_inter_op_threads", "int", True), ("deterministic_algorithms", "bool", True), ("deterministic_warn_only", "bool", True), ("float32_matmul_precision", "str", True), ("default_dtype", "str", True), ("default_device", "str", True), ("grad_mode_enabled", "bool", True), ("cpu_autocast_enabled", "bool", True), ("inference_mode_enabled", "bool", True), ("mkldnn_enabled", "bool", True), ("torch_build_config_sha256", "str", True), ("torch_parallel_info_sha256", "str", True), ("requirements_lock_sha256", "str", True)),
        False,
    ),
    "Phase8Configuration": (
        (("schema_version", "int", True), ("runtime", "Phase8RuntimeIdentity", True), ("context_length", "int", True), ("stride", "int", True), ("retain_unpadded_tail", "bool", True), ("logical_batch_capacity", "int", True), ("allow_final_partial_batch", "bool", True), ("maximum_epochs", "int", True), ("order_seed", "int", True), ("order_algorithm", "str", True), ("optimizer_name", "str", True), ("learning_rate", "float", True), ("betas", "tuple[float, float]", True), ("epsilon", "float", True), ("weight_decay", "float", True), ("amsgrad", "bool", True), ("maximize", "bool", True), ("foreach", "bool", True), ("capturable", "bool", True), ("differentiable", "bool", True), ("fused", "bool", True), ("scheduler_name", "None", True), ("warmup_steps", "int", True), ("gradient_clip_norm_type", "float", True), ("gradient_clip_max_norm", "float", True), ("gradient_clip_epsilon", "float", True), ("evaluate_initialized_state", "bool", True), ("evaluation_interval_epochs", "int", True), ("evaluation_split_order", "tuple[str, str]", True), ("best_comparison", "str", True), ("validation_early_stopping", "bool", True), ("parameter_device", "str", True), ("parameter_dtype", "str", True), ("token_dtype", "str", True), ("checkpoint_schema_version", "int", True), ("catalog_schema_version", "int", True), ("maximum_catalog_bytes", "int", True), ("maximum_checkpoint_object_bytes", "int", True)),
        True,
    ),
    "Phase8MetricState": (
        (("initialized_training", "Phase8Evaluation", True), ("initialized_validation", "Phase8Evaluation", True), ("epoch_training", "tuple[Phase8Evaluation, ...]", True), ("epoch_validation", "tuple[Phase8Evaluation, ...]", True), ("current_training", "Phase8Evaluation", True), ("current_validation", "Phase8Evaluation", True), ("best_validation_loss", "float", True), ("best_validation_epoch", "int", True), ("best_logical_id", "str", True)),
        True,
    ),
    "Phase8TrainingState": (
        (("run_id", "str", True), ("code_commit", "str", True), ("configuration", "Phase8Configuration", True), ("model", "MiniGPT", False), ("optimizer", "AdamW", False), ("order_generator", "torch.Generator", False), ("progress", "Phase8Progress", True), ("metrics", "Phase8MetricState", True), ("global_cpu_rng_state", "torch.Tensor", False), ("resumed_from_checkpoint_sha256", "str | None", True), ("resume_checkpoint_sha256s", "tuple[str, ...]", True)),
        True,
    ),
}


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _resolved_name(node: ast.AST, aliases: dict[str, str]) -> str:
    raw = _qualified_name(node)
    if not raw:
        return ""
    first, separator, remainder = raw.partition(".")
    resolved_first = aliases.get(first.lower(), first)
    return (
        f"{resolved_first}.{remainder}" if separator else resolved_first
    ).lower()


def _constant_string(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return node.value.lower()
    if isinstance(node, ast.Name):
        return constants.get(node.id.lower())
    return None


def _phase8_static_violations(source: str) -> set[str]:
    tree = ast.parse(source)
    violations: set[str] = set()
    identifiers: set[str] = set()
    qualified_names: set[str] = set()
    string_constants: list[str] = []
    aliases: dict[str, str] = {}
    assignments: list[tuple[str, ast.AST]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name.lower())
        elif isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())
            qualified_names.add(_qualified_name(node).lower())
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg.lower())
        elif isinstance(node, ast.keyword) and node.arg is not None:
            identifiers.add(node.arg.lower())
        elif isinstance(node, ast.Constant) and type(node.value) is str:
            string_constants.append(node.value.lower())
        elif isinstance(node, ast.Import):
            for imported in node.names:
                qualified = imported.name.lower()
                bound = (imported.asname or imported.name.split(".", 1)[0]).lower()
                aliases[bound] = qualified
                qualified_names.add(qualified)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            if module:
                qualified_names.add(module)
            for imported in node.names:
                qualified = f"{module}.{imported.name}" if module else imported.name
                aliases[(imported.asname or imported.name).lower()] = qualified.lower()
                qualified_names.add(qualified.lower())
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id.lower(), node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments.append((node.target.id.lower(), node.value))

    constants: dict[str, str] = {}
    for _ in range(2):
        for target, value in assignments:
            resolved = _constant_string(value, constants)
            if resolved is not None:
                constants[target] = resolved
            elif isinstance(value, (ast.Name, ast.Attribute)):
                qualified = _resolved_name(value, aliases)
                if qualified:
                    aliases[target] = qualified

    calls = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    called_names = {_resolved_name(node.func, aliases) for node in calls}
    if any(
        name.startswith("generate")
        or name.startswith("generation_")
        or "text_generation" in name
        for name in identifiers | called_names
    ):
        violations.add("generation")
    if any(
        name.startswith("sample")
        or "sampling" in name
        or name.endswith(".multinomial")
        for name in identifiers | called_names
        if "generator" not in name
    ):
        violations.add("sampling")
    authority_call_markers = (
        "build_phase8_windows",
        "build_windows",
        "evaluate_phase8_split",
        "phase8configuration",
        "_make_configuration",
    )
    authority_calls = tuple(
        call
        for call in calls
        if any(marker in _resolved_name(call.func, aliases) for marker in authority_call_markers)
    )
    if any(
        _constant_string(argument, constants) == "test"
        for call in authority_calls
        for argument in call.args
    ) or any(
        keyword.arg == "split"
        and _constant_string(keyword.value, constants) == "test"
        for call in authority_calls
        for keyword in call.keywords
    ):
        violations.add("test_or_sealed_split")
    sealed_markers = ("twelfth night", "twelfth-night", "twelfth_night")
    def is_sealed_value(value: str | None) -> bool:
        return value is not None and any(marker in value for marker in sealed_markers)

    if any(
        ("sealed" in name or "path" in name or "work" in name)
        and any(marker in value for marker in sealed_markers)
        for name, value in constants.items()
    ) or any(
        is_sealed_value(_constant_string(argument, constants))
        for call in authority_calls
        for argument in call.args
    ) or any(
        is_sealed_value(_constant_string(keyword.value, constants))
        for call in authority_calls
        for keyword in call.keywords
    ):
        violations.add("sealed_work_identity_or_path")
    for device in ("cuda", "mps"):
        if any(
            name == device
            or name.startswith(f"torch.{device}")
            or f".{device}." in name
            for name in identifiers | qualified_names | called_names
        ) or any(value == device or value.startswith(f"{device}:") for value in string_constants):
            violations.add(device)
    prohibited_precision_literals = {
        "torch.float16",
        "float16",
        "fp16",
        "torch.bfloat16",
        "bfloat16",
        "bf16",
        "mixed_precision",
    }
    if any(
        "mixed_precision" in name or "gradscaler" in name
        for name in identifiers | qualified_names | called_names
    ) or any(value in prohibited_precision_literals for value in string_constants):
        violations.add("mixed_precision")
    if any(name.endswith(".half") or name == "half" for name in called_names):
        violations.add("half")
    if any(name.endswith("float16") for name in identifiers | qualified_names) or any(
        value in {"torch.float16", "float16", "fp16"} for value in string_constants
    ):
        violations.add("float16")
    if any(name.endswith("bfloat16") for name in identifiers | qualified_names) or any(
        value in {"torch.bfloat16", "bfloat16", "bf16"} for value in string_constants
    ):
        violations.add("bfloat16")
    if any(
        name == "autocast" or name.endswith(".autocast")
        for name in called_names
    ):
        violations.add("autocast_mixed_precision")
    padding_calls = {
        "torch.nn.functional.pad",
        "torch.nn.utils.rnn.pad_sequence",
    }
    if any(
        name == "padding" or name.startswith("padding_") or name.endswith("_padding")
        for name in identifiers
    ) or any(name in padding_calls for name in called_names):
        violations.add("padding")
    if any("pad_sequence" in name for name in identifiers | qualified_names | called_names):
        violations.add("pad_sequence")
    special_token_prefixes = ("bos", "eos", "pad", "unk", "mask")
    if any(
        name in {"special_token", "special_tokens"}
        or any(
            name == f"{prefix}_id"
            or name.startswith(f"{prefix}_id_")
            or name.endswith(f"_{prefix}_id")
            or name == f"{prefix}_token"
            or name.startswith(f"{prefix}_token_")
            or name.endswith(f"_{prefix}_token")
            for prefix in special_token_prefixes
        )
        for name in identifiers
    ) or any(
        value in {"<pad>", "<bos>", "<eos>", "<unk>", "<mask>"}
        for value in string_constants
    ):
        violations.add("special_tokens")
    if any(
        marker in name
        for name in identifiers | qualified_names | called_names
        for marker in ("from_pretrained", "pretrained", "transformers", "huggingface")
    ):
        violations.add("pretrained_api")
    if any(
        ("scheduler" in name and name != "scheduler_name")
        or "lr_scheduler" in name
        for name in identifiers | qualified_names | called_names
    ):
        violations.add("scheduler")
    if any(
        name.endswith("steplr")
        or name.endswith("cosineannealinglr")
        or name.endswith("reducelronplateau")
        for name in identifiers | qualified_names | called_names
    ):
        violations.add("scheduler_class")
    if any("warmup" in name and name != "warmup_steps" for name in identifiers | called_names):
        violations.add("warmup")
    if any(
        "randomsampler" in name
        or "weightedsampler" in name
        or "replacement_sampling" in name
        for name in identifiers | qualified_names | called_names
    ) or any(
        keyword.arg == "replacement"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for call in calls
        for keyword in call.keywords
    ):
        violations.add("replacement_sampling")
    return violations

OBLIGATION_EVIDENCE = {
    1: ("public-schema-literal", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_public_signatures_are_exact", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_exact_record_fields", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_exception_surface_is_content_safe")),
    2: ("prior-phase-byte-integrity", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_accepted_prior_phase_byte_identities_are_literal",)),
    3: ("corpus-governance-behavior", ("tests.test_phase8_data.Phase8DataTests.test_metadata_failure_precedes_text_access", "tests.test_phase8_experiment.Phase8ExperimentTests.test_sealed_supplier_is_rejected_before_configuration_or_repository")),
    4: ("phase8-phase9-static-boundary", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_static_phase8_surface_excludes_phase9_operations", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_static_boundary_detector_rejects_each_prohibited_mutation_in_every_module", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_static_boundary_detector_false_positive_controls")),
    5: ("window-mathematics", ("tests.test_phase8_data.Phase8DataTests.test_context_stride_tail_and_exact_transition_coverage",)),
    6: ("window-edge-behavior", ("tests.test_phase8_data.Phase8DataTests.test_empty_and_one_token_documents_produce_no_windows", "tests.test_phase8_data.Phase8DataTests.test_exact_boundary_lengths")),
    7: ("window-identity-and-boundary", ("tests.test_phase8_data.Phase8DataTests.test_work_and_validation_boundaries_remain_separate",)),
    8: ("ordering-literal-and-rng", ("tests.test_phase8_data.Phase8DataTests.test_epoch_shuffle_has_literal_supported_permutations_and_rng_isolation",)),
    9: ("checkpointed-ordering-state", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_checkpointed_order_state_reproduces_the_next_epoch_permutation",)),
    10: ("logical-batch-counts-0-1-7-8-9-17", ("tests.test_phase8_data.Phase8DataTests.test_literal_batch_grouping_for_zero_one_seven_eight_nine_and_larger", "tests.test_phase8_data.Phase8DataTests.test_empty_batching_requires_a_paired_empty_order")),
    11: ("independent-gradient-oracle", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_independent_single_expression_gradient_oracle_matches_accumulation",)),
    12: ("rank-one-order-and-step", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_logical_batch_produces_one_step_and_none_gradients",)),
    13: ("adamw-literal-state", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_optimizer_has_exact_supported_group_schema",)),
    14: ("gradient-and-clipping-boundaries", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_clipping_boundaries_match_literal_supported_formula", "tests.test_phase8_optimization.Phase8OptimizationTests.test_nonfinite_clipping_norm_is_rejected")),
    15: ("pre-step-failure-transaction", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_training_authority_rejections_precede_forward_and_preserve_state", "tests.test_phase8_optimization.Phase8OptimizationTests.test_post_clip_invalid_gradient_prevents_step_and_clears_failure_state")),
    16: ("evaluation-cadence-runner", ("tests.test_phase8_experiment.Phase8ExperimentTests.test_runner_has_no_epoch_zero_state_and_executes_durable_dual_branch_audit",)),
    17: ("evaluation-nonmutation", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_evaluation_is_target_weighted_and_nonmutating",)),
    18: ("evaluation-failure-matrix", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_evaluation_failure_matrix_preserves_each_defined_boundary", "tests.test_phase8_optimization.Phase8OptimizationTests.test_evaluation_mode_restoration_failure_is_attempted_without_new_rollback_claim")),
    19: ("best-and-epoch0-lifecycle", ("tests.test_phase8_experiment.Phase8ExperimentTests.test_runner_has_no_epoch_zero_state_and_executes_durable_dual_branch_audit", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_latest_best_catalog_semantics_are_bound_to_latest_payload")),
    20: ("complete-payload-schema", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_payload_schema_optimizer_steps_and_configuration_version", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_every_nested_schema_category_is_noncoercing")),
    21: ("incomplete-payload-refusal", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_model_state_dict_alone_is_rejected_as_incomplete_checkpoint", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_missing_and_unknown_configuration_schema_versions_are_rejected")),
    22: ("single-serialization-content-addressing", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_save_bootstraps_content_addressed_object_and_catalog", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_hash_and_deserialization_use_the_same_bounded_bytes")),
    23: ("schema-corruption-refusal", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_progress_schema_is_literal_and_cross_validated", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_historical_best_payload_is_fully_validated")),
    24: ("publication-crash-failure-matrix", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_publication_failure_matrix_preserves_exact_authority_boundaries",)),
    25: ("atomic-latest-best-semantics", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_latest_best_catalog_semantics_are_bound_to_latest_payload",)),
    26: ("ordered-restoration-final-global-commit", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_restoration_order_commits_global_rng_as_final_semantic_action", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_global_rng_install_failure_rolls_back_once", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_global_rng_install_and_rollback_failure_reports_unknown_once")),
    27: ("checkpoint-content-exclusions", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_checkpoint_and_catalog_schemas_directly_exclude_forbidden_content",)),
    # The runner test exercises the production audit, both durable roots, public-path reloads, and semantic comparison; the older in-memory-only test is intentionally not used here.
    28: ("durable-dual-branch-exact-resume", ("tests.test_phase8_experiment.Phase8ExperimentTests.test_runner_has_no_epoch_zero_state_and_executes_durable_dual_branch_audit",)),
    29: ("truthful-resume-lineage", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_round_trip_restores_model_optimizer_progress_rng_and_lineage",)),
    30: ("fixed-runner-lifecycle", ("tests.test_phase8_experiment.Phase8ExperimentTests.test_runner_has_no_epoch_zero_state_and_executes_durable_dual_branch_audit",)),
    31: ("failure-matrix-no-completion", ("tests.test_phase8_optimization.Phase8OptimizationTests.test_training_authority_rejections_precede_forward_and_preserve_state", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_catalog_replace_failure_leaves_only_non_authoritative_object")),
    32: ("regression-and-byte-integrity", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_accepted_prior_phase_byte_identities_are_literal",)),
    33: ("manual-static-boundary", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_static_phase8_surface_excludes_phase9_operations", "tests.test_phase8_experiment.Phase8ExperimentTests.test_sealed_supplier_is_rejected_before_configuration_or_repository")),
    34: ("bootstrap-reuse-and-failure-matrix", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_bootstrap_reuse_partial_and_unsafe_entry_matrix", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_fsync_order_places_objects_before_catalog_and_run_commit")),
    35: ("exact-descriptor-fsync-order", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_fsync_order_places_objects_before_catalog_and_run_commit",)),
    36: ("crash-state-interpretation", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_orphan_object_never_changes_catalog_authority", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_post_commit_diagnostic_failure_reports_committed_reference")),
    37: ("late-load-and-rng-rollback", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_late_load_failure_rolls_back_global_rng", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_global_rng_install_failure_rolls_back_once", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_global_rng_install_and_rollback_failure_reports_unknown_once")),
    38: ("complete-runtime-mismatch-matrix", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_every_runtime_identity_field_mismatch_is_rejected", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_runtime_integer_fields_reject_boolean_coercion")),
    39: ("complete-public-record-schema-oracle", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_exact_module_and_package_exports", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_every_public_object_has_exact_owner", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_exact_record_fields", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_public_signatures_are_exact", "tests.test_phase8_types.Phase8PublicSurfaceTests.test_window_repr_exposes_only_safe_identity_fields")),
    40: ("catalog-cross-field-and-mode-negatives", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_latest_best_catalog_semantics_are_bound_to_latest_payload", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_malformed_historical_best_mode_maps_are_rejected_before_cross_validation", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_malformed_latest_mode_maps_are_rejected_before_restoration")),
    41: ("path-grammar-and-symlink-refusal", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_literal_reference_grammar_rejects_alternate_spellings", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_run_id_traversal_is_rejected_before_path_access", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_symlinked_objects_directory_is_rejected")),
    42: ("descriptor-relative-no-follow-no-ambient-reopen", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_checkpoint_filesystem_calls_are_descriptor_relative_no_follow", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_cleanup_refuses_substituted_temporary_entry", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_hash_and_deserialization_use_the_same_bounded_bytes")),
    43: ("bounded-read-edge-matrix", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_bounded_read_zero_limit_growth_shrink_short_and_truncated_cases", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_hash_and_deserialization_use_the_same_bounded_bytes")),
    44: ("trusted-local-resource-boundary", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_object_byte_limit_is_checked_before_deserialization", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_symlinked_objects_directory_is_rejected")),
    45: ("live-preflight-and-preregistration", ("tests.test_phase8_experiment.Phase8ExperimentTests.test_planned_code_commit_is_literal_implementation_anchor", "tests.test_phase8_experiment.Phase8ExperimentTests.test_corrected_live_repository_preflight_matrix", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_checkpoint_save_and_load_validate_corrected_provenance", "tests.test_phase8_checkpoint.Phase8CheckpointTests.test_resume_under_different_pre_registration_authority_fails", "tests.test_phase8_experiment.Phase8ExperimentTests.test_result_provenance_representation_keeps_commits_distinct", "tests.test_phase8_experiment.Phase8ExperimentTests.test_run_authority_path_rejects_symlinks_and_wrong_types_without_outside_write", "tests.test_phase8_experiment.Phase8ExperimentTests.test_run_authority_path_existing_tree_and_descriptor_relative_fsync_order", "tests.test_phase8_experiment.Phase8ExperimentTests.test_run_authority_directory_failure_maps_to_governance", "tests.test_phase8_experiment.Phase8ExperimentTests.test_run_authority_cleanup_preserves_substituted_evidence")),
    46: ("latest-only-continuation", ("tests.test_phase8_checkpoint.Phase8CheckpointTests.test_best_validation_is_not_a_continuation_role",)),
    47: ("dual-branch-public-reload-audit", ("tests.test_phase8_experiment.Phase8ExperimentTests.test_runner_has_no_epoch_zero_state_and_executes_durable_dual_branch_audit",)),
    48: ("literal-sequential-gate-oracle", ("tests.test_phase8_types.Phase8PublicSurfaceTests.test_contract_gate_oracle_is_literal_sequential_and_distinct",)),
}


class Phase8PublicSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        training.configure_phase8_runtime()

    def test_exact_module_and_package_exports(self) -> None:
        self.assertEqual(
            tuple(types_module.__all__),
            PACKAGE_EXPORTS[:17],
        )
        self.assertEqual(
            tuple(data.__all__),
            (
                "build_phase8_windows",
                "create_phase8_epoch_order",
                "iter_phase8_logical_batches",
            ),
        )
        self.assertEqual(
            tuple(optimization.__all__),
            (
                "create_phase8_optimizer",
                "evaluate_phase8_split",
                "run_phase8_logical_batch",
            ),
        )
        self.assertEqual(
            tuple(checkpoint.__all__),
            ("load_phase8_checkpoint", "save_phase8_checkpoint"),
        )
        self.assertEqual(
            tuple(experiment.__all__),
            (
                "configure_phase8_runtime",
                "load_phase8_configuration",
                "run_fixed_phase8_experiment",
                "validate_phase8_runtime",
            ),
        )
        self.assertEqual(tuple(training.__all__), PACKAGE_EXPORTS)

    def test_all_48_contract_obligations_map_to_runnable_independent_evidence(self) -> None:
        self.assertEqual(tuple(OBLIGATION_EVIDENCE), tuple(range(1, 49)))
        loader = unittest.TestLoader()
        categories: set[str] = set()
        for obligation, (category, test_ids) in OBLIGATION_EVIDENCE.items():
            self.assertTrue(category)
            self.assertTrue(test_ids)
            categories.add(category)
            for test_id in test_ids:
                with self.subTest(obligation=obligation, category=category, test_id=test_id):
                    suite = loader.loadTestsFromName(test_id)
                    self.assertEqual(suite.countTestCases(), 1)
                    self.assertFalse(
                        any(
                            isinstance(test, unittest.loader._FailedTest)
                            for test in suite
                        )
                    )
        self.assertGreaterEqual(len(categories), 40)

    def test_every_public_object_has_exact_owner(self) -> None:
        for name in PACKAGE_EXPORTS[:17]:
            self.assertEqual(
                getattr(training, name).__module__,
                "sebgpt.training.phase8_types",
            )
        owners = {
            "build_phase8_windows": "sebgpt.training.phase8_data",
            "create_phase8_epoch_order": "sebgpt.training.phase8_data",
            "iter_phase8_logical_batches": "sebgpt.training.phase8_data",
            "create_phase8_optimizer": "sebgpt.training.phase8_optimization",
            "evaluate_phase8_split": "sebgpt.training.phase8_optimization",
            "run_phase8_logical_batch": "sebgpt.training.phase8_optimization",
            "load_phase8_checkpoint": "sebgpt.training.phase8_checkpoint",
            "save_phase8_checkpoint": "sebgpt.training.phase8_checkpoint",
            "configure_phase8_runtime": "sebgpt.training.phase8_experiment",
            "load_phase8_configuration": "sebgpt.training.phase8_experiment",
            "run_fixed_phase8_experiment": "sebgpt.training.phase8_experiment",
            "validate_phase8_runtime": "sebgpt.training.phase8_experiment",
        }
        for name, owner in owners.items():
            self.assertEqual(getattr(training, name).__module__, owner)

    def test_accepted_prior_phase_byte_identities_are_literal(self) -> None:
        expected = {
            "docs/MINI_GPT_SPEC.md": "3e1987658d4c9ece59bebbea7061f939c1138aaa73751a523f659f8b3217c022",
            "src/sebgpt/model/mini_gpt.py": "6b8db96577a0ad1f5daa4649d7ca9375e59873d4b635c3f76e75a6751f74f12d",
            "tests/test_mini_gpt.py": "95ec62d1164c48525e59e33f3f90fada6d482c84a31ca16edba38178b16b2cab",
            "src/sebgpt/model/__init__.py": "af056658e6d7ffe6de89b3ac0486929305655ea76029240d26869b705a2a5d86",
        }
        for relative_path, expected_digest in expected.items():
            with self.subTest(path=relative_path):
                self.assertEqual(
                    hashlib.sha256(
                        (REPOSITORY_ROOT / relative_path).read_bytes()
                    ).hexdigest(),
                    expected_digest,
                )

    def test_static_phase8_surface_excludes_phase9_operations(self) -> None:
        implementation_paths = tuple(
            sorted((REPOSITORY_ROOT / "src/sebgpt/training").glob("*.py"))
        )
        self.assertEqual(len(implementation_paths), 6)
        for path in implementation_paths:
            with self.subTest(module=path.name):
                self.assertEqual(
                    _phase8_static_violations(path.read_text(encoding="utf-8")),
                    set(),
                )
        self.assertTrue(
            all(
                "generate" not in name.lower() and "sample" not in name.lower()
                for name in training.__all__
            )
        )
        config = training.load_phase8_configuration()
        self.assertEqual(config.parameter_device, "cpu")
        self.assertEqual(config.parameter_dtype, "torch.float32")
        self.assertFalse(config.runtime.cpu_autocast_enabled)
        self.assertFalse(config.runtime.inference_mode_enabled)
        self.assertIsNone(config.scheduler_name)
        self.assertEqual(config.warmup_steps, 0)
        self.assertEqual(config.order_algorithm, "torch_randperm_local_cpu_generator")
        self.assertNotIn("replacement", config.order_algorithm)
        configuration_fields = set(training.Phase8Configuration.__annotations__)
        self.assertTrue(
            {
                "padding",
                "padding_token",
                "bos_token",
                "eos_token",
                "special_tokens",
                "scheduler",
                "mixed_precision",
            }.isdisjoint(configuration_fields)
        )

    def test_static_boundary_detector_rejects_each_prohibited_mutation_in_every_module(self) -> None:
        mutations = (
            ("generation_helper", "generation", "def generate_text():\n    return 'x'"),
            ("sampling_helper", "sampling", "def sample_next_token():\n    return 0"),
            ("direct_test_positional", "test_or_sealed_split", "build_windows((), None, {}, 'test')"),
            ("direct_test_keyword", "test_or_sealed_split", "build_windows((), None, {}, split='test')"),
            ("indirect_test", "test_or_sealed_split", "split = 'test'\nbuild_windows((), None, {}, split)"),
            ("aliased_test", "test_or_sealed_split", "heldout_split = 'test'\nsplit = heldout_split\nbuild_windows((), None, {}, split)"),
            ("direct_sealed_work", "sealed_work_identity_or_path", "build_windows('Twelfth Night', None, {})"),
            ("direct_sealed_path", "sealed_work_identity_or_path", "sealed_path = 'data/processed/twelfth-night.txt'"),
            ("indirect_sealed_work", "sealed_work_identity_or_path", "work = 'Twelfth Night'\nbuild_windows(work, None, {})"),
            ("aliased_sealed_work", "sealed_work_identity_or_path", "heldout_work = 'Twelfth Night'\nwork = heldout_work\nbuild_windows(work, None, {})"),
            ("cuda", "cuda", "device = torch.device('cuda')"),
            ("mps", "mps", "device = torch.device('mps')"),
            ("mixed_precision_identifier", "mixed_precision", "mixed_precision = True"),
            ("mixed_precision_string", "mixed_precision", "precision = 'mixed_precision'"),
            ("half", "half", "value = tensor.half()"),
            ("torch_float16", "float16", "value = torch.zeros(1, dtype=torch.float16)"),
            ("torch_bfloat16", "bfloat16", "value = torch.zeros(1, dtype=torch.bfloat16)"),
            ("string_torch_float16", "float16", "parameter_dtype = 'torch.float16'"),
            ("string_bfloat16", "bfloat16", "parameter_dtype = 'bfloat16'"),
            ("string_fp16", "float16", "parameter_dtype = 'fp16'"),
            ("string_bf16", "bfloat16", "parameter_dtype = 'bf16'"),
            ("autocast", "autocast_mixed_precision", "with torch.autocast('cpu'):\n    pass"),
            ("padding_policy", "padding", "padding = True"),
            ("qualified_pad", "padding", "value = torch.nn.functional.pad(tensor, (0, 1))"),
            ("aliased_pad", "padding", "import torch.nn.functional as F\nvalue = F.pad(tensor, (0, 1))"),
            ("imported_pad", "padding", "from torch.nn.functional import pad\nvalue = pad(tensor, (0, 1))"),
            ("pad_sequence", "pad_sequence", "value = torch.nn.utils.rnn.pad_sequence(items)"),
            ("aliased_pad_sequence", "pad_sequence", "from torch.nn.utils.rnn import pad_sequence as ps\nvalue = ps(items)"),
            ("bos_id", "special_tokens", "bos_id = 1"),
            ("pad_token", "special_tokens", "pad_token = '<pad>'"),
            ("generic_pad_literal", "special_tokens", "token = '<pad>'"),
            ("generic_bos_literal", "special_tokens", "token = '<BOS>'"),
            ("generic_eos_literal", "special_tokens", "token = '<eos>'"),
            ("mask_id", "special_tokens", "MASK_ID = 4"),
            ("pretrained", "pretrained_api", "from transformers import AutoModel\nmodel = AutoModel.from_pretrained('x')"),
            ("scheduler", "scheduler", "scheduler = object()"),
            ("scheduler_class", "scheduler_class", "scheduler_value = torch.optim.lr_scheduler.StepLR(optimizer, 1)"),
            ("warmup", "warmup", "def linear_warmup(step):\n    return step"),
            ("replacement_sampling", "replacement_sampling", "sampler = torch.utils.data.RandomSampler(items, replacement=True)"),
        )
        implementation_paths = tuple(
            sorted((REPOSITORY_ROOT / "src/sebgpt/training").glob("*.py"))
        )
        for path in implementation_paths:
            original = path.read_text(encoding="utf-8")
            for label, expected_category, mutation in mutations:
                with self.subTest(module=path.name, case=label, category=expected_category):
                    self.assertIn(
                        expected_category,
                        _phase8_static_violations(f"{original}\n{mutation}\n"),
                    )

    def test_static_boundary_detector_false_positive_controls(self) -> None:
        safe_cases = (
            "unpadded_tail = True\npad_count = 0\nlayout_note = 'padding is prohibited'",
            "boson_id = 1\ntoken_count = 81\ntoken = '<padding>'",
            "parameter_dtype = 'torch.float32'\nprecision_note = 'full float32 precision'",
            "warmup_steps = 0\nscheduler_name = None\npolicy_note = 'scheduler disabled'",
            "sealed_test_access = 'none'\nsplit = 'train'\nbuild_windows((), None, {}, split)",
            "message = 'test fixtures are synthetic'\nwork = 'The Tempest'",
        )
        for source in safe_cases:
            with self.subTest(source=source):
                self.assertEqual(_phase8_static_violations(source), set())

    def test_contract_gate_oracle_is_literal_sequential_and_distinct(self) -> None:
        content = (
            REPOSITORY_ROOT / "docs/TRAINING_CHECKPOINTING_SPEC.md"
        ).read_text(encoding="utf-8")
        gate_section = content.split("## Proposed Phase 8 gates", 1)[1].split(
            "## Current authorization boundary",
            1,
        )[0]
        gate_numbers = tuple(
            int(line.split(".", 1)[0])
            for line in gate_section.splitlines()
            if line and line[0].isdigit() and "." in line
        )
        self.assertEqual(gate_numbers, tuple(range(1, 95)))
        self.assertEqual(len(set(gate_numbers)), 94)

    def test_exact_record_fields(self) -> None:
        self.assertEqual(len(RECORD_ORACLE), 11)
        for name, (literal_fields, factory_only) in RECORD_ORACLE.items():
            with self.subTest(record=name):
                record_type = getattr(training, name)
                observed_fields = fields(record_type)
                self.assertEqual(
                    tuple(item.name for item in observed_fields),
                    tuple(item[0] for item in literal_fields),
                )
                self.assertEqual(
                    tuple(record_type.__annotations__.values()),
                    tuple(item[1] for item in literal_fields),
                )
                self.assertEqual(
                    tuple(item.repr for item in observed_fields),
                    tuple(item[2] for item in literal_fields),
                )
                self.assertTrue(record_type.__dataclass_params__.frozen)
                self.assertEqual(
                    record_type.__dataclass_params__.init,
                    not factory_only,
                )
                self.assertTrue(record_type.__dataclass_params__.repr)
                instance = object.__new__(record_type)
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, literal_fields[0][0], "mutation")

    def test_configuration_is_exact_and_factory_only(self) -> None:
        with self.assertRaises(training.Phase8ContractError):
            training.Phase8Configuration()
        config = training.load_phase8_configuration()
        self.assertEqual(
            (
                config.schema_version,
                config.context_length,
                config.stride,
                config.logical_batch_capacity,
                config.maximum_epochs,
                config.order_seed,
                config.optimizer_name,
                config.learning_rate,
                config.betas,
                config.epsilon,
                config.weight_decay,
                config.gradient_clip_max_norm,
                config.checkpoint_schema_version,
                config.catalog_schema_version,
            ),
            (
                1,
                256,
                256,
                8,
                10,
                8001,
                "AdamW",
                3e-4,
                (0.9, 0.999),
                1e-8,
                0.01,
                1.0,
                1,
                1,
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            config.context_length = 64  # type: ignore[misc]

    def test_metric_and_training_state_public_construction_is_blocked(self) -> None:
        with self.assertRaises(training.Phase8ContractError):
            training.Phase8MetricState()
        with self.assertRaises(training.Phase8ContractError):
            training.Phase8TrainingState()

    def test_runtime_and_configuration_field_annotations_are_literal(self) -> None:
        self.assertEqual(
            tuple(training.Phase8RuntimeIdentity.__annotations__),
            (
                "schema_version", "python_version", "python_implementation",
                "torch_version", "operating_system", "operating_system_release",
                "machine", "processor", "device", "parameter_dtype", "token_dtype",
                "torch_intra_op_threads", "torch_inter_op_threads",
                "deterministic_algorithms", "deterministic_warn_only",
                "float32_matmul_precision", "default_dtype", "default_device",
                "grad_mode_enabled", "cpu_autocast_enabled", "inference_mode_enabled",
                "mkldnn_enabled", "torch_build_config_sha256",
                "torch_parallel_info_sha256", "requirements_lock_sha256",
            ),
        )
        self.assertEqual(
            next(iter(training.Phase8Configuration.__annotations__)),
            "schema_version",
        )

    def test_exception_surface_is_content_safe(self) -> None:
        classes = (
            training.Phase8TypeError,
            training.Phase8ContractError,
            training.Phase8GovernanceError,
            training.Phase8NumericalError,
            training.Phase8CheckpointError,
            training.Phase8PublicationError,
        )
        for error_type in classes:
            self.assertEqual(error_type.__module__, "sebgpt.training.phase8_types")
            self.assertEqual(
                str(inspect.signature(error_type)),
                "(invariant: 'str', **safe_facts: 'object') -> 'None'",
            )
            error = error_type("phase8.contract.configuration", field="x")
            self.assertEqual(
                dict(error.details),
                {"invariant": "phase8.contract.configuration", "field": "x"},
            )
            self.assertEqual(
                str(error),
                '{"field": "x", "invariant": "phase8.contract.configuration"}',
            )

    def test_public_signatures_are_exact(self) -> None:
        expected_signatures = {
            training.build_phase8_windows: "(works: 'tuple[ExtractedWork, ...]', vocabulary: 'VocabularyBinding', phase_1_manifest: 'Mapping[str, object]', *, split: 'str') -> 'tuple[Phase8Window, ...]'",
            training.configure_phase8_runtime: "() -> 'Phase8RuntimeIdentity'",
            training.create_phase8_epoch_order: "(number_of_windows: 'int', *, epoch: 'int', generator: 'torch.Generator') -> 'tuple[int, ...]'",
            training.create_phase8_optimizer: "(model: 'MiniGPT') -> 'torch.optim.AdamW'",
            training.evaluate_phase8_split: "(model: 'MiniGPT', optimizer: 'torch.optim.AdamW', windows: 'tuple[Phase8Window, ...]', order_generator: 'torch.Generator', progress: 'Phase8Progress', *, split: 'str') -> 'Phase8Evaluation'",
            training.iter_phase8_logical_batches: "(windows: 'tuple[Phase8Window, ...]', order: 'tuple[int, ...]', *, epoch: 'int') -> 'Iterator[Phase8LogicalBatch]'",
            training.load_phase8_checkpoint: "(repository_root: 'Path', *, run_id: 'str', role: \"Literal['latest']\") -> 'Phase8TrainingState'",
            training.load_phase8_configuration: "() -> 'Phase8Configuration'",
            training.run_fixed_phase8_experiment: "(corpus: 'Phase4PermittedCorpus', vocabulary: 'VocabularyBinding', config: 'Phase8Configuration', *, code_commit: 'str') -> 'Phase8RunResult'",
            training.run_phase8_logical_batch: "(model: 'MiniGPT', optimizer: 'torch.optim.AdamW', windows: 'tuple[Phase8Window, ...]', batch: 'Phase8LogicalBatch') -> 'Phase8BatchUpdate'",
            training.save_phase8_checkpoint: "(state: 'Phase8TrainingState', repository_root: 'Path') -> 'Phase8CheckpointReference'",
            training.validate_phase8_runtime: "() -> 'Phase8RuntimeIdentity'",
        }
        for operation, expected in expected_signatures.items():
            signature = inspect.signature(operation)
            self.assertEqual(str(signature), expected)
            self.assertTrue(
                all(
                    parameter.default is inspect.Parameter.empty
                    for parameter in signature.parameters.values()
                )
            )

    def test_window_repr_exposes_only_safe_identity_fields(self) -> None:
        window = training.Phase8Window(
            "hamlet",
            1,
            "train",
            0,
            (1, 2),
            (2, 3),
        )
        self.assertEqual(
            repr(window),
            "Phase8Window(work_id='hamlet', manifest_order=1, split='train', start_index=0)",
        )
        repr_fields = {
            item.name: item.repr for item in fields(training.Phase8Window)
        }
        self.assertEqual(
            repr_fields,
            {
                "work_id": True,
                "manifest_order": True,
                "split": True,
                "start_index": True,
                "input_ids": False,
                "target_ids": False,
            },
        )

    def test_every_runtime_identity_field_mismatch_is_rejected(self) -> None:
        identity = training.validate_phase8_runtime()
        for item in fields(training.Phase8RuntimeIdentity):
            value = getattr(identity, item.name)
            if type(value) is bool:
                replacement = not value
            elif type(value) is int:
                replacement = value + 1
            else:
                replacement = value + "x"
            with self.subTest(field=item.name):
                with self.assertRaises(training.Phase8ContractError) as caught:
                    types_module._validate_runtime_identity(
                        replace(identity, **{item.name: replacement})
                    )
                self.assertEqual(caught.exception.details["field"], item.name)

    def test_runtime_integer_fields_reject_boolean_coercion(self) -> None:
        identity = training.validate_phase8_runtime()
        for name in (
            "schema_version",
            "torch_intra_op_threads",
            "torch_inter_op_threads",
        ):
            with self.subTest(field=name):
                with self.assertRaises(training.Phase8ContractError):
                    types_module._validate_runtime_identity(
                        replace(identity, **{name: True})
                    )

    def test_runtime_identity_and_mismatch_refusal(self) -> None:
        identity = training.validate_phase8_runtime()
        self.assertEqual(identity.schema_version, 1)
        self.assertEqual(identity.python_version, "3.14.4")
        self.assertEqual(identity.torch_version, "2.14.0")
        self.assertEqual(identity.torch_intra_op_threads, 1)
        self.assertEqual(identity.torch_inter_op_threads, 1)
        self.assertFalse(identity.deterministic_warn_only)
        self.assertFalse(identity.cpu_autocast_enabled)
        self.assertFalse(identity.inference_mode_enabled)
        with patch(
            "sebgpt.training.phase8_types.torch.get_float32_matmul_precision",
            return_value="medium",
        ):
            with self.assertRaises(training.Phase8ContractError) as caught:
                training.validate_phase8_runtime()
        self.assertEqual(
            caught.exception.details["invariant"],
            "phase8.contract.runtime",
        )


if __name__ == "__main__":
    unittest.main()
