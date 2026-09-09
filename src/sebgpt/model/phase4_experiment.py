"""Fail-closed orchestration for the fixed Phase 4 experiment."""

from __future__ import annotations

import math
import platform
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import torch

from sebgpt.data.shakespeare_examples import (
    DocumentShiftedExample,
    iter_shakespeare_phase_4_examples,
)
from sebgpt.data.shakespeare_extract import ExtractedWork
from sebgpt.data.shifted_examples import (
    Phase4ContractError,
    Phase4GovernanceError,
    Phase4TypeError,
    ShiftedTokenExample,
)
from sebgpt.model.simple_language_model import (
    SimpleNeuralLanguageModel,
    clear_gradients,
    explicit_cross_entropy,
    manual_sgd_step,
    model_parameter_digest,
    scale_backward_loss,
)
from sebgpt.tokenization.vocabulary_artifact import (
    VocabularyBinding,
    _is_verified_vocabulary_binding,
)


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
EXPERIMENT_ID: Final = "EXP-20260909-01"
MODEL_IMPLEMENTATION_COMMIT: Final = (
    "497ecde3577677903f14669722d61dcdf8caa1d6"
)
TOKENIZER_IMPLEMENTATION_COMMIT: Final = (
    "de7a7f096fbbd8607c944412eaef30be9b686b56"
)
EMBEDDING_DECISION: Final = "DEC-0015"
EMBEDDING_CONTRACT_COMMIT: Final = (
    "0e458cc8b8bce9d8e89b23abbb8ce6385669b7d5"
)
EMBEDDING_IMPLEMENTATION_COMMIT: Final = (
    "68b47bb404d55357cadce35b97036c72c5876d62"
)
PHASE_1_MANIFEST_SHA256: Final = (
    "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"
)
VOCABULARY_ID: Final = "shakespeare-code-point-v1"
VOCABULARY_SCHEMA_VERSION: Final = 1
VOCABULARY_SIZE: Final = 81
VOCABULARY_ARTIFACT_PATH: Final = (
    "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json"
)
VOCABULARY_ARTIFACT_SHA256: Final = (
    "9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e"
)
MODEL_NAME: Final = "SimpleNeuralLanguageModel"
MODEL_ARCHITECTURE: Final = "positionwise_linear_non_attention"
PARAMETER_NAMES: Final = (
    "output_weight",
    "output_bias",
    "representation.token_embeddings",
    "representation.position_embeddings",
)
PARAMETER_SHAPES: Final = (
    (32, 81),
    (81,),
    (81, 32),
    (256, 32),
)
PARAMETER_COUNT: Final = 13_457
EMBEDDING_DIM: Final = 32
MAX_POSITIONS: Final = 256
CONTEXT_LENGTH: Final = 64
STRIDE: Final = 64
EMBEDDING_SEED: Final = 1337
OUTPUT_HEAD_SEED: Final = 4004
LEARNING_RATE: Final = 0.05
TRAINING_PASSES: Final = 2
TRAINING_SOURCE_TOKENS: Final = 792_705
TRAINING_TARGETS: Final = 792_699
TRAINING_EXAMPLES: Final = 12_389
VALIDATION_SOURCE_TOKENS: Final = 98_296
VALIDATION_TARGETS: Final = 98_295
VALIDATION_EXAMPLES: Final = 1_536
TOTAL_UPDATES: Final = 24_778
UNIFORM_BASELINE: Final = math.log(VOCABULARY_SIZE)
SUCCESS_PREDICATE: Final = (
    "final_training_loss < ln(81) and "
    "final_training_loss < initial_training_loss"
)
TRAINING_WORK_ORDER: Final = (
    "hamlet",
    "romeo-and-juliet",
    "macbeth",
    "a-midsummer-nights-dream",
    "much-ado-about-nothing",
    "henry-v",
)
VALIDATION_WORK_ORDER: Final = ("the-tempest",)
_CONFIG_FACTORY_MARKER: Final = object()


def _contract(condition: bool, invariant: str, **safe_facts: object) -> None:
    if not condition:
        raise Phase4ContractError(invariant, **safe_facts)


def _governance(condition: bool, invariant: str, **safe_facts: object) -> None:
    if not condition:
        raise Phase4GovernanceError(invariant, **safe_facts)


def _is_git_commit(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class _RepositoryState:
    commit: str
    clean: bool


def _read_repository_state() -> _RepositoryState:
    commit_result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _contract(
        commit_result.returncode == 0 and status_result.returncode == 0,
        "experiment.repository.readable",
    )
    return _RepositoryState(
        commit=commit_result.stdout.strip(),
        clean=status_result.stdout == "",
    )


@dataclass(frozen=True, init=False)
class Phase4ExperimentConfig:
    """The one immutable pre-registered Phase 4 configuration."""

    experiment_id: str
    code_commit: str
    model_implementation_commit: str
    repository_clean: bool
    tokenizer_implementation_commit: str
    embedding_decision: str
    embedding_contract_commit: str
    embedding_implementation_commit: str
    phase_1_manifest_sha256: str
    vocabulary_id: str
    vocabulary_schema_version: int
    vocabulary_size: int
    vocabulary_artifact_path: str
    vocabulary_artifact_sha256: str
    model_name: str
    model_architecture: str
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    parameter_count: int
    embedding_dim: int
    max_positions: int
    context_length: int
    stride: int
    variable_tail: bool
    padding: bool
    embedding_seed: int
    output_head_seed: int
    embedding_initialization: str
    output_head_initialization: str
    initialization_order: str
    rng_scope: str
    device: str
    dtype: str
    trainable_parameters: tuple[str, ...]
    training_work_order: tuple[str, ...]
    validation_work_order: tuple[str, ...]
    traversal: str
    shuffle: bool
    one_example_at_a_time: bool
    update_rule: str
    gradient_clearing: str
    learning_rate: float
    training_passes: int
    tail_scale: str
    aggregate_rule: str
    training_source_tokens: int
    training_targets: int
    training_examples: int
    validation_source_tokens: int
    validation_targets: int
    validation_examples: int
    total_updates: int
    uniform_baseline: float
    success_predicate: str
    test_access: str
    _factory_marker: object = field(repr=False, compare=False)

    def __new__(cls, *args: object, **kwargs: object) -> Phase4ExperimentConfig:
        raise Phase4ContractError("experiment.config.factory")


def load_phase4_experiment_config() -> Phase4ExperimentConfig:
    """Capture repository identity and return the fixed configuration."""

    state = _read_repository_state()
    config = object.__new__(Phase4ExperimentConfig)
    values: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "code_commit": state.commit,
        "model_implementation_commit": MODEL_IMPLEMENTATION_COMMIT,
        "repository_clean": state.clean,
        "tokenizer_implementation_commit": TOKENIZER_IMPLEMENTATION_COMMIT,
        "embedding_decision": EMBEDDING_DECISION,
        "embedding_contract_commit": EMBEDDING_CONTRACT_COMMIT,
        "embedding_implementation_commit": EMBEDDING_IMPLEMENTATION_COMMIT,
        "phase_1_manifest_sha256": PHASE_1_MANIFEST_SHA256,
        "vocabulary_id": VOCABULARY_ID,
        "vocabulary_schema_version": VOCABULARY_SCHEMA_VERSION,
        "vocabulary_size": VOCABULARY_SIZE,
        "vocabulary_artifact_path": VOCABULARY_ARTIFACT_PATH,
        "vocabulary_artifact_sha256": VOCABULARY_ARTIFACT_SHA256,
        "model_name": MODEL_NAME,
        "model_architecture": MODEL_ARCHITECTURE,
        "parameter_names": PARAMETER_NAMES,
        "parameter_shapes": PARAMETER_SHAPES,
        "parameter_count": PARAMETER_COUNT,
        "embedding_dim": EMBEDDING_DIM,
        "max_positions": MAX_POSITIONS,
        "context_length": CONTEXT_LENGTH,
        "stride": STRIDE,
        "variable_tail": True,
        "padding": False,
        "embedding_seed": EMBEDDING_SEED,
        "output_head_seed": OUTPUT_HEAD_SEED,
        "embedding_initialization": "Normal(0,1/sqrt(32))",
        "output_head_initialization": "Normal(0,1/sqrt(32));zero_bias",
        "initialization_order": "token_then_position_then_output_weight_then_bias",
        "rng_scope": "separate_local_cpu_generators;global_rng_unchanged",
        "device": "cpu",
        "dtype": "torch.float32",
        "trainable_parameters": PARAMETER_NAMES,
        "training_work_order": TRAINING_WORK_ORDER,
        "validation_work_order": VALIDATION_WORK_ORDER,
        "traversal": "manifest_order_then_ascending_start",
        "shuffle": False,
        "one_example_at_a_time": True,
        "update_rule": "manual_sgd_one_update_per_example",
        "gradient_clearing": "parameter.grad=None_before_example_and_after_update",
        "learning_rate": LEARNING_RATE,
        "training_passes": TRAINING_PASSES,
        "tail_scale": "mean_loss_times_L_over_64",
        "aggregate_rule": "target_token_weighted_math_fsum",
        "training_source_tokens": TRAINING_SOURCE_TOKENS,
        "training_targets": TRAINING_TARGETS,
        "training_examples": TRAINING_EXAMPLES,
        "validation_source_tokens": VALIDATION_SOURCE_TOKENS,
        "validation_targets": VALIDATION_TARGETS,
        "validation_examples": VALIDATION_EXAMPLES,
        "total_updates": TOTAL_UPDATES,
        "uniform_baseline": UNIFORM_BASELINE,
        "success_predicate": SUCCESS_PREDICATE,
        "test_access": "none",
        "_factory_marker": _CONFIG_FACTORY_MARKER,
    }
    for name, value in values.items():
        object.__setattr__(config, name, value)
    return config


@dataclass(frozen=True)
class Phase4PermittedCorpus:
    """Metadata-validated seven-work handoff; text remains repr-hidden."""

    phase_1_manifest: Mapping[str, object] = field(repr=False)
    phase_1_manifest_sha256: str
    training_works: tuple[ExtractedWork, ...] = field(repr=False)
    validation_works: tuple[ExtractedWork, ...] = field(repr=False)
    training_source_tokens: int
    training_targets: int
    training_examples: int
    validation_source_tokens: int
    validation_targets: int
    validation_examples: int
    sealed_test_supplier: object | None = field(default=None, repr=False)


@dataclass(frozen=True)
class AggregateMeasurement:
    loss: float
    target_count: int
    example_count: int
    parameter_digest_before: str
    parameter_digest_after: str


@dataclass(frozen=True)
class TrainingPassResult:
    target_count: int
    example_count: int
    update_count: int


@dataclass(frozen=True)
class Phase4ExperimentResult:
    experiment_id: str
    utc_run_at: str
    status: str
    question: str
    code_commit: str
    model_implementation_commit: str
    tokenizer_implementation_commit: str
    vocabulary_id: str
    vocabulary_schema_version: int
    vocabulary_size: int
    vocabulary_artifact_path: str
    vocabulary_artifact_sha256: str
    dataset_id: str
    dataset_manifest_sha256: str
    permitted_work_hashes: tuple[tuple[str, str], ...]
    embedding_decision: str
    embedding_contract_commit: str
    embedding_implementation_commit: str
    model_name: str
    model_architecture: str
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    parameter_count: int
    trainable_parameters: tuple[str, ...]
    weight_tying: bool
    python_version: str
    pytorch_version: str
    operating_system: str
    machine_architecture: str
    device: str
    dtype: str
    embedding_seed: int
    output_head_seed: int
    embedding_initialization: str
    output_head_initialization: str
    initialization_order: str
    rng_scope: str
    context_length: int
    stride: int
    variable_tail: bool
    padding: bool
    traversal: str
    shuffle: bool
    one_example_at_a_time: bool
    update_rule: str
    gradient_clearing: str
    tail_scale: str
    aggregate_rule: str
    learning_rate: float
    training_passes: int
    training_source_token_count: int
    training_target_count: int
    training_example_count: int
    validation_source_token_count: int
    validation_target_count: int
    validation_example_count: int
    update_count: int
    initial_training_loss: float
    initial_validation_loss: float
    final_training_loss: float
    final_validation_loss: float
    uniform_baseline: float
    success_predicate: str
    below_uniform_baseline: bool
    below_initial_training_loss: bool
    passed: bool
    validation_direction: str
    model_constructor_count: int
    model_identity: int
    parameter_identities: tuple[int, ...]
    parameter_identity_stable: bool
    initial_parameter_digest: str
    final_parameter_digest: str
    measurement_digests: tuple[tuple[str, str, str], ...]
    measurement_gradients_none: bool
    validation_observation_only: bool
    sealed_test_accessed: bool
    retained_model_or_data_artifacts: bool
    failed_run_predecessor: str | None
    amended_pre_registration: str | None
    observations: tuple[str, ...]
    conclusion: str
    follow_up: str
    lifecycle_events: tuple[str, ...]


def _preflight_01_config_type(
    config: object,
    vocabulary: object,
    corpus: object,
) -> None:
    del vocabulary, corpus
    if type(config) is not Phase4ExperimentConfig:
        raise Phase4TypeError("experiment.config.exact_type")
    _contract(
        getattr(config, "_factory_marker", None) is _CONFIG_FACTORY_MARKER,
        "experiment.config.factory",
    )


def _preflight_02_authority_types(
    config: Phase4ExperimentConfig,
    vocabulary: object,
    corpus: object,
) -> None:
    del config
    if type(vocabulary) is not VocabularyBinding:
        raise Phase4TypeError("experiment.vocabulary.binding_type")
    if type(corpus) is not Phase4PermittedCorpus:
        raise Phase4TypeError("experiment.corpus.exact_type")
    if type(corpus.training_works) is not tuple:
        raise Phase4TypeError("experiment.corpus.training_works.exact_tuple")
    if type(corpus.validation_works) is not tuple:
        raise Phase4TypeError("experiment.corpus.validation_works.exact_tuple")
    for position, work in enumerate((*corpus.training_works, *corpus.validation_works)):
        if not isinstance(work, ExtractedWork):
            raise Phase4TypeError(
                "experiment.corpus.work.extracted_work",
                position=position,
            )


def _preflight_03_repository(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    _contract(config.experiment_id == EXPERIMENT_ID, "experiment.identity")
    _contract(
        config.model_implementation_commit == MODEL_IMPLEMENTATION_COMMIT,
        "experiment.model.implementation_commit",
    )
    _contract(_is_git_commit(config.code_commit), "experiment.code.commit_grammar")
    _contract(
        type(config.repository_clean) is bool and config.repository_clean,
        "experiment.code.clean_checkpoint",
    )
    current = _read_repository_state()
    _contract(current.clean, "experiment.code.clean_checkpoint")
    _contract(
        current.commit == config.code_commit,
        "experiment.code.checked_out_identity",
    )


def _preflight_04_tokenizer(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    _contract(
        config.tokenizer_implementation_commit == TOKENIZER_IMPLEMENTATION_COMMIT,
        "experiment.tokenizer.implementation_commit",
    )


def _preflight_05_embedding(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    _contract(
        config.embedding_decision == EMBEDDING_DECISION,
        "experiment.embedding.decision",
    )
    _contract(
        config.embedding_contract_commit == EMBEDDING_CONTRACT_COMMIT,
        "experiment.embedding.contract_commit",
    )
    _contract(
        config.embedding_implementation_commit == EMBEDDING_IMPLEMENTATION_COMMIT,
        "experiment.embedding.implementation_commit",
    )


def _preflight_06_vocabulary(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del corpus
    expected = (
        (config.vocabulary_id, VOCABULARY_ID, "id"),
        (config.vocabulary_schema_version, VOCABULARY_SCHEMA_VERSION, "schema"),
        (config.vocabulary_size, VOCABULARY_SIZE, "size"),
        (config.vocabulary_artifact_path, VOCABULARY_ARTIFACT_PATH, "path"),
        (config.vocabulary_artifact_sha256, VOCABULARY_ARTIFACT_SHA256, "sha256"),
        (vocabulary.tokenizer_id, VOCABULARY_ID, "binding_id"),
        (vocabulary.schema_version, VOCABULARY_SCHEMA_VERSION, "binding_schema"),
        (vocabulary.vocabulary_size, VOCABULARY_SIZE, "binding_size"),
        (vocabulary.artifact_sha256, VOCABULARY_ARTIFACT_SHA256, "binding_sha256"),
    )
    for observed, accepted, field_name in expected:
        _contract(
            observed == accepted,
            "experiment.vocabulary.identity",
            field=field_name,
        )
    _contract(
        _is_verified_vocabulary_binding(vocabulary),
        "experiment.vocabulary.factory_verified",
    )


def _preflight_07_dataset(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    _governance(
        config.phase_1_manifest_sha256 == PHASE_1_MANIFEST_SHA256,
        "experiment.dataset.config_manifest_sha256",
    )
    _governance(
        corpus.phase_1_manifest_sha256 == PHASE_1_MANIFEST_SHA256,
        "experiment.dataset.manifest_sha256",
    )
    _governance(
        isinstance(corpus.phase_1_manifest, Mapping),
        "experiment.dataset.manifest_mapping",
    )
    iter_shakespeare_phase_4_examples(
        corpus.training_works,
        vocabulary,
        corpus.phase_1_manifest,
        split="train",
    )
    iter_shakespeare_phase_4_examples(
        corpus.validation_works,
        vocabulary,
        corpus.phase_1_manifest,
        split="validation",
    )


def _preflight_08_architecture(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    checks = (
        (config.model_name == MODEL_NAME, "name"),
        (config.model_architecture == MODEL_ARCHITECTURE, "architecture"),
        (config.parameter_names == PARAMETER_NAMES, "parameter_names"),
        (config.parameter_shapes == PARAMETER_SHAPES, "parameter_shapes"),
        (config.parameter_count == PARAMETER_COUNT, "parameter_count"),
        (config.embedding_dim == EMBEDDING_DIM, "embedding_dim"),
        (config.max_positions == MAX_POSITIONS, "max_positions"),
        (config.device == "cpu", "device"),
        (config.dtype == "torch.float32", "dtype"),
    )
    for condition, field_name in checks:
        _contract(condition, "experiment.model.configuration", field=field_name)


def _preflight_09_seeds(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    _contract(
        config.embedding_seed == EMBEDDING_SEED,
        "experiment.seed.embedding",
    )
    _contract(
        config.output_head_seed == OUTPUT_HEAD_SEED,
        "experiment.seed.output_head",
    )
    initialization_checks = (
        (config.embedding_initialization == "Normal(0,1/sqrt(32))", "embedding"),
        (
            config.output_head_initialization == "Normal(0,1/sqrt(32));zero_bias",
            "output_head",
        ),
        (
            config.initialization_order
            == "token_then_position_then_output_weight_then_bias",
            "order",
        ),
        (
            config.rng_scope
            == "separate_local_cpu_generators;global_rng_unchanged",
            "rng_scope",
        ),
    )
    for condition, field_name in initialization_checks:
        _contract(
            condition,
            "experiment.initialization.configuration",
            field=field_name,
        )


def _preflight_10_examples(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    checks = (
        (config.context_length == CONTEXT_LENGTH, "context_length"),
        (config.stride == STRIDE, "stride"),
        (config.variable_tail is True, "variable_tail"),
        (config.padding is False, "padding"),
    )
    for condition, field_name in checks:
        _contract(condition, "experiment.examples.configuration", field=field_name)


def _preflight_11_learning_rate(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    if type(config.learning_rate) is not float:
        raise Phase4TypeError("sgd.learning_rate.exact_float")
    _contract(
        math.isfinite(config.learning_rate)
        and config.learning_rate == LEARNING_RATE,
        "sgd.learning_rate.value",
    )


def _preflight_12_training_policy(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    checks = (
        (config.training_passes == TRAINING_PASSES, "training_passes"),
        (config.training_work_order == TRAINING_WORK_ORDER, "training_work_order"),
        (config.validation_work_order == VALIDATION_WORK_ORDER, "validation_work_order"),
        (
            config.traversal == "manifest_order_then_ascending_start",
            "traversal",
        ),
        (config.shuffle is False, "shuffle"),
        (config.one_example_at_a_time is True, "one_example_at_a_time"),
        (
            config.trainable_parameters == PARAMETER_NAMES,
            "trainable_parameters",
        ),
        (config.update_rule == "manual_sgd_one_update_per_example", "update_rule"),
        (
            config.gradient_clearing
            == "parameter.grad=None_before_example_and_after_update",
            "gradient_clearing",
        ),
        (config.tail_scale == "mean_loss_times_L_over_64", "tail_scale"),
        (config.aggregate_rule == "target_token_weighted_math_fsum", "aggregate_rule"),
    )
    for condition, field_name in checks:
        _contract(condition, "experiment.training.configuration", field=field_name)


def _preflight_13_counts(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary
    config_counts = (
        (config.training_source_tokens, TRAINING_SOURCE_TOKENS, "training_source_tokens"),
        (config.training_targets, TRAINING_TARGETS, "training_targets"),
        (config.training_examples, TRAINING_EXAMPLES, "training_examples"),
        (
            config.validation_source_tokens,
            VALIDATION_SOURCE_TOKENS,
            "validation_source_tokens",
        ),
        (config.validation_targets, VALIDATION_TARGETS, "validation_targets"),
        (config.validation_examples, VALIDATION_EXAMPLES, "validation_examples"),
        (config.total_updates, TOTAL_UPDATES, "total_updates"),
    )
    for observed, expected, field_name in config_counts:
        _contract(
            type(observed) is int and observed == expected,
            "experiment.counts.configuration",
            field=field_name,
        )
    corpus_counts = (
        (corpus.training_source_tokens, TRAINING_SOURCE_TOKENS, "training_source_tokens"),
        (corpus.training_targets, TRAINING_TARGETS, "training_targets"),
        (corpus.training_examples, TRAINING_EXAMPLES, "training_examples"),
        (
            corpus.validation_source_tokens,
            VALIDATION_SOURCE_TOKENS,
            "validation_source_tokens",
        ),
        (corpus.validation_targets, VALIDATION_TARGETS, "validation_targets"),
        (corpus.validation_examples, VALIDATION_EXAMPLES, "validation_examples"),
    )
    for observed, expected, field_name in corpus_counts:
        _governance(
            type(observed) is int and observed == expected,
            "experiment.counts.corpus",
            field=field_name,
        )


def _preflight_14_success(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary, corpus
    _contract(
        config.uniform_baseline == UNIFORM_BASELINE,
        "experiment.success.uniform_baseline",
    )
    _contract(
        config.success_predicate == SUCCESS_PREDICATE,
        "experiment.success.predicate",
    )


def _preflight_15_sealed_test(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    del vocabulary
    _governance(config.test_access == "none", "experiment.sealed_test.config")
    _governance(
        corpus.sealed_test_supplier is None,
        "experiment.sealed_test.supplier",
    )
    _governance(
        len(corpus.training_works) == 6 and len(corpus.validation_works) == 1,
        "experiment.sealed_test.seven_permitted_works",
    )


_PREFLIGHT_STAGES: Final = (
    _preflight_01_config_type,
    _preflight_02_authority_types,
    _preflight_03_repository,
    _preflight_04_tokenizer,
    _preflight_05_embedding,
    _preflight_06_vocabulary,
    _preflight_07_dataset,
    _preflight_08_architecture,
    _preflight_09_seeds,
    _preflight_10_examples,
    _preflight_11_learning_rate,
    _preflight_12_training_policy,
    _preflight_13_counts,
    _preflight_14_success,
    _preflight_15_sealed_test,
)


def validate_phase4_experiment_preflight(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> None:
    """Run the accepted fifteen stages before model construction or traversal."""

    for stage in _PREFLIGHT_STAGES:
        stage(config, vocabulary, corpus)


def _validate_training_example(
    example: object,
    *,
    previous_order_key: tuple[int, int] | None,
) -> tuple[int, int]:
    """Validate training governance and order before token tensorization."""

    if type(example) is not DocumentShiftedExample:
        raise Phase4TypeError("experiment.example.exact_document_shifted_example")
    if type(example.example) is not ShiftedTokenExample:
        raise Phase4TypeError("experiment.example.exact_shifted_token_example")
    _governance(example.split == "train", "training.example.split")
    _governance(
        type(example.manifest_order) is int
        and 1 <= example.manifest_order <= len(TRAINING_WORK_ORDER),
        "training.example.manifest_order",
    )
    _governance(
        example.work_id == TRAINING_WORK_ORDER[example.manifest_order - 1],
        "training.example.work_id",
    )
    _governance(
        type(example.example.start_index) is int
        and example.example.start_index >= 0
        and example.example.start_index % STRIDE == 0,
        "training.example.start",
    )
    order_key = (example.manifest_order, example.example.start_index)
    _governance(
        previous_order_key is None or order_key > previous_order_key,
        "training.example.order",
    )
    return order_key


def _tensorize(example: DocumentShiftedExample) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(example.example.input_ids, dtype=torch.long, device="cpu")
    targets = torch.tensor(example.example.target_ids, dtype=torch.long, device="cpu")
    return input_ids, targets


def measure_aggregate_loss(
    model: SimpleNeuralLanguageModel,
    examples: Iterable[DocumentShiftedExample],
) -> AggregateMeasurement:
    """Measure one target-token-weighted aggregate without mutation."""

    if not isinstance(examples, Iterable):
        raise Phase4TypeError("aggregate.examples.iterable")
    items = tuple(model.named_parameters())
    identities = tuple(id(parameter) for _, parameter in items)
    _contract(
        all(parameter.grad is None for _, parameter in items),
        "aggregate.gradients.none_before",
    )
    before_digest = model_parameter_digest(model)
    target_count = 0
    example_count = 0

    def weighted_losses() -> Iterable[float]:
        nonlocal target_count, example_count
        with torch.no_grad():
            for document_example in examples:
                input_ids, targets = _tensorize(document_example)
                length = targets.shape[0]
                mean_loss = explicit_cross_entropy(model(input_ids), targets)
                target_count += length
                example_count += 1
                yield float(mean_loss.item()) * length

    total_negative_log_likelihood = math.fsum(weighted_losses())
    _contract(target_count > 0, "aggregate.target_count.positive")
    aggregate_loss = total_negative_log_likelihood / target_count
    _contract(math.isfinite(aggregate_loss), "aggregate.loss.finite")
    after_items = tuple(model.named_parameters())
    _contract(
        tuple(id(parameter) for _, parameter in after_items) == identities,
        "aggregate.parameter_identity",
    )
    after_digest = model_parameter_digest(model)
    _contract(after_digest == before_digest, "aggregate.parameter_digest")
    _contract(
        all(parameter.grad is None for _, parameter in after_items),
        "aggregate.gradients.none_after",
    )
    return AggregateMeasurement(
        loss=aggregate_loss,
        target_count=target_count,
        example_count=example_count,
        parameter_digest_before=before_digest,
        parameter_digest_after=after_digest,
    )


def run_training_pass(
    model: SimpleNeuralLanguageModel,
    examples: Iterable[DocumentShiftedExample],
) -> TrainingPassResult:
    """Perform one deterministic fixed-policy pass over supplied examples."""

    if not isinstance(examples, Iterable):
        raise Phase4TypeError("training.examples.iterable")
    target_count = 0
    example_count = 0
    update_count = 0
    previous_order_key: tuple[int, int] | None = None
    for document_example in examples:
        previous_order_key = _validate_training_example(
            document_example,
            previous_order_key=previous_order_key,
        )
        input_ids, targets = _tensorize(document_example)
        clear_gradients(model)
        mean_loss = explicit_cross_entropy(model(input_ids), targets)
        scaled_loss = scale_backward_loss(
            mean_loss,
            example_length=targets.shape[0],
            context_length=CONTEXT_LENGTH,
        )
        scaled_loss.backward()
        manual_sgd_step(model, learning_rate=LEARNING_RATE)
        clear_gradients(model)
        target_count += targets.shape[0]
        example_count += 1
        update_count += 1
    _contract(
        all(parameter.grad is None for parameter in model.parameters()),
        "training.gradients.none_after",
    )
    return TrainingPassResult(
        target_count=target_count,
        example_count=example_count,
        update_count=update_count,
    )


def _examples(
    corpus: Phase4PermittedCorpus,
    vocabulary: VocabularyBinding,
    *,
    split: str,
) -> Iterable[DocumentShiftedExample]:
    works = corpus.training_works if split == "train" else corpus.validation_works
    return iter_shakespeare_phase_4_examples(
        works,
        vocabulary,
        corpus.phase_1_manifest,
        split=split,
    )


def _verify_model_boundary(
    model: SimpleNeuralLanguageModel,
    *,
    model_identity: int,
    parameter_identities: tuple[int, ...],
) -> None:
    _contract(id(model) == model_identity, "experiment.lifecycle.model_identity")
    _contract(
        tuple(id(parameter) for parameter in model.parameters())
        == parameter_identities,
        "experiment.lifecycle.parameter_identity",
    )
    _contract(
        all(parameter.grad is None for parameter in model.parameters()),
        "experiment.lifecycle.gradients_none",
    )


def _require_measurement_counts(
    measurement: AggregateMeasurement,
    *,
    target_count: int,
    example_count: int,
    role: str,
) -> None:
    _governance(
        measurement.target_count == target_count,
        "experiment.measurement.target_count",
        role=role,
    )
    _governance(
        measurement.example_count == example_count,
        "experiment.measurement.example_count",
        role=role,
    )


def _require_pass_counts(result: TrainingPassResult, *, pass_number: int) -> None:
    _governance(
        result.target_count == TRAINING_TARGETS,
        "experiment.training.target_count",
        pass_number=pass_number,
    )
    _governance(
        result.example_count == TRAINING_EXAMPLES,
        "experiment.training.example_count",
        pass_number=pass_number,
    )
    _contract(
        result.update_count == TRAINING_EXAMPLES,
        "experiment.training.update_count",
        pass_number=pass_number,
    )


def _utc_run_at() -> str:
    """Return completion provenance as an explicit second-resolution UTC value."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def _result_narrative(
    *,
    passed: bool,
    below_baseline: bool,
    below_initial: bool,
    validation_direction: str,
) -> tuple[tuple[str, ...], str, str]:
    observations = (
        f"Final training loss below ln(81): {str(below_baseline).lower()}.",
        f"Final training loss below initial training loss: {str(below_initial).lower()}.",
        (
            f"Validation loss {validation_direction}; this is an observation "
            "and is not part of the primary success predicate."
        ),
        "No sealed-test metric exists and no model or data artifact was retained.",
    )
    if passed:
        conclusion = (
            "The fixed run met its pre-registered training-loss predicate. "
            "This does not establish generalization or test performance."
        )
        follow_up = (
            "Proceed only to Gate 25 independent experiment and Phase 4 exit "
            "review; do not begin Phase 5."
        )
    else:
        conclusion = (
            "The fixed run did not meet its pre-registered training-loss "
            "predicate. Preserve the result; it supports no tuning, retry, "
            "generalization, or test-performance claim."
        )
        follow_up = (
            "Proceed only through the separately authorized controlled Gate 26 "
            "failure-analysis path; do not tune or rerun."
        )
    return observations, conclusion, follow_up


def run_fixed_phase4_experiment(
    config: Phase4ExperimentConfig,
    vocabulary: VocabularyBinding,
    corpus: Phase4PermittedCorpus,
) -> Phase4ExperimentResult:
    """Execute the single fixed Phase 4 lifecycle after complete preflight."""

    validate_phase4_experiment_preflight(config, vocabulary, corpus)
    model = SimpleNeuralLanguageModel(
        vocabulary,
        embedding_dim=EMBEDDING_DIM,
        max_positions=MAX_POSITIONS,
        embedding_seed=EMBEDDING_SEED,
        output_head_seed=OUTPUT_HEAD_SEED,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model_identity = id(model)
    parameter_identities = tuple(id(parameter) for parameter in model.parameters())
    _contract(
        len(parameter_identities) == 4,
        "experiment.lifecycle.parameter_count",
    )
    _verify_model_boundary(
        model,
        model_identity=model_identity,
        parameter_identities=parameter_identities,
    )
    initial_digest = model_parameter_digest(model)
    events: list[str] = ["model_constructed"]

    initial_training = measure_aggregate_loss(
        model,
        _examples(corpus, vocabulary, split="train"),
    )
    _require_measurement_counts(
        initial_training,
        target_count=TRAINING_TARGETS,
        example_count=TRAINING_EXAMPLES,
        role="initial_training",
    )
    events.append("initial_training_measured")
    _verify_model_boundary(
        model,
        model_identity=model_identity,
        parameter_identities=parameter_identities,
    )

    initial_validation = measure_aggregate_loss(
        model,
        _examples(corpus, vocabulary, split="validation"),
    )
    _require_measurement_counts(
        initial_validation,
        target_count=VALIDATION_TARGETS,
        example_count=VALIDATION_EXAMPLES,
        role="initial_validation",
    )
    events.append("initial_validation_measured")
    _verify_model_boundary(
        model,
        model_identity=model_identity,
        parameter_identities=parameter_identities,
    )

    pass_results: list[TrainingPassResult] = []
    for pass_number in range(1, TRAINING_PASSES + 1):
        pass_result = run_training_pass(
            model,
            _examples(corpus, vocabulary, split="train"),
        )
        _require_pass_counts(pass_result, pass_number=pass_number)
        pass_results.append(pass_result)
        events.append(f"training_pass_{pass_number}_completed")
        _verify_model_boundary(
            model,
            model_identity=model_identity,
            parameter_identities=parameter_identities,
        )
    total_updates = sum(result.update_count for result in pass_results)
    _contract(total_updates == TOTAL_UPDATES, "experiment.training.total_updates")

    clear_gradients(model)
    final_training = measure_aggregate_loss(
        model,
        _examples(corpus, vocabulary, split="train"),
    )
    _require_measurement_counts(
        final_training,
        target_count=TRAINING_TARGETS,
        example_count=TRAINING_EXAMPLES,
        role="final_training",
    )
    events.append("final_training_measured")
    _verify_model_boundary(
        model,
        model_identity=model_identity,
        parameter_identities=parameter_identities,
    )

    final_validation = measure_aggregate_loss(
        model,
        _examples(corpus, vocabulary, split="validation"),
    )
    _require_measurement_counts(
        final_validation,
        target_count=VALIDATION_TARGETS,
        example_count=VALIDATION_EXAMPLES,
        role="final_validation",
    )
    events.append("final_validation_measured")
    _verify_model_boundary(
        model,
        model_identity=model_identity,
        parameter_identities=parameter_identities,
    )
    final_digest = model_parameter_digest(model)

    below_baseline = final_training.loss < UNIFORM_BASELINE
    below_initial = final_training.loss < initial_training.loss
    passed = below_baseline and below_initial
    if final_validation.loss < initial_validation.loss:
        validation_direction = "improved"
    elif final_validation.loss > initial_validation.loss:
        validation_direction = "worsened"
    else:
        validation_direction = "unchanged"
    observations, conclusion, follow_up = _result_narrative(
        passed=passed,
        below_baseline=below_baseline,
        below_initial=below_initial,
        validation_direction=validation_direction,
    )
    permitted_work_hashes = tuple(
        (work.work_id, work.processed.sha256)
        for work in (*corpus.training_works, *corpus.validation_works)
    )
    measurement_digests = (
        (
            "initial_training",
            initial_training.parameter_digest_before,
            initial_training.parameter_digest_after,
        ),
        (
            "initial_validation",
            initial_validation.parameter_digest_before,
            initial_validation.parameter_digest_after,
        ),
        (
            "final_training",
            final_training.parameter_digest_before,
            final_training.parameter_digest_after,
        ),
        (
            "final_validation",
            final_validation.parameter_digest_before,
            final_validation.parameter_digest_after,
        ),
    )
    return Phase4ExperimentResult(
        experiment_id=config.experiment_id,
        utc_run_at=_utc_run_at(),
        status="completed",
        question=(
            "Does the fixed two-pass positionwise baseline lower aggregate "
            "training loss below both ln(81) and its initialized value?"
        ),
        code_commit=config.code_commit,
        model_implementation_commit=config.model_implementation_commit,
        tokenizer_implementation_commit=config.tokenizer_implementation_commit,
        vocabulary_id=config.vocabulary_id,
        vocabulary_schema_version=config.vocabulary_schema_version,
        vocabulary_size=config.vocabulary_size,
        vocabulary_artifact_path=config.vocabulary_artifact_path,
        vocabulary_artifact_sha256=config.vocabulary_artifact_sha256,
        dataset_id="shakespeare-eight-play",
        dataset_manifest_sha256=config.phase_1_manifest_sha256,
        permitted_work_hashes=permitted_work_hashes,
        embedding_decision=config.embedding_decision,
        embedding_contract_commit=config.embedding_contract_commit,
        embedding_implementation_commit=config.embedding_implementation_commit,
        model_name=config.model_name,
        model_architecture=config.model_architecture,
        parameter_names=config.parameter_names,
        parameter_shapes=config.parameter_shapes,
        parameter_count=config.parameter_count,
        trainable_parameters=config.trainable_parameters,
        weight_tying=False,
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        operating_system=platform.platform(),
        machine_architecture=platform.machine(),
        device=config.device,
        dtype=config.dtype,
        embedding_seed=config.embedding_seed,
        output_head_seed=config.output_head_seed,
        embedding_initialization=config.embedding_initialization,
        output_head_initialization=config.output_head_initialization,
        initialization_order=config.initialization_order,
        rng_scope=config.rng_scope,
        context_length=config.context_length,
        stride=config.stride,
        variable_tail=config.variable_tail,
        padding=config.padding,
        traversal=config.traversal,
        shuffle=config.shuffle,
        one_example_at_a_time=config.one_example_at_a_time,
        update_rule=config.update_rule,
        gradient_clearing=config.gradient_clearing,
        tail_scale=config.tail_scale,
        aggregate_rule=config.aggregate_rule,
        learning_rate=config.learning_rate,
        training_passes=config.training_passes,
        training_source_token_count=TRAINING_SOURCE_TOKENS,
        training_target_count=TRAINING_TARGETS,
        training_example_count=TRAINING_EXAMPLES,
        validation_source_token_count=VALIDATION_SOURCE_TOKENS,
        validation_target_count=VALIDATION_TARGETS,
        validation_example_count=VALIDATION_EXAMPLES,
        update_count=total_updates,
        initial_training_loss=initial_training.loss,
        initial_validation_loss=initial_validation.loss,
        final_training_loss=final_training.loss,
        final_validation_loss=final_validation.loss,
        uniform_baseline=UNIFORM_BASELINE,
        success_predicate=config.success_predicate,
        below_uniform_baseline=below_baseline,
        below_initial_training_loss=below_initial,
        passed=passed,
        validation_direction=validation_direction,
        model_constructor_count=1,
        model_identity=model_identity,
        parameter_identities=parameter_identities,
        parameter_identity_stable=True,
        initial_parameter_digest=initial_digest,
        final_parameter_digest=final_digest,
        measurement_digests=measurement_digests,
        measurement_gradients_none=True,
        validation_observation_only=True,
        sealed_test_accessed=False,
        retained_model_or_data_artifacts=False,
        failed_run_predecessor=None,
        amended_pre_registration=None,
        observations=observations,
        conclusion=conclusion,
        follow_up=follow_up,
        lifecycle_events=tuple(events),
    )


__all__ = [
    "AggregateMeasurement",
    "Phase4ExperimentConfig",
    "Phase4ExperimentResult",
    "Phase4PermittedCorpus",
    "TrainingPassResult",
    "load_phase4_experiment_config",
    "measure_aggregate_loss",
    "run_fixed_phase4_experiment",
    "run_training_pass",
    "validate_phase4_experiment_preflight",
]
