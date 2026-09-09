"""Independent synthetic tests for fixed Phase 4 experiment orchestration."""

from __future__ import annotations

import hashlib
import inspect
import math
import sys
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.nn import functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.model.phase4_experiment as experiment_module  # noqa: E402
from sebgpt.data.shakespeare_examples import DocumentShiftedExample  # noqa: E402
from sebgpt.data.shakespeare_extract import ByteStage, ExtractedWork  # noqa: E402
from sebgpt.data.shifted_examples import (  # noqa: E402
    Phase4ContractError,
    Phase4GovernanceError,
    Phase4TypeError,
    ShiftedTokenExample,
)
from sebgpt.model.phase4_experiment import (  # noqa: E402
    AggregateMeasurement,
    Phase4ExperimentConfig,
    Phase4ExperimentResult,
    Phase4PermittedCorpus,
    TrainingPassResult,
    load_phase4_experiment_config,
    measure_aggregate_loss,
    run_fixed_phase4_experiment,
    run_training_pass,
    validate_phase4_experiment_preflight,
)
from sebgpt.model.simple_language_model import (  # noqa: E402
    SimpleNeuralLanguageModel,
    model_parameter_digest,
)
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)


TEST_RUNNER_COMMIT = "a" * 40
TEST_MODEL_IMPLEMENTATION_COMMIT = (
    "497ecde3577677903f14669722d61dcdf8caa1d6"
)
TEST_MANIFEST_SHA256 = (
    "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"
)
TRAINING_IDENTITIES = (
    (1, "hamlet"),
    (2, "romeo-and-juliet"),
    (3, "macbeth"),
    (4, "a-midsummer-nights-dream"),
    (5, "much-ado-about-nothing"),
    (6, "henry-v"),
)
VALIDATION_IDENTITY = (7, "the-tempest")
TRAINING_TEXTS = ("A\n", "B\n", "C\n", "D\n", "E\n", "F\n")
VALIDATION_TEXT = "G\n"


def _stage(text: str) -> ByteStage:
    content = text.encode("utf-8")
    return ByteStage(hashlib.sha256(content).hexdigest(), len(content))


class GuardedExtractedWork(ExtractedWork):
    def __getattribute__(self, name: str) -> Any:
        if name == "processed_text":
            count = object.__getattribute__(self, "_text_access_count")
            object.__setattr__(self, "_text_access_count", count + 1)
            raise AssertionError("content must not be accessed")
        return super().__getattribute__(name)

    @property
    def text_access_count(self) -> int:
        return object.__getattribute__(self, "_text_access_count")


def _work(
    order: int,
    work_id: str,
    split: str,
    text: str,
    *,
    guarded: bool = False,
) -> ExtractedWork:
    work_type = GuardedExtractedWork if guarded else ExtractedWork
    work = work_type(
        work_id=work_id,
        title="Synthetic",
        split=split,
        manifest_order=order,
        body_marker="Synthetic",
        successor_marker="Synthetic successor",
        processed_text=text,
        outer_range=None,  # type: ignore[arg-type]
        successor_position=None,  # type: ignore[arg-type]
        local_contents_position=None,  # type: ignore[arg-type]
        dramatis_position=None,  # type: ignore[arg-type]
        outer_raw=_stage(text),
        retained_raw=_stage(text),
        processed=_stage(text),
        removed_local_contents_byte_count=0,
        processed_code_point_count=len(text),
        processed_line_count=text.count("\n"),
        processed_word_count=len(text.split()),
    )
    if guarded:
        object.__setattr__(work, "_text_access_count", 0)
    return work


def _manifest() -> dict[str, Any]:
    records = (
        *((order, work_id, "train", text) for (order, work_id), text in zip(
            TRAINING_IDENTITIES,
            TRAINING_TEXTS,
            strict=True,
        )),
        (*VALIDATION_IDENTITY, "validation", VALIDATION_TEXT),
    )
    return {
        "schema_version": 4,
        "dataset_id": "shakespeare-eight-play",
        "works": [
            {"order": order, "work_id": work_id, "split": split}
            for order, work_id, split, _ in records
        ],
        "processing": {
            "per_work_results": [
                {
                    "work_id": work_id,
                    "manifest_order": order,
                    "split": split,
                    "normalization": "crlf_to_lf_only",
                    "processed": {
                        "sha256": _stage(text).sha256,
                        "byte_count": _stage(text).byte_count,
                        "code_point_count": len(text),
                    },
                }
                for order, work_id, split, text in records
            ]
        },
    }


def _corpus(*, guarded: bool = False, sealed_supplier: object | None = None):
    training = tuple(
        _work(order, work_id, "train", text, guarded=guarded)
        for (order, work_id), text in zip(
            TRAINING_IDENTITIES,
            TRAINING_TEXTS,
            strict=True,
        )
    )
    validation = (
        _work(
            *VALIDATION_IDENTITY,
            "validation",
            VALIDATION_TEXT,
            guarded=guarded,
        ),
    )
    return Phase4PermittedCorpus(
        phase_1_manifest=_manifest(),
        phase_1_manifest_sha256=TEST_MANIFEST_SHA256,
        training_works=training,
        validation_works=validation,
        training_source_tokens=792_705,
        training_targets=792_699,
        training_examples=12_389,
        validation_source_tokens=98_296,
        validation_targets=98_295,
        validation_examples=1_536,
        sealed_test_supplier=sealed_supplier,
    )


def _document_example(
    input_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
    *,
    start: int = 0,
    work_id: str = "hamlet",
    manifest_order: int = 1,
    split: str = "train",
) -> DocumentShiftedExample:
    return DocumentShiftedExample(
        work_id=work_id,
        manifest_order=manifest_order,
        split=split,
        example=ShiftedTokenExample(start, input_ids, target_ids),
    )


class Phase4ExperimentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)

    def config(self) -> Phase4ExperimentConfig:
        state = experiment_module._RepositoryState(TEST_RUNNER_COMMIT, True)
        with patch.object(experiment_module, "_read_repository_state", return_value=state):
            return load_phase4_experiment_config()

    def model(self) -> SimpleNeuralLanguageModel:
        return SimpleNeuralLanguageModel(
            self.vocabulary,
            embedding_dim=32,
            max_positions=256,
            embedding_seed=1337,
            output_head_seed=4004,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def valid_preflight(self, config=None, corpus=None) -> None:
        selected_config = self.config() if config is None else config
        selected_corpus = _corpus() if corpus is None else corpus
        state = experiment_module._RepositoryState(TEST_RUNNER_COMMIT, True)
        with patch.object(experiment_module, "_read_repository_state", return_value=state):
            validate_phase4_experiment_preflight(
                selected_config,
                self.vocabulary,
                selected_corpus,
            )


class FrozenConfigurationAndPreflightTests(Phase4ExperimentTestCase):
    def test_config_is_factory_created_frozen_and_has_no_runtime_knobs(self) -> None:
        config = self.config()

        with self.assertRaises(Phase4ContractError):
            Phase4ExperimentConfig()
        with self.assertRaises(FrozenInstanceError):
            config.learning_rate = 0.1  # type: ignore[misc]
        self.assertEqual(inspect.signature(load_phase4_experiment_config).parameters, {})
        self.assertEqual(config.learning_rate, 0.05)
        self.assertEqual(config.training_passes, 2)
        self.assertEqual(config.context_length, 64)
        self.assertEqual(config.stride, 64)
        self.assertEqual(config.uniform_baseline, math.log(81))

    def test_valid_synthetic_authorities_complete_all_preflight_stages(self) -> None:
        self.valid_preflight()

    def test_model_implementation_authority_is_pinned_and_distinct(self) -> None:
        config = self.config()

        self.assertEqual(
            config.model_implementation_commit,
            TEST_MODEL_IMPLEMENTATION_COMMIT,
        )
        self.assertEqual(config.code_commit, TEST_RUNNER_COMMIT)
        self.assertNotEqual(
            config.code_commit,
            config.model_implementation_commit,
        )
        self.valid_preflight(config=config)

        object.__setattr__(config, "model_implementation_commit", "b" * 40)
        with self.assertRaises(Phase4ContractError) as raised:
            self.valid_preflight(config=config)
        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.model.implementation_commit",
        )

    def test_preflight_stage_sequence_is_exactly_one_through_fifteen(self) -> None:
        expected = tuple(
            f"_preflight_{position:02d}_{suffix}"
            for position, suffix in (
                (1, "config_type"),
                (2, "authority_types"),
                (3, "repository"),
                (4, "tokenizer"),
                (5, "embedding"),
                (6, "vocabulary"),
                (7, "dataset"),
                (8, "architecture"),
                (9, "seeds"),
                (10, "examples"),
                (11, "learning_rate"),
                (12, "training_policy"),
                (13, "counts"),
                (14, "success"),
                (15, "sealed_test"),
            )
        )

        self.assertEqual(
            tuple(stage.__name__ for stage in experiment_module._PREFLIGHT_STAGES),
            expected,
        )

    def test_wrong_config_type_is_the_first_failure(self) -> None:
        with self.assertRaises(Phase4TypeError) as raised:
            validate_phase4_experiment_preflight(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.config.exact_type",
        )

    def test_altered_frozen_settings_fail_at_their_contract_stages(self) -> None:
        cases = (
            ("experiment_id", "EXP-OTHER", "experiment.identity"),
            ("tokenizer_implementation_commit", "b" * 40, "experiment.tokenizer.implementation_commit"),
            ("embedding_decision", "DEC-9999", "experiment.embedding.decision"),
            ("vocabulary_size", 82, "experiment.vocabulary.identity"),
            ("model_architecture", "attention", "experiment.model.configuration"),
            ("embedding_seed", 7, "experiment.seed.embedding"),
            ("rng_scope", "global", "experiment.initialization.configuration"),
            ("context_length", 63, "experiment.examples.configuration"),
            ("learning_rate", 0.1, "sgd.learning_rate.value"),
            ("training_passes", 3, "experiment.training.configuration"),
            ("gradient_clearing", "zero", "experiment.training.configuration"),
            ("training_targets", 1, "experiment.counts.configuration"),
            ("uniform_baseline", 0.0, "experiment.success.uniform_baseline"),
        )
        for name, value, invariant in cases:
            with self.subTest(name=name):
                config = self.config()
                object.__setattr__(config, name, value)
                with self.assertRaises(Phase4ContractError) as raised:
                    self.valid_preflight(config=config)
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_altered_test_access_setting_has_governance_ownership(self) -> None:
        config = self.config()
        object.__setattr__(config, "test_access", "allowed")

        with self.assertRaises(Phase4GovernanceError) as raised:
            self.valid_preflight(config=config)

        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.sealed_test.config",
        )

    def test_learning_rate_exact_type_owns_bool_failure(self) -> None:
        config = self.config()
        object.__setattr__(config, "learning_rate", True)

        with self.assertRaises(Phase4TypeError) as raised:
            self.valid_preflight(config=config)

        self.assertEqual(raised.exception.details["invariant"], "sgd.learning_rate.exact_float")

    def test_simultaneous_defects_obey_preflight_stage_priority(self) -> None:
        config = self.config()
        object.__setattr__(config, "tokenizer_implementation_commit", "b" * 40)
        object.__setattr__(config, "embedding_decision", "DEC-9999")
        with self.assertRaises(Phase4ContractError) as earlier:
            self.valid_preflight(config=config)
        self.assertEqual(
            earlier.exception.details["invariant"],
            "experiment.tokenizer.implementation_commit",
        )

        config = self.config()
        object.__setattr__(config, "uniform_baseline", 0.0)
        with self.assertRaises(Phase4ContractError) as later:
            self.valid_preflight(
                config=config,
                corpus=_corpus(sealed_supplier=object()),
            )
        self.assertEqual(
            later.exception.details["invariant"],
            "experiment.success.uniform_baseline",
        )

    def test_repository_identity_and_cleanliness_are_rechecked(self) -> None:
        config = self.config()
        cases = (
            (
                experiment_module._RepositoryState("b" * 40, True),
                "experiment.code.checked_out_identity",
            ),
            (
                experiment_module._RepositoryState(TEST_RUNNER_COMMIT, False),
                "experiment.code.clean_checkpoint",
            ),
        )
        for state, invariant in cases:
            with self.subTest(invariant=invariant):
                with patch.object(
                    experiment_module,
                    "_read_repository_state",
                    return_value=state,
                ):
                    with self.assertRaises(Phase4ContractError) as raised:
                        validate_phase4_experiment_preflight(
                            config,
                            self.vocabulary,
                            _corpus(),
                        )
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_wrong_corpus_counts_fail_before_content_or_model_construction(self) -> None:
        config = self.config()
        corpus = replace(_corpus(guarded=True), training_targets=1)
        state = experiment_module._RepositoryState(TEST_RUNNER_COMMIT, True)

        with patch.object(experiment_module, "_read_repository_state", return_value=state):
            with patch.object(experiment_module, "SimpleNeuralLanguageModel") as model_type:
                with self.assertRaises(Phase4GovernanceError) as raised:
                    run_fixed_phase4_experiment(config, self.vocabulary, corpus)

        self.assertEqual(raised.exception.details["invariant"], "experiment.counts.corpus")
        model_type.assert_not_called()
        guarded_works = (*corpus.training_works, *corpus.validation_works)
        self.assertTrue(
            all(
                isinstance(work, GuardedExtractedWork)
                and work.text_access_count == 0
                for work in guarded_works
            )
        )

    def test_sealed_supplier_is_refused_without_accessing_it_or_content(self) -> None:
        class SealedSupplier:
            accessed = False

            def __iter__(self):
                self.accessed = True
                raise AssertionError("sealed supplier was accessed")

        supplier = SealedSupplier()
        corpus = _corpus(guarded=True, sealed_supplier=supplier)
        with self.assertRaises(Phase4GovernanceError) as raised:
            self.valid_preflight(corpus=corpus)

        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.sealed_test.supplier",
        )
        self.assertFalse(supplier.accessed)
        self.assertTrue(
            all(
                isinstance(work, GuardedExtractedWork)
                and work.text_access_count == 0
                for work in (*corpus.training_works, *corpus.validation_works)
            )
        )


class AggregateMeasurementTests(Phase4ExperimentTestCase):
    def test_aggregate_is_target_weighted_not_mean_of_example_means(self) -> None:
        model = self.model()
        examples = (
            _document_example((1,), (2,)),
            _document_example((3, 4), (5, 6), start=1),
        )
        with torch.no_grad():
            means = tuple(
                float(
                    F.cross_entropy(
                        model(torch.tensor(item.example.input_ids)),
                        torch.tensor(item.example.target_ids),
                    ).item()
                )
                for item in examples
            )
        expected = math.fsum((means[0], means[1] * 2)) / 3
        wrong_equal_example_mean = math.fsum(means) / 2

        observed = measure_aggregate_loss(model, iter(examples))

        self.assertAlmostEqual(observed.loss, expected, places=6)
        self.assertNotAlmostEqual(observed.loss, wrong_equal_example_mean, places=5)
        self.assertEqual(observed.target_count, 3)
        self.assertEqual(observed.example_count, 2)

    def test_measurement_preserves_values_identities_digest_and_none_gradients(self) -> None:
        model = self.model()
        parameters = tuple(model.parameters())
        identities = tuple(id(parameter) for parameter in parameters)
        snapshots = tuple(parameter.detach().clone() for parameter in parameters)
        digest = model_parameter_digest(model)

        result = measure_aggregate_loss(
            model,
            (_document_example((1, 2), (2, 3)),),
        )

        self.assertEqual(tuple(id(parameter) for parameter in model.parameters()), identities)
        self.assertEqual(result.parameter_digest_before, digest)
        self.assertEqual(result.parameter_digest_after, digest)
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(model.parameters(), snapshots, strict=True)
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_measurement_refuses_existing_gradient_before_consumption(self) -> None:
        model = self.model()
        model(torch.tensor([1])).sum().backward()
        consumed = False

        def examples():
            nonlocal consumed
            consumed = True
            yield _document_example((1,), (2,))

        with self.assertRaises(Phase4ContractError) as raised:
            measure_aggregate_loss(model, examples())

        self.assertEqual(raised.exception.details["invariant"], "aggregate.gradients.none_before")
        self.assertFalse(consumed)

    def test_measurement_detects_parameter_mutation(self) -> None:
        model = self.model()

        def examples():
            yield _document_example((1,), (2,))
            with torch.no_grad():
                model.output_bias[0] += 1.0
            yield _document_example((2,), (3,), start=1)

        with self.assertRaises(Phase4ContractError) as raised:
            measure_aggregate_loss(model, examples())

        self.assertEqual(raised.exception.details["invariant"], "aggregate.parameter_digest")

    def test_zero_target_aggregate_fails_closed(self) -> None:
        with self.assertRaises(Phase4ContractError) as raised:
            measure_aggregate_loss(self.model(), ())

        self.assertEqual(raised.exception.details["invariant"], "aggregate.target_count.positive")


class TrainingPassTests(Phase4ExperimentTestCase):
    def test_tail_scaling_and_manual_sgd_match_independent_arithmetic(self) -> None:
        observed_model = self.model()
        reference_model = self.model()
        example = _document_example((3, 1), (1, 2))
        reference_loss = F.cross_entropy(
            reference_model(torch.tensor([3, 1])),
            torch.tensor([1, 2]),
        )
        (reference_loss * (2 / 64)).backward()
        expected = tuple(
            parameter.detach().clone() - 0.05 * parameter.grad.detach().clone()
            for parameter in reference_model.parameters()
        )

        result = run_training_pass(observed_model, (example,))

        self.assertEqual(result, TrainingPassResult(2, 1, 1))
        for parameter, expected_parameter in zip(
            observed_model.parameters(), expected, strict=True
        ):
            torch.testing.assert_close(
                parameter,
                expected_parameter,
                rtol=1e-6,
                atol=1e-7,
            )
            self.assertIsNone(parameter.grad)

    def test_one_update_per_example_and_input_order_are_preserved(self) -> None:
        model = self.model()
        consumed: list[int] = []

        def examples():
            for position, token_id in enumerate((5, 2, 7)):
                consumed.append(token_id)
                yield _document_example(
                    (token_id,),
                    ((token_id + 1) % 81,),
                    start=position * 64,
                )

        with patch.object(
            experiment_module,
            "manual_sgd_step",
            wraps=experiment_module.manual_sgd_step,
        ) as update:
            with patch.object(
                experiment_module,
                "clear_gradients",
                wraps=experiment_module.clear_gradients,
            ) as clear:
                result = run_training_pass(model, examples())

        self.assertEqual(consumed, [5, 2, 7])
        self.assertEqual(result, TrainingPassResult(3, 3, 3))
        self.assertEqual(update.call_count, 3)
        self.assertEqual(clear.call_count, 6)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_two_fresh_synthetic_passes_are_deterministic(self) -> None:
        first = self.model()
        second = self.model()
        examples = (
            _document_example((1, 2), (2, 3)),
            _document_example((4,), (5,), start=64),
        )

        first_result = run_training_pass(first, iter(examples))
        second_result = run_training_pass(second, iter(examples))

        self.assertEqual(first_result, second_result)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first.parameters(), second.parameters(), strict=True)
            )
        )

    def test_invalid_example_fails_before_any_update(self) -> None:
        model = self.model()
        digest = model_parameter_digest(model)

        with self.assertRaises(Phase4TypeError) as raised:
            run_training_pass(model, (object(),))  # type: ignore[arg-type]

        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.example.exact_document_shifted_example",
        )
        self.assertEqual(model_parameter_digest(model), digest)

    def test_nontraining_example_is_rejected_before_tensorization_or_update(self) -> None:
        model = self.model()
        digest = model_parameter_digest(model)
        validation = _document_example((1,), (2,), split="validation")
        with patch.object(experiment_module, "_tensorize") as tensorize:
            with patch.object(model, "forward", wraps=model.forward) as forward:
                with patch.object(experiment_module, "explicit_cross_entropy") as loss:
                    with patch.object(experiment_module, "manual_sgd_step") as update:
                        with self.assertRaises(Phase4GovernanceError) as split_error:
                            run_training_pass(model, (validation,))

        self.assertEqual(
            split_error.exception.details["invariant"],
            "training.example.split",
        )
        tensorize.assert_not_called()
        forward.assert_not_called()
        loss.assert_not_called()
        update.assert_not_called()
        self.assertEqual(model_parameter_digest(model), digest)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_training_governance_precedes_tensorization(self) -> None:
        model = self.model()
        calls: list[str] = []
        real_validate = experiment_module._validate_training_example
        real_tensorize = experiment_module._tensorize

        def validate(*args, **kwargs):
            calls.append("governance")
            return real_validate(*args, **kwargs)

        def tensorize(*args, **kwargs):
            calls.append("tensorize")
            return real_tensorize(*args, **kwargs)

        with patch.object(
            experiment_module,
            "_validate_training_example",
            side_effect=validate,
        ):
            with patch.object(
                experiment_module,
                "_tensorize",
                side_effect=tensorize,
            ):
                run_training_pass(model, (_document_example((1,), (2,)),))

        self.assertEqual(calls[:2], ["governance", "tensorize"])

    def test_out_of_order_example_is_refused(self) -> None:
        model = self.model()

        examples = (
            _document_example((1,), (2,), start=64),
            _document_example((2,), (3,), start=0),
        )
        with self.assertRaises(Phase4GovernanceError) as order_error:
            run_training_pass(model, examples)
        self.assertEqual(
            order_error.exception.details["invariant"],
            "training.example.order",
        )


class FixedLifecycleTests(Phase4ExperimentTestCase):
    def _measurement(self, loss: float, *, validation: bool = False):
        return AggregateMeasurement(
            loss=loss,
            target_count=98_295 if validation else 792_699,
            example_count=1_536 if validation else 12_389,
            parameter_digest_before="d" * 64,
            parameter_digest_after="d" * 64,
        )

    def _run_with_fixed_results(self, losses=(4.5, 4.6, 4.0, 4.7)):
        model = self.model()
        config = self.config()
        measurements = (
            self._measurement(losses[0]),
            self._measurement(losses[1], validation=True),
            self._measurement(losses[2]),
            self._measurement(losses[3], validation=True),
        )
        pass_result = TrainingPassResult(792_699, 12_389, 12_389)
        with patch.object(
            experiment_module,
            "validate_phase4_experiment_preflight",
        ) as preflight:
            with patch.object(
                experiment_module,
                "SimpleNeuralLanguageModel",
                return_value=model,
            ) as model_type:
                with patch.object(
                    experiment_module,
                    "_examples",
                    return_value=(),
                ):
                    with patch.object(
                        experiment_module,
                        "measure_aggregate_loss",
                        side_effect=measurements,
                    ) as measure:
                        with patch.object(
                            experiment_module,
                            "run_training_pass",
                            side_effect=(pass_result, pass_result),
                        ) as train:
                            with patch.object(
                                experiment_module,
                                "_utc_run_at",
                                return_value="2026-09-09T12:34:56Z",
                            ):
                                result = run_fixed_phase4_experiment(
                                    config,
                                    self.vocabulary,
                                    _corpus(),
                                )
        return result, preflight, model_type, measure, train

    def test_fixed_entry_point_has_one_constructor_and_exact_lifecycle_order(self) -> None:
        result, preflight, model_type, measure, train = self._run_with_fixed_results()

        preflight.assert_called_once()
        model_type.assert_called_once()
        self.assertEqual(measure.call_count, 4)
        self.assertEqual(train.call_count, 2)
        self.assertEqual(
            result.lifecycle_events,
            (
                "model_constructed",
                "initial_training_measured",
                "initial_validation_measured",
                "training_pass_1_completed",
                "training_pass_2_completed",
                "final_training_measured",
                "final_validation_measured",
            ),
        )
        self.assertEqual(result.model_constructor_count, 1)
        self.assertTrue(result.parameter_identity_stable)
        self.assertTrue(result.measurement_gradients_none)
        self.assertFalse(result.sealed_test_accessed)
        self.assertEqual(
            result.model_implementation_commit,
            TEST_MODEL_IMPLEMENTATION_COMMIT,
        )
        self.assertNotEqual(result.code_commit, result.model_implementation_commit)

    def test_external_spies_prove_global_call_order_and_boundaries(self) -> None:
        class SplitSentinel:
            def __iter__(self):
                return iter(())

        calls: list[str] = []
        model = self.model()
        train_sentinel = SplitSentinel()
        validation_sentinel = SplitSentinel()
        measurement_counts = {"train": 0, "validation": 0}
        pass_names = iter(("pass_1", "pass_2"))
        pass_result = TrainingPassResult(792_699, 12_389, 12_389)
        real_boundary = experiment_module._verify_model_boundary
        real_clear = experiment_module.clear_gradients
        real_result_type = Phase4ExperimentResult

        def preflight(*args, **kwargs):
            del args, kwargs
            calls.append("preflight")

        def construct_model(*args, **kwargs):
            del args, kwargs
            calls.append("model_constructed")
            return model

        def boundary(*args, **kwargs):
            calls.append("identity_gradient_boundary")
            return real_boundary(*args, **kwargs)

        def examples(*args, split, **kwargs):
            del args, kwargs
            if split == "train":
                return train_sentinel
            if split == "validation":
                return validation_sentinel
            raise AssertionError("unexpected split")

        def split_role(examples_argument):
            if examples_argument is train_sentinel:
                return "train"
            if examples_argument is validation_sentinel:
                return "validation"
            raise AssertionError("measurement received an unknown iterable")

        def measure(model_argument, examples_argument):
            self.assertIs(model_argument, model)
            role = split_role(examples_argument)
            if role == "train":
                result = self._measurement(
                    4.5 if measurement_counts[role] == 0 else 4.0
                )
            else:
                result = self._measurement(
                    4.6 if measurement_counts[role] == 0 else 4.7,
                    validation=True,
                )
            occurrence = measurement_counts[role]
            calls.append(f"{'initial' if occurrence == 0 else 'final'}_{role}")
            measurement_counts[role] += 1
            return result

        def train(model_argument, examples_argument):
            self.assertIs(model_argument, model)
            self.assertIs(examples_argument, train_sentinel)
            calls.append(next(pass_names))
            return pass_result

        def final_clear(*args, **kwargs):
            calls.append("final_gradient_clear")
            return real_clear(*args, **kwargs)

        def construct_result(**kwargs):
            calls.append("result_constructed")
            return real_result_type(**kwargs)

        with patch.object(
            experiment_module,
            "validate_phase4_experiment_preflight",
            side_effect=preflight,
        ):
            with patch.object(
                experiment_module,
                "SimpleNeuralLanguageModel",
                side_effect=construct_model,
            ):
                with patch.object(
                    experiment_module,
                    "_examples",
                    side_effect=examples,
                ):
                    with patch.object(
                        experiment_module,
                        "_verify_model_boundary",
                        side_effect=boundary,
                    ):
                        with patch.object(
                            experiment_module,
                            "measure_aggregate_loss",
                            side_effect=measure,
                        ):
                            with patch.object(
                                experiment_module,
                                "run_training_pass",
                                side_effect=train,
                            ):
                                with patch.object(
                                    experiment_module,
                                    "clear_gradients",
                                    side_effect=final_clear,
                                ):
                                    with patch.object(
                                        experiment_module,
                                        "_utc_run_at",
                                        return_value="2026-09-09T12:34:56Z",
                                    ):
                                        with patch.object(
                                            experiment_module,
                                            "Phase4ExperimentResult",
                                            side_effect=construct_result,
                                        ):
                                            result = run_fixed_phase4_experiment(
                                                self.config(),
                                                self.vocabulary,
                                                _corpus(),
                                            )

        self.assertIsInstance(result, real_result_type)
        expected = [
            "preflight",
            "model_constructed",
            "identity_gradient_boundary",
            "initial_train",
            "identity_gradient_boundary",
            "initial_validation",
            "identity_gradient_boundary",
            "pass_1",
            "identity_gradient_boundary",
            "pass_2",
            "identity_gradient_boundary",
            "final_gradient_clear",
            "final_train",
            "identity_gradient_boundary",
            "final_validation",
            "identity_gradient_boundary",
            "result_constructed",
        ]
        self.assertEqual(calls, expected)
        simulated_swapped_initial_role = f"initial_{split_role(validation_sentinel)}"
        self.assertEqual(simulated_swapped_initial_role, "initial_validation")
        self.assertNotEqual(simulated_swapped_initial_role, expected[3])

    def test_result_uses_both_fixed_predicates_and_reports_validation_direction(self) -> None:
        passed, *_ = self._run_with_fixed_results((4.5, 4.6, 4.0, 4.7))
        failed, *_ = self._run_with_fixed_results((4.5, 4.6, 4.5, 4.4))

        self.assertTrue(passed.below_uniform_baseline)
        self.assertTrue(passed.below_initial_training_loss)
        self.assertTrue(passed.passed)
        self.assertEqual(passed.validation_direction, "worsened")
        self.assertFalse(failed.below_uniform_baseline)
        self.assertFalse(failed.below_initial_training_loss)
        self.assertFalse(failed.passed)
        self.assertEqual(failed.validation_direction, "improved")

    def test_result_narrative_is_evidence_limited_and_workflow_specific(self) -> None:
        passed, *_ = self._run_with_fixed_results((4.5, 4.6, 4.0, 4.7))
        failed, *_ = self._run_with_fixed_results((4.5, 4.6, 4.5, 4.4))

        self.assertEqual(passed.utc_run_at, "2026-09-09T12:34:56Z")
        self.assertIn("does not establish generalization", passed.conclusion)
        self.assertIn("Gate 25", passed.follow_up)
        self.assertIn("did not meet", failed.conclusion)
        self.assertIn("Gate 26", failed.follow_up)
        self.assertIn("do not tune or rerun", failed.follow_up)
        joined = " ".join((*passed.observations, passed.conclusion, passed.follow_up))
        self.assertNotIn("test loss", joined.lower())
        self.assertNotIn("transformer", joined.lower())

    def test_utc_run_timestamp_is_explicit_second_resolution_utc(self) -> None:
        value = experiment_module._utc_run_at()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

        self.assertRegex(value, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertEqual(parsed.tzinfo, timezone.utc)

    def _assert_initial_measurement_count_failure(
        self,
        measurement: AggregateMeasurement,
        *,
        invariant: str,
    ) -> None:
        model = self.model()
        with patch.object(experiment_module, "validate_phase4_experiment_preflight"):
            with patch.object(
                experiment_module,
                "SimpleNeuralLanguageModel",
                return_value=model,
            ):
                with patch.object(experiment_module, "_examples", return_value=()):
                    with patch.object(
                        experiment_module,
                        "measure_aggregate_loss",
                        return_value=measurement,
                    ):
                        with patch.object(
                            experiment_module,
                            "run_training_pass",
                        ) as train:
                            with self.assertRaises(Phase4GovernanceError) as raised:
                                run_fixed_phase4_experiment(
                                    self.config(),
                                    self.vocabulary,
                                    _corpus(),
                                )
        self.assertEqual(raised.exception.details["invariant"], invariant)
        train.assert_not_called()

    def test_training_target_count_mismatch_fails_closed(self) -> None:
        self._assert_initial_measurement_count_failure(
            replace(self._measurement(4.5), target_count=792_698),
            invariant="experiment.measurement.target_count",
        )

    def test_training_example_count_mismatch_fails_closed(self) -> None:
        self._assert_initial_measurement_count_failure(
            replace(self._measurement(4.5), example_count=12_388),
            invariant="experiment.measurement.example_count",
        )

    def _assert_validation_measurement_count_failure(
        self,
        measurement: AggregateMeasurement,
        *,
        invariant: str,
    ) -> None:
        model = self.model()
        measurements = (self._measurement(4.5), measurement)
        with patch.object(experiment_module, "validate_phase4_experiment_preflight"):
            with patch.object(
                experiment_module,
                "SimpleNeuralLanguageModel",
                return_value=model,
            ):
                with patch.object(experiment_module, "_examples", return_value=()):
                    with patch.object(
                        experiment_module,
                        "measure_aggregate_loss",
                        side_effect=measurements,
                    ):
                        with patch.object(
                            experiment_module,
                            "run_training_pass",
                        ) as train:
                            with self.assertRaises(Phase4GovernanceError) as raised:
                                run_fixed_phase4_experiment(
                                    self.config(),
                                    self.vocabulary,
                                    _corpus(),
                                )
        self.assertEqual(raised.exception.details["invariant"], invariant)
        train.assert_not_called()

    def test_validation_target_count_mismatch_fails_closed(self) -> None:
        self._assert_validation_measurement_count_failure(
            replace(self._measurement(4.6, validation=True), target_count=98_294),
            invariant="experiment.measurement.target_count",
        )

    def test_validation_example_count_mismatch_fails_closed(self) -> None:
        self._assert_validation_measurement_count_failure(
            replace(self._measurement(4.6, validation=True), example_count=1_535),
            invariant="experiment.measurement.example_count",
        )

    def test_wrong_update_count_fails_without_final_measurements_or_result(self) -> None:
        model = self.model()
        measurements = (
            self._measurement(4.5),
            self._measurement(4.6, validation=True),
        )
        wrong = TrainingPassResult(792_699, 12_389, 12_388)
        with patch.object(experiment_module, "validate_phase4_experiment_preflight"):
            with patch.object(experiment_module, "SimpleNeuralLanguageModel", return_value=model):
                with patch.object(experiment_module, "_examples", return_value=()):
                    with patch.object(
                        experiment_module,
                        "measure_aggregate_loss",
                        side_effect=measurements,
                    ) as measure:
                        with patch.object(
                            experiment_module,
                            "run_training_pass",
                            return_value=wrong,
                        ):
                            with self.assertRaises(Phase4ContractError) as raised:
                                run_fixed_phase4_experiment(
                                    self.config(),
                                    self.vocabulary,
                                    _corpus(),
                                )

        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.training.update_count",
        )
        self.assertEqual(measure.call_count, 2)

    def test_final_total_update_count_mismatch_fails_closed(self) -> None:
        model = self.model()
        measurements = (
            self._measurement(4.5),
            self._measurement(4.6, validation=True),
        )
        passes = (
            TrainingPassResult(792_699, 12_389, 12_389),
            TrainingPassResult(792_699, 12_389, 12_388),
        )
        with patch.object(experiment_module, "validate_phase4_experiment_preflight"):
            with patch.object(
                experiment_module,
                "SimpleNeuralLanguageModel",
                return_value=model,
            ):
                with patch.object(experiment_module, "_examples", return_value=()):
                    with patch.object(
                        experiment_module,
                        "measure_aggregate_loss",
                        side_effect=measurements,
                    ) as measure:
                        with patch.object(
                            experiment_module,
                            "_require_pass_counts",
                        ):
                            with patch.object(
                                experiment_module,
                                "run_training_pass",
                                side_effect=passes,
                            ):
                                with self.assertRaises(Phase4ContractError) as raised:
                                    run_fixed_phase4_experiment(
                                        self.config(),
                                        self.vocabulary,
                                        _corpus(),
                                    )

        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.training.total_updates",
        )
        self.assertEqual(measure.call_count, 2)

    def _assert_training_exception_stops_without_retry(
        self,
        side_effect: tuple[object, ...],
        *,
        expected_calls: int,
    ) -> None:
        model = self.model()
        measurements = (
            self._measurement(4.5),
            self._measurement(4.6, validation=True),
        )
        log_before = (REPOSITORY_ROOT / "EXPERIMENT_LOG.md").read_bytes()
        persisted_before = tuple(
            path.relative_to(REPOSITORY_ROOT)
            for root_name in ("checkpoints", "experiments")
            for path in sorted((REPOSITORY_ROOT / root_name).rglob("*"))
            if path.is_file()
        )
        with patch.object(experiment_module, "validate_phase4_experiment_preflight"):
            with patch.object(
                experiment_module,
                "SimpleNeuralLanguageModel",
                return_value=model,
            ):
                with patch.object(experiment_module, "_examples", return_value=()):
                    with patch.object(
                        experiment_module,
                        "measure_aggregate_loss",
                        side_effect=measurements,
                    ) as measure:
                        with patch.object(
                            experiment_module,
                            "run_training_pass",
                            side_effect=side_effect,
                        ) as train:
                            with patch.object(
                                experiment_module,
                                "Phase4ExperimentResult",
                            ) as result_type:
                                with self.assertRaises(RuntimeError):
                                    run_fixed_phase4_experiment(
                                        self.config(),
                                        self.vocabulary,
                                        _corpus(),
                                    )

        self.assertEqual(train.call_count, expected_calls)
        self.assertEqual(measure.call_count, 2)
        result_type.assert_not_called()
        self.assertEqual(
            (REPOSITORY_ROOT / "EXPERIMENT_LOG.md").read_bytes(),
            log_before,
        )
        persisted_after = tuple(
            path.relative_to(REPOSITORY_ROOT)
            for root_name in ("checkpoints", "experiments")
            for path in sorted((REPOSITORY_ROOT / root_name).rglob("*"))
            if path.is_file()
        )
        self.assertEqual(persisted_after, persisted_before)

    def test_pass_one_exception_stops_without_pass_two_or_final_measurement(self) -> None:
        self._assert_training_exception_stops_without_retry(
            (RuntimeError("pass one stopped"),),
            expected_calls=1,
        )

    def test_pass_two_exception_stops_without_retry_or_final_measurement(self) -> None:
        self._assert_training_exception_stops_without_retry(
            (
                TrainingPassResult(792_699, 12_389, 12_389),
                RuntimeError("pass two stopped"),
            ),
            expected_calls=2,
        )

    def test_parameter_replacement_during_pass_fails_closed(self) -> None:
        model = self.model()
        measurements = (
            self._measurement(4.5),
            self._measurement(4.6, validation=True),
        )

        def replace_parameter(*args, **kwargs):
            del args, kwargs
            model.output_bias = torch.nn.Parameter(model.output_bias.detach().clone())
            return TrainingPassResult(792_699, 12_389, 12_389)

        with patch.object(experiment_module, "validate_phase4_experiment_preflight"):
            with patch.object(experiment_module, "SimpleNeuralLanguageModel", return_value=model):
                with patch.object(experiment_module, "_examples", return_value=()):
                    with patch.object(
                        experiment_module,
                        "measure_aggregate_loss",
                        side_effect=measurements,
                    ):
                        with patch.object(
                            experiment_module,
                            "run_training_pass",
                            side_effect=replace_parameter,
                        ):
                            with self.assertRaises(Phase4ContractError) as raised:
                                run_fixed_phase4_experiment(
                                    self.config(),
                                    self.vocabulary,
                                    _corpus(),
                                )

        self.assertEqual(
            raised.exception.details["invariant"],
            "experiment.lifecycle.parameter_identity",
        )

    def test_result_is_frozen_and_contains_complete_fixed_observations(self) -> None:
        result, *_ = self._run_with_fixed_results()

        self.assertIsInstance(result, Phase4ExperimentResult)
        with self.assertRaises(FrozenInstanceError):
            result.passed = False  # type: ignore[misc]
        self.assertEqual(result.training_target_count, 792_699)
        self.assertEqual(result.validation_target_count, 98_295)
        self.assertEqual(result.update_count, 24_778)
        self.assertEqual(len(result.measurement_digests), 4)
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.dtype, "torch.float32")
        self.assertFalse(result.weight_tying)
        self.assertFalse(result.retained_model_or_data_artifacts)
        self.assertIsNone(result.failed_run_predecessor)
        self.assertIsNone(result.amended_pre_registration)

    def test_result_schema_covers_every_mandatory_durable_record_field(self) -> None:
        expected = {
            "experiment_id",
            "utc_run_at",
            "status",
            "question",
            "code_commit",
            "model_implementation_commit",
            "tokenizer_implementation_commit",
            "vocabulary_id",
            "vocabulary_schema_version",
            "vocabulary_size",
            "vocabulary_artifact_path",
            "vocabulary_artifact_sha256",
            "dataset_id",
            "dataset_manifest_sha256",
            "permitted_work_hashes",
            "embedding_decision",
            "embedding_contract_commit",
            "embedding_implementation_commit",
            "model_name",
            "model_architecture",
            "parameter_names",
            "parameter_shapes",
            "parameter_count",
            "trainable_parameters",
            "weight_tying",
            "python_version",
            "pytorch_version",
            "operating_system",
            "machine_architecture",
            "device",
            "dtype",
            "embedding_seed",
            "output_head_seed",
            "embedding_initialization",
            "output_head_initialization",
            "initialization_order",
            "rng_scope",
            "context_length",
            "stride",
            "variable_tail",
            "padding",
            "traversal",
            "shuffle",
            "one_example_at_a_time",
            "update_rule",
            "gradient_clearing",
            "tail_scale",
            "aggregate_rule",
            "learning_rate",
            "training_passes",
            "training_source_token_count",
            "training_target_count",
            "training_example_count",
            "validation_source_token_count",
            "validation_target_count",
            "validation_example_count",
            "update_count",
            "initial_training_loss",
            "initial_validation_loss",
            "final_training_loss",
            "final_validation_loss",
            "uniform_baseline",
            "success_predicate",
            "below_uniform_baseline",
            "below_initial_training_loss",
            "passed",
            "validation_direction",
            "model_constructor_count",
            "model_identity",
            "parameter_identities",
            "parameter_identity_stable",
            "initial_parameter_digest",
            "final_parameter_digest",
            "measurement_digests",
            "measurement_gradients_none",
            "validation_observation_only",
            "sealed_test_accessed",
            "retained_model_or_data_artifacts",
            "failed_run_predecessor",
            "amended_pre_registration",
            "observations",
            "conclusion",
            "follow_up",
            "lifecycle_events",
        }

        self.assertEqual(
            {result_field.name for result_field in fields(Phase4ExperimentResult)},
            expected,
        )

    def test_runner_source_has_no_checkpoint_or_experiment_log_write_path(self) -> None:
        source = inspect.getsource(experiment_module)

        self.assertNotIn("torch.save", source)
        self.assertNotIn("EXPERIMENT_LOG.md", source)
        self.assertNotIn("checkpoints/", source)


if __name__ == "__main__":
    unittest.main()
