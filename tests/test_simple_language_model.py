"""Independent tests for the Phase 4 model, loss, and update slice."""

from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.model.simple_language_model as model_module  # noqa: E402
from sebgpt.data.shifted_examples import (  # noqa: E402
    Phase4ContractError,
    Phase4TypeError,
)
from sebgpt.model.simple_language_model import (  # noqa: E402
    SimpleNeuralLanguageModel,
    clear_gradients,
    explicit_cross_entropy,
    manual_sgd_step,
    measure_example_loss,
    model_parameter_digest,
    probabilities_from_logits,
    scale_backward_loss,
)
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    VocabularyBinding,
    load_accepted_vocabulary_binding,
)


TEST_VOCABULARY_SIZE = 81
TEST_EMBEDDING_DIM = 32
TEST_MAX_POSITIONS = 256
TEST_CONTEXT_LENGTH = 64
TEST_EMBEDDING_SEED = 1337
TEST_HEAD_SEED = 4004
TEST_DEVICE = torch.device("cpu")
TEST_DTYPE = torch.float32
TEST_PARAMETER_NAMES = (
    "output_weight",
    "output_bias",
    "representation.token_embeddings",
    "representation.position_embeddings",
)
TEST_PARAMETER_SHAPES = (
    (32, 81),
    (81,),
    (81, 32),
    (256, 32),
)


class Phase4ModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)

    def model(self, **overrides: object) -> SimpleNeuralLanguageModel:
        arguments: dict[str, object] = {
            "embedding_dim": TEST_EMBEDDING_DIM,
            "max_positions": TEST_MAX_POSITIONS,
            "embedding_seed": TEST_EMBEDDING_SEED,
            "output_head_seed": TEST_HEAD_SEED,
            "device": TEST_DEVICE,
            "dtype": TEST_DTYPE,
        }
        arguments.update(overrides)
        return SimpleNeuralLanguageModel(
            self.vocabulary,
            **arguments,  # type: ignore[arg-type]
        )


class ModelStructureAndInitializationTests(Phase4ModelTestCase):
    def test_exact_parameter_enumeration_shapes_count_and_no_buffers(self) -> None:
        model = self.model()
        items = tuple(model.named_parameters())

        self.assertEqual(tuple(name for name, _ in items), TEST_PARAMETER_NAMES)
        self.assertEqual(
            tuple(tuple(parameter.shape) for _, parameter in items),
            TEST_PARAMETER_SHAPES,
        )
        self.assertEqual(sum(parameter.numel() for _, parameter in items), 13457)
        self.assertEqual(tuple(model.named_buffers()), ())
        self.assertEqual(
            tuple(name for name, _ in model.named_modules()),
            ("", "representation"),
        )

    def test_all_parameters_are_trainable_cpu_float32_and_head_is_untied(self) -> None:
        model = self.model()

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(
            all(parameter.device == TEST_DEVICE for parameter in model.parameters())
        )
        self.assertTrue(
            all(parameter.dtype is TEST_DTYPE for parameter in model.parameters())
        )
        self.assertNotEqual(
            model.output_weight.untyped_storage().data_ptr(),
            model.representation.token_embeddings.untyped_storage().data_ptr(),
        )

    def test_head_initialization_matches_an_independent_local_generator(self) -> None:
        model = self.model()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(TEST_HEAD_SEED)
        expected = torch.empty((32, 81), dtype=torch.float32, device="cpu")
        torch.nn.init.normal_(
            expected,
            mean=0.0,
            std=1.0 / math.sqrt(32),
            generator=generator,
        )

        self.assertTrue(torch.equal(model.output_weight, expected))
        self.assertTrue(torch.equal(model.output_bias, torch.zeros(81)))

    def test_head_does_not_restart_the_phase_3_seed(self) -> None:
        model = self.model()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(TEST_EMBEDDING_SEED)
        wrong_seed = torch.empty((32, 81), dtype=torch.float32)
        torch.nn.init.normal_(
            wrong_seed,
            mean=0.0,
            std=1.0 / math.sqrt(32),
            generator=generator,
        )

        self.assertFalse(torch.equal(model.output_weight, wrong_seed))

    def test_repeated_construction_is_bitwise_deterministic(self) -> None:
        first = self.model()
        second = self.model()

        self.assertTrue(
            all(
                torch.equal(first_parameter, second_parameter)
                for first_parameter, second_parameter in zip(
                    first.parameters(),
                    second.parameters(),
                    strict=True,
                )
            )
        )

    def test_construction_preserves_and_ignores_global_rng_state(self) -> None:
        torch.manual_seed(9191)
        before = torch.random.get_rng_state().clone()
        first = self.model()
        after = torch.random.get_rng_state().clone()
        torch.rand(37)
        second = self.model()

        self.assertTrue(torch.equal(before, after))
        self.assertTrue(
            all(
                torch.equal(first_parameter, second_parameter)
                for first_parameter, second_parameter in zip(
                    first.parameters(),
                    second.parameters(),
                    strict=True,
                )
            )
        )

    def test_constructor_type_validation_precedes_allocation(self) -> None:
        cases = (
            ("embedding_dim", True, "model.embedding_dim.exact_int"),
            ("max_positions", 256.0, "model.max_positions.exact_int"),
            ("embedding_seed", False, "model.embedding_seed.exact_int"),
            ("output_head_seed", "4004", "model.output_head_seed.exact_int"),
            ("device", "cpu", "model.device.exact_torch_device"),
            ("dtype", "float32", "model.dtype.torch_dtype"),
        )
        for key, value, invariant in cases:
            with self.subTest(invariant=invariant):
                with patch.object(torch, "empty") as empty:
                    with self.assertRaises(Phase4TypeError) as raised:
                        self.model(**{key: value})
                self.assertEqual(raised.exception.details["invariant"], invariant)
                empty.assert_not_called()

    def test_constructor_value_validation_precedes_allocation(self) -> None:
        cases = (
            ("embedding_dim", 31, "model.embedding_dim.value"),
            ("max_positions", 255, "model.max_positions.value"),
            ("embedding_seed", 1338, "model.embedding_seed.value"),
            ("output_head_seed", 4005, "model.output_head_seed.value"),
            ("device", torch.device("meta"), "model.device.value"),
            ("dtype", torch.float64, "model.dtype.value"),
        )
        for key, value, invariant in cases:
            with self.subTest(invariant=invariant):
                with patch.object(torch, "empty") as empty:
                    with self.assertRaises(Phase4ContractError) as raised:
                        self.model(**{key: value})
                self.assertEqual(raised.exception.details["invariant"], invariant)
                empty.assert_not_called()

    def test_vocabulary_type_and_factory_verification_order(self) -> None:
        arguments = {
            "embedding_dim": 32,
            "max_positions": 256,
            "embedding_seed": 1337,
            "output_head_seed": 4004,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        with self.assertRaises(Phase4TypeError) as type_error:
            SimpleNeuralLanguageModel(object(), **arguments)  # type: ignore[arg-type]
        forged = object.__new__(VocabularyBinding)
        with self.assertRaises(Phase4ContractError) as binding_error:
            SimpleNeuralLanguageModel(forged, **arguments)

        self.assertEqual(
            type_error.exception.details["invariant"],
            "model.vocabulary.binding_type",
        )
        self.assertEqual(
            binding_error.exception.details["invariant"],
            "model.vocabulary.verified",
        )


class ModelForwardTests(Phase4ModelTestCase):
    def test_zero_one_and_maximum_lengths_have_exact_logits_shapes(self) -> None:
        model = self.model()
        for length in (0, 1, 64):
            with self.subTest(length=length):
                logits = model(torch.arange(length, dtype=torch.long) % 81)
                self.assertEqual(tuple(logits.shape), (length, 81))
                self.assertIs(logits.dtype, torch.float32)
                self.assertEqual(logits.device, torch.device("cpu"))

    def test_length_65_is_rejected_before_representation_lookup(self) -> None:
        model = self.model()
        with patch.object(model.representation, "forward") as forward:
            with self.assertRaises(Phase4ContractError) as raised:
                model(torch.arange(65, dtype=torch.long) % 81)

        self.assertEqual(raised.exception.details["invariant"], "model.input.length")
        forward.assert_not_called()

    def test_forward_validation_order_is_type_rank_dtype_device_length_range(self) -> None:
        model = self.model()
        cases = (
            (object(), Phase4TypeError, "model.input.tensor"),
            (torch.tensor(1, dtype=torch.long), Phase4ContractError, "model.input.rank"),
            (torch.zeros((1, 1), dtype=torch.float32), Phase4ContractError, "model.input.rank"),
            (torch.zeros(1, dtype=torch.float32), Phase4ContractError, "model.input.dtype"),
            (torch.empty(1, dtype=torch.long, device="meta"), Phase4ContractError, "model.input.device"),
            (torch.full((65,), 81, dtype=torch.long), Phase4ContractError, "model.input.length"),
            (torch.tensor([0, 81]), Phase4ContractError, "model.input.token_id_range"),
        )
        for value, error_type, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(error_type) as raised:
                    model(value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_tensor_subclass_and_noncontiguous_inputs_are_accepted(self) -> None:
        class TensorSubclass(torch.Tensor):
            pass

        model = self.model()
        subclass = torch.tensor([1, 2, 3], dtype=torch.long).as_subclass(
            TensorSubclass
        )
        backing = torch.tensor([1, 9, 2, 9, 3, 9], dtype=torch.long)
        noncontiguous = backing[::2]

        self.assertEqual(tuple(model(subclass).shape), (3, 81))
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(tuple(model(noncontiguous).shape), (3, 81))

    def test_logits_equal_exact_matrix_arithmetic_and_forward_returns_only_logits(self) -> None:
        model = self.model()
        token_ids = torch.tensor([3, 1, 3], dtype=torch.long)
        representations = model.representation(token_ids)
        expected = representations @ model.output_weight + model.output_bias
        observed = model(token_ids)

        self.assertIsInstance(observed, torch.Tensor)
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(tuple(observed.shape), (3, 81))

    def test_changing_one_token_cannot_change_other_logits_rows(self) -> None:
        model = self.model()
        first = model(torch.tensor([1, 2, 3], dtype=torch.long))
        second = model(torch.tensor([1, 4, 3], dtype=torch.long))

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertFalse(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_forward_does_not_mutate_input_or_parameters(self) -> None:
        model = self.model()
        token_ids = torch.tensor([3, 1, 3], dtype=torch.long)
        input_before = token_ids.clone()
        parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())
        model(token_ids)

        self.assertTrue(torch.equal(token_ids, input_before))
        self.assertTrue(
            all(
                torch.equal(parameter, before)
                for parameter, before in zip(
                    model.parameters(), parameters_before, strict=True
                )
            )
        )


class ProbabilityTests(unittest.TestCase):
    def test_empty_logits_return_a_distinct_exact_empty_tensor(self) -> None:
        logits = torch.empty((0, 81), dtype=torch.float32)
        probabilities = probabilities_from_logits(logits)

        self.assertIsNot(probabilities, logits)
        self.assertEqual(tuple(probabilities.shape), (0, 81))
        self.assertIs(probabilities.dtype, torch.float32)
        self.assertEqual(probabilities.device, torch.device("cpu"))

    def test_probabilities_match_independent_stable_row_arithmetic(self) -> None:
        logits = torch.linspace(-3.0, 3.0, 162, dtype=torch.float32).reshape(2, 81)
        row_max = logits.max(dim=1, keepdim=True).values
        exponentials = torch.exp(logits - row_max)
        expected = exponentials / exponentials.sum(dim=1, keepdim=True)
        observed = probabilities_from_logits(logits)

        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            observed.sum(dim=1),
            torch.ones(2),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_extreme_representable_logits_are_stable(self) -> None:
        logits = torch.full((1, 81), -10000.0)
        logits[0, 17] = 10000.0
        probabilities = probabilities_from_logits(logits)

        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertEqual(probabilities[0, 17].item(), 1.0)
        self.assertEqual(probabilities.sum().item(), 1.0)

    def test_probability_validation_order_is_exact(self) -> None:
        cases = (
            (object(), Phase4TypeError, "probabilities.logits.tensor"),
            (torch.zeros(81), Phase4ContractError, "probabilities.logits.rank"),
            (torch.zeros((1, 81), dtype=torch.float64), Phase4ContractError, "probabilities.logits.dtype"),
            (torch.empty((1, 81), device="meta"), Phase4ContractError, "probabilities.logits.device"),
            (torch.zeros((1, 80)), Phase4ContractError, "probabilities.logits.class_dimension"),
            (torch.full((1, 81), float("nan")), Phase4ContractError, "probabilities.logits.finite"),
        )
        for value, error_type, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(error_type) as raised:
                    probabilities_from_logits(value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)


class ExplicitCrossEntropyTests(unittest.TestCase):
    def test_uniform_logits_equal_log_vocabulary_size(self) -> None:
        logits = torch.zeros((3, 81), dtype=torch.float32)
        targets = torch.tensor([0, 40, 80], dtype=torch.long)
        loss = explicit_cross_entropy(logits, targets)

        self.assertEqual(tuple(loss.shape), ())
        self.assertIs(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), math.log(81), places=6)

    def test_ordinary_loss_matches_pytorch_reference(self) -> None:
        logits = torch.linspace(-2.0, 2.0, 243, dtype=torch.float32).reshape(3, 81)
        logits.requires_grad_()
        targets = torch.tensor([0, 40, 80], dtype=torch.long)
        observed = explicit_cross_entropy(logits, targets)
        reference = F.cross_entropy(logits, targets)

        torch.testing.assert_close(observed, reference, rtol=1e-6, atol=1e-6)
        self.assertTrue(observed.requires_grad)

    def test_plus_minus_10000_loss_is_representable_and_matches_reference(self) -> None:
        logits = torch.full((1, 81), -10000.0)
        logits[0, 7] = 10000.0
        targets = torch.tensor([7])
        observed = explicit_cross_entropy(logits, targets)
        reference = F.cross_entropy(logits, targets)

        self.assertTrue(torch.isfinite(observed))
        torch.testing.assert_close(observed, reference, rtol=1e-6, atol=1e-6)

    def test_negative_infinite_shift_is_allowed_when_loss_is_representable(self) -> None:
        limits = torch.finfo(torch.float32)
        logits = torch.full((1, 81), limits.min)
        logits[0, 0] = limits.max
        loss = explicit_cross_entropy(logits, torch.tensor([0]))

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)

    def test_unrepresentable_target_margin_is_refused_deterministically(self) -> None:
        limits = torch.finfo(torch.float32)
        logits = torch.full((1, 81), limits.min)
        logits[0, 0] = limits.max
        with self.assertRaises(Phase4ContractError) as raised:
            explicit_cross_entropy(logits, torch.tensor([1]))

        self.assertEqual(
            raised.exception.details["invariant"],
            "loss.unrepresentable.target_margin",
        )

    def test_loss_input_validation_order_is_exact(self) -> None:
        valid_logits = torch.zeros((1, 81), dtype=torch.float32)
        valid_targets = torch.zeros(1, dtype=torch.long)
        cases = (
            (object(), object(), Phase4TypeError, "loss.logits.tensor"),
            (valid_logits, object(), Phase4TypeError, "loss.targets.tensor"),
            (torch.zeros(81), torch.zeros((1, 1)), Phase4ContractError, "loss.logits.rank"),
            (valid_logits, torch.zeros((1, 1)), Phase4ContractError, "loss.targets.rank"),
            (valid_logits.double(), valid_targets, Phase4ContractError, "loss.logits.dtype"),
            (valid_logits, valid_targets.int(), Phase4ContractError, "loss.targets.dtype"),
            (torch.empty((1, 81), device="meta"), valid_targets, Phase4ContractError, "loss.input.device"),
            (torch.zeros((1, 80)), valid_targets, Phase4ContractError, "loss.logits.class_dimension"),
            (torch.zeros((2, 81)), valid_targets, Phase4ContractError, "loss.input.length_match"),
            (torch.empty((0, 81)), torch.empty(0, dtype=torch.long), Phase4ContractError, "loss.input.nonempty"),
            (torch.full((1, 81), float("inf")), valid_targets, Phase4ContractError, "loss.logits.finite"),
            (valid_logits, torch.tensor([81]), Phase4ContractError, "loss.targets.range"),
        )
        for logits, targets, error_type, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(error_type) as raised:
                    explicit_cross_entropy(logits, targets)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_loss_does_not_mutate_logits_or_targets(self) -> None:
        logits = torch.linspace(-1.0, 1.0, 162).reshape(2, 81)
        targets = torch.tensor([7, 13])
        logits_before = logits.clone()
        targets_before = targets.clone()

        explicit_cross_entropy(logits, targets)

        self.assertTrue(torch.equal(logits, logits_before))
        self.assertTrue(torch.equal(targets, targets_before))

    def test_implementation_does_not_call_a_combined_reference_loss(self) -> None:
        source = inspect.getsource(model_module.explicit_cross_entropy)

        self.assertNotIn("functional.cross_entropy", source)
        self.assertNotIn("CrossEntropyLoss", source)


class GradientAndUpdateTests(Phase4ModelTestCase):
    def synthetic_backward(self):
        model = self.model()
        token_ids = torch.tensor([3, 1, 3], dtype=torch.long)
        targets = torch.tensor([1, 3, 2], dtype=torch.long)
        logits = model(token_ids)
        logits.retain_grad()
        loss = explicit_cross_entropy(logits, targets)
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        loss.backward()
        return model, logits, loss, before

    def test_synthetic_backward_reaches_all_parameters_without_mutation(self) -> None:
        model, _, loss, before = self.synthetic_backward()

        self.assertEqual(tuple(loss.shape), ())
        for parameter, snapshot in zip(model.parameters(), before, strict=True):
            self.assertTrue(torch.equal(parameter, snapshot))
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertEqual(tuple(parameter.grad.shape), tuple(parameter.shape))
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(torch.count_nonzero(model.output_weight.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(model.output_bias.grad).item(), 0)

    def test_repeated_token_gradient_is_sum_of_both_position_contributions(self) -> None:
        model, logits, _, _ = self.synthetic_backward()
        assert logits.grad is not None
        representation_gradients = logits.grad @ model.output_weight.detach().T
        token_gradients = model.representation.token_embeddings.grad
        position_gradients = model.representation.position_embeddings.grad
        assert token_gradients is not None
        assert position_gradients is not None

        torch.testing.assert_close(
            token_gradients[3],
            representation_gradients[0] + representation_gradients[2],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            token_gradients[1], representation_gradients[1], rtol=1e-6, atol=1e-6
        )
        self.assertEqual(torch.count_nonzero(token_gradients[80]).item(), 0)
        torch.testing.assert_close(
            position_gradients[:3], representation_gradients, rtol=1e-6, atol=1e-6
        )
        self.assertEqual(torch.count_nonzero(position_gradients[3]).item(), 0)

    def test_manual_sgd_updates_all_parameters_with_exact_arithmetic(self) -> None:
        model, _, _, before = self.synthetic_backward()
        gradients = tuple(parameter.grad.detach().clone() for parameter in model.parameters())  # type: ignore[union-attr]
        manual_sgd_step(model, learning_rate=0.05)

        for parameter, snapshot, gradient in zip(
            model.parameters(), before, gradients, strict=True
        ):
            torch.testing.assert_close(
                parameter,
                snapshot - 0.05 * gradient,
                rtol=0.0,
                atol=0.0,
            )
            assert parameter.grad is not None
            self.assertTrue(torch.equal(parameter.grad, gradient))

    def test_learning_rate_failures_precede_model_and_never_mutate(self) -> None:
        model = self.model()
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        cases = (
            (True, Phase4TypeError, "sgd.learning_rate.exact_float"),
            (1, Phase4TypeError, "sgd.learning_rate.exact_float"),
            (0.0, Phase4ContractError, "sgd.learning_rate.value"),
            (-0.1, Phase4ContractError, "sgd.learning_rate.value"),
            (float("nan"), Phase4ContractError, "sgd.learning_rate.value"),
        )
        for value, error_type, invariant in cases:
            with self.subTest(invariant=invariant, value_type=type(value).__name__):
                with self.assertRaises(error_type) as raised:
                    manual_sgd_step(object(), learning_rate=value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(model.parameters(), before, strict=True)
            )
        )

    def test_valid_learning_rate_then_wrong_model_uses_type_error(self) -> None:
        with self.assertRaises(Phase4TypeError) as raised:
            manual_sgd_step(object(), learning_rate=0.05)  # type: ignore[arg-type]

        self.assertEqual(raised.exception.details["invariant"], "sgd.model.exact_type")

    def test_extra_parameter_refuses_before_any_update(self) -> None:
        model, _, _, _ = self.synthetic_backward()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        model.register_parameter(
            "unexpected_parameter",
            torch.nn.Parameter(torch.zeros(1)),
        )

        with self.assertRaises(Phase4ContractError) as raised:
            manual_sgd_step(model, learning_rate=0.05)

        self.assertEqual(raised.exception.details["invariant"], "parameters.enumeration")
        for name, snapshot in before.items():
            self.assertTrue(torch.equal(model.get_parameter(name), snapshot))

    def test_nonfinite_parameter_refuses_before_any_update(self) -> None:
        model, _, _, _ = self.synthetic_backward()
        with torch.no_grad():
            model.output_bias[0] = float("inf")
        before = tuple(parameter.detach().clone() for parameter in model.parameters())

        with self.assertRaises(Phase4ContractError) as raised:
            manual_sgd_step(model, learning_rate=0.05)

        self.assertEqual(
            raised.exception.details["invariant"],
            "parameters.parameter.finite",
        )
        self.assertEqual(raised.exception.details["position"], 1)
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(model.parameters(), before, strict=True)
            )
        )

    def test_missing_gradient_refuses_without_partial_mutation(self) -> None:
        model, _, _, before = self.synthetic_backward()
        model.output_weight.grad = None

        with self.assertRaises(Phase4ContractError) as raised:
            manual_sgd_step(model, learning_rate=0.05)

        self.assertEqual(raised.exception.details["invariant"], "sgd.gradient.present")
        self.assertEqual(raised.exception.details["position"], 0)
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(model.parameters(), before, strict=True)
            )
        )

    def test_gradient_shape_dtype_device_and_finiteness_are_atomic(self) -> None:
        factories = (
            (
                lambda parameter: torch.zeros(80),
                "sgd.gradient.shape",
            ),
            (
                lambda parameter: torch.zeros(parameter.shape, dtype=torch.float64),
                "sgd.gradient.dtype",
            ),
            (
                lambda parameter: torch.empty(parameter.shape, device="meta"),
                "sgd.gradient.device",
            ),
            (
                lambda parameter: torch.full(parameter.shape, float("inf")),
                "sgd.gradient.finite",
            ),
        )
        for factory, invariant in factories:
            with self.subTest(invariant=invariant):
                model, _, _, before = self.synthetic_backward()
                real_gradients = {
                    id(parameter): parameter.grad for parameter in model.parameters()
                }

                def gradient(parameter):
                    if parameter is model.output_bias:
                        return factory(parameter)
                    return real_gradients[id(parameter)]

                with patch.object(model_module, "_gradient_for_parameter", gradient):
                    with self.assertRaises(Phase4ContractError) as raised:
                        manual_sgd_step(model, learning_rate=0.05)

                self.assertEqual(raised.exception.details["invariant"], invariant)
                self.assertEqual(raised.exception.details["position"], 1)
                self.assertTrue(
                    all(
                        torch.equal(parameter, snapshot)
                        for parameter, snapshot in zip(
                            model.parameters(), before, strict=True
                        )
                    )
                )

    def test_missing_gradient_precedes_later_shape_failure(self) -> None:
        model, _, _, before = self.synthetic_backward()
        real_gradients = {
            id(parameter): parameter.grad for parameter in model.parameters()
        }

        def gradient(parameter):
            if parameter is model.output_weight:
                return None
            if parameter is model.output_bias:
                return torch.zeros(80)
            return real_gradients[id(parameter)]

        with patch.object(model_module, "_gradient_for_parameter", gradient):
            with self.assertRaises(Phase4ContractError) as raised:
                manual_sgd_step(model, learning_rate=0.05)

        self.assertEqual(raised.exception.details["invariant"], "sgd.gradient.present")
        self.assertEqual(raised.exception.details["position"], 0)
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(model.parameters(), before, strict=True)
            )
        )

    def test_clear_gradients_accepts_none_and_clears_all_after_backward(self) -> None:
        model = self.model()
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        clear_gradients(model)
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(model.parameters(), before, strict=True)
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

        model, _, _, _ = self.synthetic_backward()
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        clear_gradients(model)
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(model.parameters(), before, strict=True)
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_clear_gradients_parameter_validation_is_category_major(self) -> None:
        model, _, _, _ = self.synthetic_backward()
        with torch.no_grad():
            model.output_weight[0, 0] = float("inf")
            model.output_bias.set_(torch.zeros(80, dtype=torch.float32))
        parameters = tuple(model.parameters())
        parameter_snapshots = tuple(
            parameter.detach().clone() for parameter in parameters
        )
        gradient_snapshots = tuple(parameter.grad for parameter in parameters)

        with self.assertRaises(Phase4ContractError) as raised:
            clear_gradients(model)

        self.assertEqual(
            dict(raised.exception.details),
            {"invariant": "parameters.parameter.shape", "position": 1},
        )
        self.assertTrue(
            all(
                torch.equal(parameter, snapshot)
                for parameter, snapshot in zip(
                    parameters, parameter_snapshots, strict=True
                )
            )
        )
        self.assertTrue(
            all(
                parameter.grad is gradient
                for parameter, gradient in zip(
                    parameters, gradient_snapshots, strict=True
                )
            )
        )

    def test_clear_gradients_preflight_failure_clears_nothing(self) -> None:
        model, _, _, _ = self.synthetic_backward()
        before = tuple(parameter.grad for parameter in model.parameters())
        real_gradients = {id(parameter): parameter.grad for parameter in model.parameters()}

        def gradient(parameter):
            if parameter is model.output_bias:
                return torch.zeros(80)
            return real_gradients[id(parameter)]

        with patch.object(model_module, "_gradient_for_parameter", gradient):
            with self.assertRaises(Phase4ContractError) as raised:
                clear_gradients(model)

        self.assertEqual(
            raised.exception.details["invariant"],
            "clear_gradients.gradient.shape",
        )
        self.assertTrue(
            all(
                parameter.grad is gradient
                for parameter, gradient in zip(
                    model.parameters(), before, strict=True
                )
            )
        )

    def test_clear_gradient_dtype_device_and_finiteness_fail_without_clearing(self) -> None:
        factories = (
            (
                lambda parameter: torch.zeros(parameter.shape, dtype=torch.float64),
                "clear_gradients.gradient.dtype",
            ),
            (
                lambda parameter: torch.empty(parameter.shape, device="meta"),
                "clear_gradients.gradient.device",
            ),
            (
                lambda parameter: torch.full(parameter.shape, float("inf")),
                "clear_gradients.gradient.finite",
            ),
        )
        for factory, invariant in factories:
            with self.subTest(invariant=invariant):
                model, _, _, _ = self.synthetic_backward()
                before = tuple(parameter.grad for parameter in model.parameters())
                real_gradients = {
                    id(parameter): parameter.grad for parameter in model.parameters()
                }

                def gradient(parameter):
                    if parameter is model.output_bias:
                        return factory(parameter)
                    return real_gradients[id(parameter)]

                with patch.object(model_module, "_gradient_for_parameter", gradient):
                    with self.assertRaises(Phase4ContractError) as raised:
                        clear_gradients(model)

                self.assertEqual(raised.exception.details["invariant"], invariant)
                self.assertEqual(raised.exception.details["position"], 1)
                self.assertTrue(
                    all(
                        parameter.grad is original
                        for parameter, original in zip(
                            model.parameters(), before, strict=True
                        )
                    )
                )


class TailScalingAndMeasurementTests(Phase4ModelTestCase):
    def test_full_length_scaling_leaves_mean_loss_unchanged(self) -> None:
        mean_loss = torch.tensor(2.5, dtype=torch.float32, requires_grad=True)
        scaled = scale_backward_loss(
            mean_loss,
            example_length=64,
            context_length=64,
        )

        self.assertEqual(scaled.item(), mean_loss.item())
        self.assertTrue(scaled.requires_grad)

    def test_short_tail_scaling_equals_sum_over_64(self) -> None:
        token_losses = torch.tensor([1.0, 2.0, 4.0, 8.0], requires_grad=True)
        mean_loss = token_losses.mean()
        scaled = scale_backward_loss(
            mean_loss,
            example_length=4,
            context_length=64,
        )

        torch.testing.assert_close(
            scaled,
            token_losses.sum() / 64,
            rtol=0.0,
            atol=0.0,
        )

    def test_tail_scaling_validation_is_exact(self) -> None:
        valid = torch.tensor(1.0)
        cases = (
            (object(), 1, 64, Phase4TypeError, "scale.mean_loss.tensor"),
            (valid, True, 64, Phase4TypeError, "scale.example_length.exact_int"),
            (valid, 1, False, Phase4TypeError, "scale.context_length.exact_int"),
            (torch.ones(1), 1, 64, Phase4ContractError, "scale.mean_loss.scalar"),
            (valid.double(), 1, 64, Phase4ContractError, "scale.mean_loss.dtype"),
            (torch.tensor(float("inf")), 1, 64, Phase4ContractError, "scale.mean_loss.finite"),
            (valid, 0, 64, Phase4ContractError, "scale.example_length.value"),
            (valid, 1, 63, Phase4ContractError, "scale.context_length.value"),
        )
        for mean_loss, length, context, error_type, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(error_type) as raised:
                    scale_backward_loss(
                        mean_loss,  # type: ignore[arg-type]
                        example_length=length,
                        context_length=context,
                    )
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_measurement_preserves_identity_raw_bytes_and_none_gradients(self) -> None:
        model = self.model()
        identities = tuple(id(parameter) for parameter in model.parameters())
        digest = model_parameter_digest(model)
        loss = measure_example_loss(
            model,
            torch.tensor([3, 1, 3]),
            torch.tensor([1, 3, 2]),
        )

        self.assertFalse(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            tuple(id(parameter) for parameter in model.parameters()),
            identities,
        )
        self.assertEqual(model_parameter_digest(model), digest)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_parameter_digest_distinguishes_positive_and_negative_zero(self) -> None:
        positive = self.model()
        negative = self.model()
        with torch.no_grad():
            positive.output_bias[0] = 0.0
            negative.output_bias[0] = -0.0

        self.assertTrue(torch.equal(positive.output_bias, negative.output_bias))
        self.assertNotEqual(
            model_parameter_digest(positive),
            model_parameter_digest(negative),
        )

    def test_measurement_refuses_existing_gradients_without_mutation(self) -> None:
        model = self.model()
        model(torch.tensor([1])).sum().backward()
        digest = model_parameter_digest(model)

        with self.assertRaises(Phase4ContractError) as raised:
            measure_example_loss(model, torch.tensor([1]), torch.tensor([2]))

        self.assertEqual(
            raised.exception.details["invariant"],
            "measurement.gradients.none_before",
        )
        self.assertEqual(model_parameter_digest(model), digest)


if __name__ == "__main__":
    unittest.main()
