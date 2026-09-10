"""Independent contract tests for the explicit Phase 6 Transformer block."""

from __future__ import annotations

import inspect
import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.model.transformer_block as block_module  # noqa: E402
from sebgpt.model import (  # noqa: E402
    AttentionContractError,
    DropoutInspection,
    ExplicitDropout,
    ExplicitLayerNorm,
    FeedForwardInspection,
    LayerNormInspection,
    MultiHeadCausalSelfAttention,
    PositionwiseFeedForward,
    TransformerBlock,
    TransformerBlockContractError,
    TransformerBlockInspection,
    TransformerBlockNumericalError,
    TransformerBlockTransactionError,
    TransformerBlockTypeError,
    explicit_gelu,
)


class FloatTensorSubclass(torch.Tensor):
    pass


def _block(**overrides: object) -> TransformerBlock:
    arguments: dict[str, object] = {
        "model_width": 32,
        "number_of_heads": 4,
        "head_width": 8,
        "hidden_width": 128,
        "max_sequence_length": 256,
        "layer_norm_epsilon": 1e-5,
        "dropout_probability": 0.1,
        "attention_seed": 5005,
        "parameter_seed": 6006,
        "attention_dropout_seed": 6007,
        "feed_forward_dropout_seed": 6008,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments.update(overrides)
    return TransformerBlock(**arguments)  # type: ignore[arg-type]


def _norm(**overrides: object) -> ExplicitLayerNorm:
    arguments: dict[str, object] = {
        "model_width": 32,
        "epsilon": 1e-5,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments.update(overrides)
    return ExplicitLayerNorm(**arguments)  # type: ignore[arg-type]


def _ffn(**overrides: object) -> PositionwiseFeedForward:
    arguments: dict[str, object] = {
        "model_width": 32,
        "hidden_width": 128,
        "seed": 6006,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments.update(overrides)
    return PositionwiseFeedForward(**arguments)  # type: ignore[arg-type]


def _dropout(seed: int = 6007, **overrides: object) -> ExplicitDropout:
    arguments: dict[str, object] = {
        "probability": 0.1,
        "seed": seed,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments.update(overrides)
    return ExplicitDropout(**arguments)  # type: ignore[arg-type]


def _input(length: int = 3, *, requires_grad: bool = False) -> torch.Tensor:
    values = torch.arange(length * 32, dtype=torch.float32).reshape(length, 32)
    values = values / 73.0 - 0.75
    return values.requires_grad_(requires_grad)


def _parameter_snapshot(module: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _assert_parameters_unchanged(
    test: unittest.TestCase,
    module: nn.Module,
    snapshot: tuple[torch.Tensor, ...],
) -> None:
    test.assertTrue(
        all(
            torch.equal(parameter, before)
            for parameter, before in zip(module.parameters(), snapshot, strict=True)
        )
    )


def _generator_states(module: TransformerBlock) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        module.attention_dropout._generator.get_state().clone(),
        module.feed_forward_dropout._generator.get_state().clone(),
    )


def _assert_states_equal(
    test: unittest.TestCase,
    observed: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    test.assertTrue(torch.equal(observed[0], expected[0]))
    test.assertTrue(torch.equal(observed[1], expected[1]))


class ConstructorAndStructureTests(unittest.TestCase):
    def test_public_signatures_have_no_defaults(self) -> None:
        callables = (
            ExplicitLayerNorm,
            PositionwiseFeedForward,
            ExplicitDropout,
            TransformerBlock,
        )
        for callable_object in callables:
            with self.subTest(callable=callable_object.__name__):
                parameters = tuple(inspect.signature(callable_object).parameters.values())
                self.assertTrue(
                    all(
                        parameter.default is inspect.Parameter.empty
                        for parameter in parameters
                    )
                )

    def test_block_constructor_type_failures_precede_allocation(self) -> None:
        cases = (
            ("model_width", True, "block.constructor.model_width.type"),
            ("number_of_heads", 4.0, "block.constructor.number_of_heads.type"),
            ("head_width", "8", "block.constructor.head_width.type"),
            ("hidden_width", False, "block.constructor.hidden_width.type"),
            ("max_sequence_length", 256.0, "block.constructor.max_sequence_length.type"),
            ("layer_norm_epsilon", 1, "block.constructor.layer_norm_epsilon.type"),
            ("dropout_probability", 1, "block.constructor.dropout_probability.type"),
            ("attention_seed", True, "block.constructor.attention_seed.type"),
            ("parameter_seed", "6006", "block.constructor.parameter_seed.type"),
            ("attention_dropout_seed", 6007.0, "block.constructor.attention_dropout_seed.type"),
            ("feed_forward_dropout_seed", None, "block.constructor.feed_forward_dropout_seed.type"),
            ("device", "cpu", "block.constructor.device.type"),
            ("dtype", "float32", "block.constructor.dtype.type"),
        )
        for name, value, invariant in cases:
            with self.subTest(name=name):
                with patch.object(block_module.torch, "ones") as ones:
                    with self.assertRaises(TransformerBlockTypeError) as raised:
                        _block(**{name: value})
                ones.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_block_constructor_value_failures_precede_allocation(self) -> None:
        cases = (
            ("model_width", 31, "block.constructor.model_width.value"),
            ("number_of_heads", 3, "block.constructor.number_of_heads.value"),
            ("head_width", 4, "block.constructor.head_width.value"),
            ("hidden_width", 64, "block.constructor.hidden_width.value"),
            ("max_sequence_length", 255, "block.constructor.max_sequence_length.value"),
            ("layer_norm_epsilon", 1e-6, "block.constructor.layer_norm_epsilon.value"),
            ("dropout_probability", 0.2, "block.constructor.dropout_probability.value"),
            ("attention_seed", 5004, "block.constructor.attention_seed.value"),
            ("parameter_seed", 6005, "block.constructor.parameter_seed.value"),
            ("attention_dropout_seed", 6008, "block.constructor.attention_dropout_seed.value"),
            ("feed_forward_dropout_seed", 6007, "block.constructor.feed_forward_dropout_seed.value"),
            ("device", torch.device("meta"), "block.constructor.device.value"),
            ("dtype", torch.float64, "block.constructor.dtype.value"),
        )
        for name, value, invariant in cases:
            with self.subTest(name=name):
                with patch.object(block_module.torch, "ones") as ones:
                    with self.assertRaises(TransformerBlockContractError) as raised:
                        _block(**{name: value})
                ones.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_component_constructor_contracts(self) -> None:
        cases = (
            (lambda: _norm(model_width=True), TransformerBlockTypeError, "normalization.constructor.model_width.type"),
            (lambda: _norm(epsilon=1), TransformerBlockTypeError, "normalization.constructor.epsilon.type"),
            (lambda: _norm(device="cpu"), TransformerBlockTypeError, "normalization.constructor.device.type"),
            (lambda: _norm(dtype="float32"), TransformerBlockTypeError, "normalization.constructor.dtype.type"),
            (lambda: _norm(model_width=31), TransformerBlockContractError, "normalization.constructor.model_width.value"),
            (lambda: _norm(epsilon=1e-6), TransformerBlockContractError, "normalization.constructor.epsilon.value"),
            (lambda: _norm(device=torch.device("meta")), TransformerBlockContractError, "normalization.constructor.device.value"),
            (lambda: _norm(dtype=torch.float64), TransformerBlockContractError, "normalization.constructor.dtype.value"),
            (lambda: _ffn(model_width=True), TransformerBlockTypeError, "ffn.constructor.model_width.type"),
            (lambda: _ffn(hidden_width=128.0), TransformerBlockTypeError, "ffn.constructor.hidden_width.type"),
            (lambda: _ffn(seed=True), TransformerBlockTypeError, "ffn.constructor.seed.type"),
            (lambda: _ffn(device="cpu"), TransformerBlockTypeError, "ffn.constructor.device.type"),
            (lambda: _ffn(dtype="float32"), TransformerBlockTypeError, "ffn.constructor.dtype.type"),
            (lambda: _ffn(model_width=31), TransformerBlockContractError, "ffn.constructor.model_width.value"),
            (lambda: _ffn(hidden_width=64), TransformerBlockContractError, "ffn.constructor.hidden_width.value"),
            (lambda: _ffn(seed=6005), TransformerBlockContractError, "ffn.constructor.seed.value"),
            (lambda: _ffn(device=torch.device("meta")), TransformerBlockContractError, "ffn.constructor.device.value"),
            (lambda: _ffn(dtype=torch.float64), TransformerBlockContractError, "ffn.constructor.dtype.value"),
            (lambda: _dropout(probability=1), TransformerBlockTypeError, "dropout.constructor.probability.type"),
            (lambda: _dropout(seed=True), TransformerBlockTypeError, "dropout.constructor.seed.type"),
            (lambda: _dropout(device="cpu"), TransformerBlockTypeError, "dropout.constructor.device.type"),
            (lambda: _dropout(dtype="float32"), TransformerBlockTypeError, "dropout.constructor.dtype.type"),
            (lambda: _dropout(probability=0.2), TransformerBlockContractError, "dropout.constructor.probability.value"),
            (lambda: _dropout(seed=6009), TransformerBlockContractError, "dropout.constructor.seed.value"),
            (lambda: _dropout(device=torch.device("meta")), TransformerBlockContractError, "dropout.constructor.device.value"),
            (lambda: _dropout(dtype=torch.float64), TransformerBlockContractError, "dropout.constructor.dtype.value"),
        )
        for factory, exception_type, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(exception_type) as raised:
                    factory()
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_constructor_simultaneous_defects_obey_types_then_values(self) -> None:
        with self.assertRaises(TransformerBlockTypeError) as raised:
            _block(model_width=31, number_of_heads="four", head_width="eight")
        self.assertEqual(
            raised.exception.details["invariant"],
            "block.constructor.number_of_heads.type",
        )
        with self.assertRaises(TransformerBlockContractError) as raised:
            _block(model_width=31, number_of_heads=3, head_width=4)
        self.assertEqual(
            raised.exception.details["invariant"],
            "block.constructor.model_width.value",
        )

    def test_exact_direct_children_parameters_counts_and_no_buffers(self) -> None:
        module = _block()
        self.assertEqual(
            tuple(name for name, _ in module.named_children()),
            (
                "norm1",
                "attention",
                "attention_dropout",
                "norm2",
                "feed_forward",
                "feed_forward_dropout",
            ),
        )
        self.assertEqual(len(module.attention.heads), 4)
        expected_names = (
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
        items = tuple(module.named_parameters())
        self.assertEqual(tuple(name for name, _ in items), expected_names)
        self.assertEqual(len(items), 21)
        self.assertEqual(sum(parameter.numel() for _, parameter in items), 12_576)
        new_names = tuple(
            name for name, _ in items if not name.startswith("attention.")
        )
        self.assertEqual(len(new_names), 8)
        self.assertEqual(
            sum(
                parameter.numel()
                for name, parameter in items
                if not name.startswith("attention.")
            ),
            8_480,
        )
        self.assertEqual(tuple(module.named_buffers()), ())
        self.assertIsNot(module.norm1.gamma, module.norm2.gamma)
        self.assertIsNot(module.norm1.beta, module.norm2.beta)

    def test_public_exports_and_phase_boundary(self) -> None:
        from sebgpt import model

        for name in (
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
        ):
            self.assertIs(getattr(model, name), getattr(block_module, name))
        source = (REPOSITORY_ROOT / "src/sebgpt/model/transformer_block.py").read_text()
        for forbidden in (
            "nn.LayerNorm",
            "nn.Dropout",
            "nn.GELU",
            "functional.gelu",
            "nn.Transformer",
            "torch.optim",
            "sebgpt.data",
            "tokenization",
        ):
            self.assertNotIn(forbidden, source)

    def test_inspection_records_are_frozen_and_hide_tensor_values(self) -> None:
        result = _block().eval().inspect(_input())
        self.assertEqual(repr(result), "TransformerBlockInspection()")
        with self.assertRaises(FrozenInstanceError):
            result.output = torch.zeros_like(result.output)  # type: ignore[misc]
        self.assertEqual(repr(_norm().inspect(_input())), "LayerNormInspection()")
        self.assertEqual(repr(_ffn().inspect(_input())), "FeedForwardInspection()")
        self.assertEqual(repr(_dropout().inspect(_input())), "DropoutInspection()")


class InitializationTests(unittest.TestCase):
    def test_norm_and_ffn_initialization_matches_independent_literals(self) -> None:
        module = _block()
        self.assertTrue(torch.equal(module.norm1.gamma, torch.ones(32)))
        self.assertTrue(torch.equal(module.norm1.beta, torch.zeros(32)))
        self.assertTrue(torch.equal(module.norm2.gamma, torch.ones(32)))
        self.assertTrue(torch.equal(module.norm2.beta, torch.zeros(32)))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(6006)
        expected_first = torch.empty((32, 128), dtype=torch.float32)
        nn.init.normal_(
            expected_first,
            mean=0.0,
            std=1.0 / math.sqrt(32),
            generator=generator,
        )
        expected_second = torch.empty((128, 32), dtype=torch.float32)
        nn.init.normal_(
            expected_second,
            mean=0.0,
            std=1.0 / math.sqrt(128),
            generator=generator,
        )
        self.assertTrue(torch.equal(module.feed_forward.first_weight, expected_first))
        self.assertTrue(torch.equal(module.feed_forward.first_bias, torch.zeros(128)))
        self.assertTrue(torch.equal(module.feed_forward.second_weight, expected_second))
        self.assertTrue(torch.equal(module.feed_forward.second_bias, torch.zeros(32)))

    def test_attention_initialization_is_unchanged_phase5_module(self) -> None:
        block = _block()
        reference = MultiHeadCausalSelfAttention(
            model_width=32,
            number_of_heads=4,
            head_width=8,
            max_sequence_length=256,
            seed=5005,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(
            all(
                left_name == right_name and torch.equal(left, right)
                for (left_name, left), (right_name, right) in zip(
                    block.attention.named_parameters(),
                    reference.named_parameters(),
                    strict=True,
                )
            )
        )

    def test_repeatability_and_global_rng_isolation(self) -> None:
        torch.manual_seed(10203)
        global_before = torch.random.get_rng_state().clone()
        first = _block()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        torch.rand(19)
        second = _block()
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first.parameters(), second.parameters(), strict=True)
            )
        )
        call_state = torch.random.get_rng_state().clone()
        first(_input())
        self.assertTrue(torch.equal(torch.random.get_rng_state(), call_state))


class NormalizationAndFeedForwardTests(unittest.TestCase):
    def test_layer_norm_matches_population_reference_and_inspection(self) -> None:
        module = _norm()
        with torch.no_grad():
            module.gamma.copy_(torch.linspace(0.5, 1.5, 32))
            module.beta.copy_(torch.linspace(-0.2, 0.2, 32))
        value = _input(2)
        inspected = module.inspect(value)
        mean = value.mean(dim=-1, keepdim=True)
        centered = value - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + 1e-5)
        expected = normalized * module.gamma + module.beta
        self.assertEqual(tuple(inspected.mean.shape), (2, 1))
        self.assertEqual(tuple(inspected.variance.shape), (2, 1))
        torch.testing.assert_close(inspected.mean, mean, rtol=0, atol=0)
        torch.testing.assert_close(inspected.variance, variance, rtol=0, atol=0)
        torch.testing.assert_close(inspected.normalized, normalized, rtol=0, atol=0)
        torch.testing.assert_close(inspected.output, expected, rtol=0, atol=0)

    def test_layer_norm_constant_row_and_position_independence(self) -> None:
        module = _norm()
        with torch.no_grad():
            module.beta.copy_(torch.linspace(-1.0, 1.0, 32))
        constant = torch.full((1, 32), 4.0, dtype=torch.float32)
        self.assertTrue(torch.equal(module(constant), module.beta.unsqueeze(0)))

        first = _input(3)
        second = first.clone()
        second[2, 7] += 100.0
        self.assertTrue(torch.equal(module(first)[:2], module(second)[:2]))

    def test_layer_norm_same_call_objects_and_gradients(self) -> None:
        module = _norm()
        value = _input(3, requires_grad=True)
        captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        original = module._compute_normalization

        def capture(tensor: torch.Tensor):
            result = original(tensor)
            captured.append(result)
            return result

        with patch.object(module, "_compute_normalization", side_effect=capture) as mocked:
            inspected = module.inspect(value)
        self.assertEqual(mocked.call_count, 1)
        self.assertIs(inspected.output, captured[0][0])
        self.assertIs(inspected.mean, captured[0][1])
        self.assertIs(inspected.variance, captured[0][2])
        self.assertIs(inspected.normalized, captured[0][3])
        inspected.output.square().sum().backward()
        for gradient in (value.grad, module.gamma.grad, module.beta.grad):
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.all(torch.isfinite(gradient)).item())

    def test_exact_erf_gelu_matches_reference_and_not_tanh_approximation(self) -> None:
        values = torch.tensor([-2.0, -0.75, 0.0, 0.75, 2.0], dtype=torch.float32)
        expected = 0.5 * values * (1.0 + torch.erf(values / math.sqrt(2.0)))
        actual = explicit_gelu(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        approximation = 0.5 * values * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (values + 0.044715 * values.pow(3))
            )
        )
        self.assertFalse(torch.equal(actual, approximation))

    def test_ffn_matches_independent_arithmetic_and_is_positionwise(self) -> None:
        module = _ffn()
        value = _input(3)
        inspected = module.inspect(value)
        expected_pre = value @ module.first_weight + module.first_bias
        expected_hidden = 0.5 * expected_pre * (
            1.0 + torch.erf(expected_pre / math.sqrt(2.0))
        )
        expected_output = expected_hidden @ module.second_weight + module.second_bias
        torch.testing.assert_close(inspected.pre_activation, expected_pre, rtol=0, atol=0)
        torch.testing.assert_close(inspected.hidden_activation, expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(inspected.output, expected_output, rtol=0, atol=0)
        changed = value.clone()
        changed[2, 0] += 9.0
        self.assertTrue(torch.equal(module(value)[:2], module(changed)[:2]))

    def test_ffn_same_call_objects_and_all_gradients(self) -> None:
        module = _ffn()
        value = _input(3, requires_grad=True)
        captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        original = module._compute_feed_forward

        def capture(tensor: torch.Tensor):
            result = original(tensor)
            captured.append(result)
            return result

        with patch.object(module, "_compute_feed_forward", side_effect=capture) as mocked:
            inspected = module.inspect(value)
        self.assertEqual(mocked.call_count, 1)
        self.assertIs(inspected.pre_activation, captured[0][0])
        self.assertIs(inspected.hidden_activation, captured[0][1])
        self.assertIs(inspected.output, captured[0][2])
        inspected.output.square().sum().backward()
        for gradient in (value.grad, *(parameter.grad for parameter in module.parameters())):
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.all(torch.isfinite(gradient)).item())
            self.assertTrue(torch.any(gradient != 0).item())

    def test_noncontiguous_and_tensor_subclass_inputs_are_accepted(self) -> None:
        value = _input(4)
        noncontiguous = value.t().contiguous().t()
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(tuple(_norm()(noncontiguous).shape), (4, 32))
        subclass = value.as_subclass(FloatTensorSubclass)
        self.assertEqual(tuple(_ffn()(subclass).shape), (4, 32))


class DropoutTests(unittest.TestCase):
    def test_training_dropout_matches_independent_local_generator(self) -> None:
        module = _dropout(6007)
        values = torch.ones((4, 32), dtype=torch.float32)
        expected_generator = torch.Generator(device="cpu")
        expected_generator.manual_seed(6007)
        uniform = torch.rand(
            values.shape,
            generator=expected_generator,
            dtype=torch.float32,
        )
        expected_mask = uniform >= 0.1
        inspected = module.inspect(values)
        self.assertTrue(torch.equal(inspected.keep_mask, expected_mask))
        self.assertTrue(
            torch.equal(
                inspected.output,
                values * expected_mask.to(torch.float32) / 0.9,
            )
        )

    def test_evaluation_is_exact_identity_and_consumes_no_rng(self) -> None:
        module = _dropout(6007)
        module.eval()
        values = _input(3)
        before = module._generator.get_state().clone()
        inspected = module.inspect(values)
        self.assertIs(inspected.output, values)
        self.assertTrue(torch.all(inspected.keep_mask).item())
        self.assertTrue(torch.equal(module._generator.get_state(), before))
        self.assertIs(module(values), values)
        self.assertTrue(torch.equal(module._generator.get_state(), before))

    def test_two_streams_are_independent_repeatable_and_advance(self) -> None:
        first_attention = _dropout(6007)
        second_attention = _dropout(6007)
        ffn = _dropout(6008)
        values = torch.ones((8, 32), dtype=torch.float32)
        first_mask = first_attention.inspect(values).keep_mask
        repeated_mask = second_attention.inspect(values).keep_mask
        second_mask = first_attention.inspect(values).keep_mask
        ffn_mask = ffn.inspect(values).keep_mask
        self.assertTrue(torch.equal(first_mask, repeated_mask))
        self.assertFalse(torch.equal(first_mask, second_mask))
        self.assertFalse(torch.equal(first_mask, ffn_mask))

    def test_only_dropout_changes_semantics_between_modes_for_both_seeds(self) -> None:
        values = torch.ones((8, 32), dtype=torch.float32)
        for seed in (6007, 6008):
            with self.subTest(seed=seed):
                training = _dropout(seed).train()
                evaluation = _dropout(seed).eval()
                training_result = training.inspect(values)
                evaluation_result = evaluation.inspect(values)
                self.assertTrue(torch.all(evaluation_result.keep_mask).item())
                self.assertIs(evaluation_result.output, values)
                self.assertFalse(torch.all(training_result.keep_mask).item())
                self.assertFalse(
                    torch.equal(training_result.output, evaluation_result.output)
                )

    def test_dropout_inspection_uses_same_call_objects(self) -> None:
        module = _dropout()
        values = _input()
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        original = module._compute_dropout

        def capture(tensor: torch.Tensor):
            result = original(tensor)
            captured.append(result)
            return result

        with patch.object(module, "_compute_dropout", side_effect=capture) as mocked:
            inspected = module.inspect(values)
        self.assertEqual(mocked.call_count, 1)
        self.assertIs(inspected.output, captured[0][0])
        self.assertIs(inspected.keep_mask, captured[0][1])

    def test_standalone_missing_and_corrupt_seed_fail_safely_before_snapshot(self) -> None:
        corruptions = (
            ("missing", lambda module: delattr(module, "seed")),
            ("wrong_type", lambda module: setattr(module, "seed", "6007")),
            ("wrong_value", lambda module: setattr(module, "seed", 6009)),
        )
        for entrypoint in ("forward", "inspect"):
            for name, corrupt in corruptions:
                with self.subTest(entrypoint=entrypoint, corruption=name):
                    module = _dropout()
                    values = _input()
                    values_before = values.clone()
                    generator_before = module._generator.get_state().clone()
                    corrupt(module)
                    messages: list[str] = []
                    with (
                        patch.object(block_module, "_read_generator_state") as read,
                        patch.object(block_module.torch, "rand") as random_draw,
                    ):
                        for _ in range(2):
                            with self.assertRaises(
                                TransformerBlockContractError
                            ) as raised:
                                getattr(module, entrypoint)(values)
                            self.assertEqual(
                                raised.exception.details["invariant"],
                                "dropout.structure",
                            )
                            messages.append(str(raised.exception))
                    read.assert_not_called()
                    random_draw.assert_not_called()
                    self.assertEqual(messages[0], messages[1])
                    self.assertTrue(torch.equal(values, values_before))
                    self.assertTrue(
                        torch.equal(
                            module._generator.get_state(),
                            generator_before,
                        )
                    )

    def test_standalone_dropout_packaging_failure_rolls_back(self) -> None:
        failed = _dropout()
        control = _dropout()
        values = _input()
        global_before = torch.random.get_rng_state().clone()
        with patch.object(block_module, "DropoutInspection", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                failed.inspect(values)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        failed_mask = failed.inspect(values).keep_mask
        control_mask = control.inspect(values).keep_mask
        self.assertTrue(torch.equal(failed_mask, control_mask))

        rollback_failed = _dropout()
        global_before = torch.random.get_rng_state().clone()
        with (
            patch.object(
                block_module,
                "DropoutInspection",
                side_effect=RuntimeError("package"),
            ),
            patch.object(
                block_module,
                "_write_generator_state",
                side_effect=RuntimeError("rollback"),
            ),
        ):
            with self.assertRaises(TransformerBlockTransactionError) as raised:
                rollback_failed.inspect(values)
        self.assertEqual(
            raised.exception.details["invariant"],
            "dropout.transaction.rollback",
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))


class BlockBehaviorTests(unittest.TestCase):
    def test_forward_shape_and_eval_topology_match_independent_composition(self) -> None:
        module = _block().eval()
        value = _input(3)
        norm1 = module.norm1(value)
        attention = module.attention(norm1)
        first_residual = value + attention
        norm2 = module.norm2(first_residual)
        pre = norm2 @ module.feed_forward.first_weight + module.feed_forward.first_bias
        hidden = 0.5 * pre * (1.0 + torch.erf(pre / math.sqrt(2.0)))
        ffn = hidden @ module.feed_forward.second_weight + module.feed_forward.second_bias
        expected = first_residual + ffn
        output = module(value)
        self.assertEqual(tuple(output.shape), (3, 32))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_inspection_exposes_exact_same_call_intermediates(self) -> None:
        module = _block().eval()
        value = _input(3)
        captured: dict[str, object] = {}

        def wrap(name: str, original):
            def call(argument: torch.Tensor):
                result = original(argument)
                captured[name] = result
                return result

            return call

        with (
            patch.object(module.norm1, "inspect", side_effect=wrap("norm1", module.norm1.inspect)) as norm1_call,
            patch.object(module.attention, "inspect", side_effect=wrap("attention", module.attention.inspect)) as attention_call,
            patch.object(module.attention_dropout, "inspect", side_effect=wrap("attention_dropout", module.attention_dropout.inspect)) as attention_dropout_call,
            patch.object(module.norm2, "inspect", side_effect=wrap("norm2", module.norm2.inspect)) as norm2_call,
            patch.object(module.feed_forward, "inspect", side_effect=wrap("ffn", module.feed_forward.inspect)) as ffn_call,
            patch.object(module.feed_forward_dropout, "inspect", side_effect=wrap("ffn_dropout", module.feed_forward_dropout.inspect)) as ffn_dropout_call,
        ):
            result = module.inspect(value)
        for mocked in (
            norm1_call,
            attention_call,
            attention_dropout_call,
            norm2_call,
            ffn_call,
            ffn_dropout_call,
        ):
            self.assertEqual(mocked.call_count, 1)
        norm1 = captured["norm1"]
        attention = captured["attention"]
        attention_dropout = captured["attention_dropout"]
        norm2 = captured["norm2"]
        ffn = captured["ffn"]
        ffn_dropout = captured["ffn_dropout"]
        assert isinstance(norm1, LayerNormInspection)
        self.assertIs(result.normalized_attention_input, norm1.output)
        self.assertIs(result.attention_branch_output_before_dropout, attention.output)  # type: ignore[union-attr]
        self.assertIs(result.attention_weights, attention.attention_weights)  # type: ignore[union-attr]
        self.assertIs(result.attention_dropout_keep_mask, attention_dropout.keep_mask)  # type: ignore[union-attr]
        self.assertIs(result.attention_branch_output_after_dropout, attention_dropout.output)  # type: ignore[union-attr]
        self.assertIs(result.normalized_feed_forward_input, norm2.output)  # type: ignore[union-attr]
        self.assertIs(result.feed_forward_pre_activation, ffn.pre_activation)  # type: ignore[union-attr]
        self.assertIs(result.feed_forward_hidden_activation, ffn.hidden_activation)  # type: ignore[union-attr]
        self.assertIs(result.feed_forward_branch_output_before_dropout, ffn.output)  # type: ignore[union-attr]
        self.assertIs(result.feed_forward_dropout_keep_mask, ffn_dropout.keep_mask)  # type: ignore[union-attr]
        self.assertIs(result.feed_forward_branch_output_after_dropout, ffn_dropout.output)  # type: ignore[union-attr]
        self.assertEqual(tuple(result.attention_weights.shape), (4, 3, 3))

    def test_first_residual_identity_value_and_gradient(self) -> None:
        module = _block().eval()
        with torch.no_grad():
            module.attention.output_weight.zero_()
        value = _input(3, requires_grad=True)
        result = module.inspect(value)
        self.assertTrue(
            torch.equal(
                result.attention_branch_output_before_dropout,
                torch.zeros_like(value),
            )
        )
        self.assertTrue(torch.equal(result.first_residual, value))
        result.first_residual.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_second_residual_identity_value_and_gradient(self) -> None:
        module = _block().eval()
        with torch.no_grad():
            module.feed_forward.second_weight.zero_()
            module.feed_forward.second_bias.zero_()
        value = _input(3, requires_grad=True)
        result = module.inspect(value)
        result.first_residual.retain_grad()
        self.assertTrue(
            torch.equal(
                result.feed_forward_branch_output_before_dropout,
                torch.zeros_like(value),
            )
        )
        self.assertTrue(torch.equal(result.output, result.first_residual))
        result.output.sum().backward()
        self.assertTrue(
            torch.equal(result.first_residual.grad, torch.ones_like(value))
        )

    def test_both_zero_branches_preserve_value_and_end_to_end_gradient(self) -> None:
        module = _block().eval()
        with torch.no_grad():
            module.attention.output_weight.zero_()
            module.feed_forward.second_weight.zero_()
            module.feed_forward.second_bias.zero_()
        value = _input(3, requires_grad=True)
        output = module(value)
        self.assertTrue(torch.equal(output, value))
        output.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_complete_block_causality_in_eval_and_training(self) -> None:
        base = _input(4)
        changed = base.clone()
        changed[3, 5] += 50.0

        evaluation = _block().eval()
        self.assertTrue(torch.equal(evaluation(base)[:3], evaluation(changed)[:3]))

        first = _block().train()
        second = _block().train()
        first_output = first(base)
        second_output = second(changed)
        self.assertTrue(torch.equal(first_output[:3], second_output[:3]))
        normalized_base = first.norm1(base)
        normalized_changed = first.norm1(changed)
        self.assertFalse(
            torch.equal(
                normalized_base.mean(dim=0),
                normalized_changed.mean(dim=0),
            )
        )

    def test_permitted_history_can_change_later_output(self) -> None:
        module = _block().eval()
        base = _input(4)
        changed = base.clone()
        changed[0, 3] += 11.0
        self.assertFalse(torch.equal(module(base)[3], module(changed)[3]))

    def test_mode_isolation_leaves_only_dropout_semantically_dependent(self) -> None:
        evaluation = _block().eval()
        training = _block().train()
        with torch.no_grad():
            evaluation.attention.output_weight.zero_()
            training.attention.output_weight.zero_()
        value = _input(4)
        eval_before = _parameter_snapshot(evaluation)
        train_before = _parameter_snapshot(training)
        eval_result = evaluation.inspect(value)
        train_result = training.inspect(value)
        for left, right in (
            (eval_result.normalized_attention_input, train_result.normalized_attention_input),
            (eval_result.attention_branch_output_before_dropout, train_result.attention_branch_output_before_dropout),
            (eval_result.attention_weights, train_result.attention_weights),
            (eval_result.first_residual, train_result.first_residual),
            (eval_result.normalized_feed_forward_input, train_result.normalized_feed_forward_input),
            (eval_result.feed_forward_pre_activation, train_result.feed_forward_pre_activation),
            (eval_result.feed_forward_hidden_activation, train_result.feed_forward_hidden_activation),
            (eval_result.feed_forward_branch_output_before_dropout, train_result.feed_forward_branch_output_before_dropout),
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.all(eval_result.feed_forward_dropout_keep_mask).item())
        self.assertFalse(torch.all(train_result.feed_forward_dropout_keep_mask).item())
        self.assertFalse(
            torch.equal(
                eval_result.feed_forward_branch_output_after_dropout,
                train_result.feed_forward_branch_output_after_dropout,
            )
        )
        _assert_parameters_unchanged(self, evaluation, eval_before)
        _assert_parameters_unchanged(self, training, train_before)

    def test_nondegenerate_output_reaches_all_parameters_and_input(self) -> None:
        module = _block().eval()
        value = _input(4, requires_grad=True)
        parameters_before = tuple(module.parameters())
        values_before = tuple(
            parameter.detach().clone() for parameter in parameters_before
        )
        shapes_before = tuple(tuple(parameter.shape) for parameter in parameters_before)
        devices_before = tuple(parameter.device for parameter in parameters_before)
        dtypes_before = tuple(parameter.dtype for parameter in parameters_before)
        coefficients = torch.linspace(0.2, 1.7, 128, dtype=torch.float32).reshape(4, 32)
        loss = (module(value) * coefficients).sum()
        loss.backward()
        gradients = (value.grad, *(parameter.grad for parameter in module.parameters()))
        self.assertEqual(len(gradients), 22)
        for position, gradient in enumerate(gradients):
            with self.subTest(position=position):
                self.assertIsNotNone(gradient)
                assert gradient is not None
                self.assertTrue(torch.all(torch.isfinite(gradient)).item())
                self.assertTrue(torch.any(gradient != 0).item())
        parameters_after = tuple(module.parameters())
        self.assertEqual(len(parameters_after), 21)
        for position, (before_object, after_object) in enumerate(
            zip(parameters_before, parameters_after, strict=True)
        ):
            with self.subTest(parameter_position=position):
                self.assertIs(after_object, before_object)
                self.assertTrue(torch.equal(after_object, values_before[position]))
                self.assertEqual(tuple(after_object.shape), shapes_before[position])
                self.assertEqual(after_object.device, devices_before[position])
                self.assertIs(after_object.dtype, dtypes_before[position])


class ValidationAndNumericalTests(unittest.TestCase):
    def test_boundary_lengths_succeed_and_batch_is_rejected(self) -> None:
        module = _block().eval()
        self.assertEqual(tuple(module(torch.zeros((1, 32))).shape), (1, 32))
        self.assertEqual(tuple(module(torch.zeros((256, 32))).shape), (256, 32))
        with self.assertRaises(TransformerBlockContractError) as raised:
            module(torch.zeros((1, 2, 32)))
        self.assertEqual(raised.exception.details["invariant"], "block.input.rank")

    def test_block_input_failures_have_literal_invariants_and_do_not_draw(self) -> None:
        cases = (
            ([], "block.input.tensor"),
            (torch.zeros(32), "block.input.rank"),
            (torch.zeros((1, 32), dtype=torch.float64), "block.input.dtype"),
            (torch.empty((1, 32), device="meta"), "block.input.device"),
            (torch.zeros((1, 31)), "block.input.model_width"),
            (torch.zeros((0, 32)), "block.input.sequence_length"),
            (torch.zeros((257, 32)), "block.input.sequence_length"),
            (torch.full((1, 32), float("nan")), "block.input.finite"),
        )
        for value, invariant in cases:
            with self.subTest(invariant=invariant):
                module = _block()
                before = _generator_states(module)
                with self.assertRaises((TransformerBlockTypeError, TransformerBlockContractError, TransformerBlockNumericalError)) as raised:
                    module(value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)
                _assert_states_equal(self, _generator_states(module), before)

    def test_component_input_contracts_and_gelu_contract(self) -> None:
        for factory, prefix in ((_norm, "normalization"), (_ffn, "ffn"), (_dropout, "dropout")):
            cases = (
                ([], f"{prefix}.input.tensor"),
                (torch.zeros(32), f"{prefix}.input.rank"),
                (torch.zeros((1, 32), dtype=torch.float64), f"{prefix}.input.dtype"),
                (torch.empty((1, 32), device="meta"), f"{prefix}.input.device"),
                (torch.zeros((1, 31)), f"{prefix}.input.model_width"),
                (torch.zeros((0, 32)), f"{prefix}.input.sequence_length"),
                (torch.full((1, 32), float("inf")), f"{prefix}.input.finite"),
            )
            for value, invariant in cases:
                with self.subTest(component=prefix, invariant=invariant):
                    with self.assertRaises(
                        (
                            TransformerBlockTypeError,
                            TransformerBlockContractError,
                            TransformerBlockNumericalError,
                        )
                    ) as raised:
                        factory()(value)  # type: ignore[arg-type,operator]
                    self.assertEqual(raised.exception.details["invariant"], invariant)

        gelu_cases = (
            ([], "gelu.input.tensor"),
            (torch.ones(1, dtype=torch.float64), "gelu.input.dtype"),
            (torch.empty(1, device="meta"), "gelu.input.device"),
            (torch.full((1,), float("nan")), "gelu.input.finite"),
        )
        for value, invariant in gelu_cases:
            with self.subTest(component="gelu", invariant=invariant):
                with self.assertRaises(
                    (
                        TransformerBlockTypeError,
                        TransformerBlockContractError,
                        TransformerBlockNumericalError,
                    )
                ) as raised:
                    explicit_gelu(value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_corrupted_mode_structure_and_parameter_fail_before_snapshot(self) -> None:
        module = _block()
        module.feed_forward_dropout.training = "yes"  # type: ignore[assignment]
        with patch.object(block_module, "_read_generator_state") as read:
            with self.assertRaises(TransformerBlockTypeError) as raised:
                module(_input())
        read.assert_not_called()
        self.assertEqual(raised.exception.details["invariant"], "dropout.mode.type")

        module = _block()
        module.feed_forward_dropout.eval()
        with patch.object(block_module, "_read_generator_state") as read:
            with self.assertRaises(TransformerBlockContractError) as raised:
                module(_input())
        read.assert_not_called()
        self.assertEqual(raised.exception.details["invariant"], "block.dropout_modes.consistent")

        module = _block()
        module.feed_forward_dropout._generator = object()  # type: ignore[assignment]
        with patch.object(block_module, "_read_generator_state") as read:
            with self.assertRaises(TransformerBlockTypeError) as raised:
                module(_input())
        read.assert_not_called()
        self.assertEqual(raised.exception.details["invariant"], "dropout.structure")

        module = _block()
        module.norm1.gamma = nn.Parameter(torch.ones(31))
        with patch.object(block_module, "_read_generator_state") as read:
            with self.assertRaises(TransformerBlockContractError) as raised:
                module(_input())
        read.assert_not_called()
        self.assertEqual(raised.exception.details["invariant"], "block.parameter.norm1.gamma.shape")

    def test_new_parameter_validation_is_category_major_and_attention_owns_its_errors(self) -> None:
        module = _block()
        with torch.no_grad():
            module.norm1.gamma.fill_(float("nan"))
        module.norm2.gamma = nn.Parameter(torch.ones(31))
        with self.assertRaises(TransformerBlockContractError) as raised:
            module(_input())
        self.assertEqual(
            raised.exception.details["invariant"],
            "block.parameter.norm2.gamma.shape",
        )

        module = _block()
        module.attention.output_weight = nn.Parameter(torch.ones((31, 32)))
        before = _generator_states(module)
        with self.assertRaises(AttentionContractError) as raised:
            module(_input())
        self.assertEqual(
            raised.exception.details["invariant"],
            "multi.parameter.output_weight.shape",
        )
        _assert_states_equal(self, _generator_states(module), before)

    def test_new_parameter_type_shape_device_dtype_finiteness_and_structure(self) -> None:
        cases = (
            (
                "type",
                lambda module: module.norm1._parameters.__setitem__(
                    "gamma", torch.ones(32)
                ),
                TransformerBlockTypeError,
                "block.parameter.norm1.gamma.type",
            ),
            (
                "shape",
                lambda module: setattr(
                    module.norm1, "gamma", nn.Parameter(torch.ones(31))
                ),
                TransformerBlockContractError,
                "block.parameter.norm1.gamma.shape",
            ),
            (
                "device",
                lambda module: setattr(
                    module.norm1,
                    "gamma",
                    nn.Parameter(torch.ones(32, device="meta")),
                ),
                TransformerBlockContractError,
                "block.parameter.norm1.gamma.device",
            ),
            (
                "dtype",
                lambda module: setattr(
                    module.norm1,
                    "gamma",
                    nn.Parameter(torch.ones(32, dtype=torch.float64)),
                ),
                TransformerBlockContractError,
                "block.parameter.norm1.gamma.dtype",
            ),
            (
                "finite",
                lambda module: module.norm1.gamma.data.fill_(float("nan")),
                TransformerBlockNumericalError,
                "block.parameter.norm1.gamma.finite",
            ),
            (
                "structure",
                lambda module: module.register_parameter(
                    "extra", nn.Parameter(torch.zeros(1))
                ),
                TransformerBlockContractError,
                "block.parameters.structure",
            ),
        )
        for name, corrupt, exception_type, invariant in cases:
            with self.subTest(name=name):
                module = _block()
                corrupt(module)
                with patch.object(block_module, "_read_generator_state") as read:
                    with self.assertRaises(exception_type) as raised:
                        module(_input())
                read.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_freezing_and_compatible_state_restore_are_permitted(self) -> None:
        original = _block().eval()
        restored = _block().eval()
        restored.load_state_dict(original.state_dict())
        restored.requires_grad_(False)
        value = _input()
        self.assertTrue(torch.equal(original(value), restored(value)))
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))

    def test_rejected_call_preserves_input_parameters_and_rng(self) -> None:
        module = _block()
        value = torch.full((1, 32), float("nan"))
        value_before = value.clone()
        parameters_before = _parameter_snapshot(module)
        states_before = _generator_states(module)
        with self.assertRaises(TransformerBlockNumericalError):
            module(value)
        torch.testing.assert_close(value, value_before, rtol=0, atol=0, equal_nan=True)
        _assert_parameters_unchanged(self, module, parameters_before)
        _assert_states_equal(self, _generator_states(module), states_before)

    def test_normalization_gelu_ffn_and_dropout_numerical_guards(self) -> None:
        extreme = torch.zeros((1, 32), dtype=torch.float32)
        extreme[0, 0] = 1e20
        extreme[0, 1] = -1e20
        with self.assertRaises(TransformerBlockNumericalError) as raised:
            _norm()(extreme)
        self.assertEqual(raised.exception.details["invariant"], "normalization.squared.finite")

        with patch.object(block_module.torch, "erf", return_value=torch.full((1,), float("nan"))):
            with self.assertRaises(TransformerBlockNumericalError) as raised:
                explicit_gelu(torch.ones(1, dtype=torch.float32))
        self.assertEqual(raised.exception.details["invariant"], "gelu.erf.finite")

        ffn = _ffn()
        with torch.no_grad():
            ffn.first_weight.fill_(torch.finfo(torch.float32).max)
        with self.assertRaises(TransformerBlockNumericalError) as raised:
            ffn(torch.ones((1, 32), dtype=torch.float32))
        self.assertEqual(raised.exception.details["invariant"], "ffn.pre_activation.finite")

        dropout = _dropout()
        with self.assertRaises(TransformerBlockNumericalError) as raised:
            dropout(torch.full((4, 32), torch.finfo(torch.float32).max))
        self.assertEqual(raised.exception.details["invariant"], "dropout.output.finite")

    def test_residual_overflow_guards(self) -> None:
        module = _block().eval()
        maximum = torch.full((2, 32), torch.finfo(torch.float32).max)
        with (
            patch.object(module.norm1, "forward", return_value=torch.zeros_like(maximum)),
            patch.object(module.attention, "forward", return_value=maximum),
        ):
            with self.assertRaises(TransformerBlockNumericalError) as raised:
                module(maximum)
        self.assertEqual(raised.exception.details["invariant"], "block.first_residual.finite")

        module = _block().eval()
        first = torch.full((2, 32), torch.finfo(torch.float32).max)
        with (
            patch.object(module.norm1, "forward", return_value=torch.zeros_like(first)),
            patch.object(module.attention, "forward", return_value=torch.zeros_like(first)),
            patch.object(module.norm2, "forward", return_value=torch.zeros_like(first)),
            patch.object(module.feed_forward, "forward", return_value=first),
        ):
            with self.assertRaises(TransformerBlockNumericalError) as raised:
                module(first)
        self.assertEqual(raised.exception.details["invariant"], "block.output.finite")

    def test_every_documented_numerical_guard_is_called_in_operation_order(self) -> None:
        norm_targets = (
            "normalization.mean.finite",
            "normalization.centered.finite",
            "normalization.squared.finite",
            "normalization.variance.finite",
            "normalization.variance_plus_epsilon.finite",
            "normalization.inverse_std.finite",
            "normalization.normalized.finite",
            "normalization.gamma_product.finite",
            "normalization.output.finite",
        )
        gelu_targets = (
            "gelu.scaled.finite",
            "gelu.erf.finite",
            "gelu.one_plus_erf.finite",
            "gelu.half_input.finite",
            "gelu.output.finite",
        )
        ffn_targets = (
            "ffn.pre_activation.finite",
            "ffn.hidden_activation.finite",
            "ffn.output.finite",
        )
        dropout_targets = (
            "dropout.uniform.finite",
            "dropout.output.finite",
        )
        block_targets = (
            "block.attention_dropout_output.finite",
            "block.first_residual.finite",
            "block.feed_forward_dropout_output.finite",
            "block.output.finite",
        )
        cases = (
            (lambda value: _norm()(value), _input(), norm_targets),
            (explicit_gelu, torch.tensor([-1.0, 0.5]), gelu_targets),
            (lambda value: _ffn()(value), _input(), ffn_targets),
            (lambda value: _dropout()(value), _input(), dropout_targets),
            (lambda value: _block().eval()(value), _input(), block_targets),
        )
        original = block_module._require_finite
        for operation, value, targets in cases:
            for target in targets:
                with self.subTest(target=target):
                    def fail_at(tensor: torch.Tensor, invariant: str, *, expected: str = target) -> None:
                        if invariant == expected:
                            raise TransformerBlockNumericalError(invariant)
                        original(tensor, invariant)

                    with patch.object(block_module, "_require_finite", side_effect=fail_at):
                        with self.assertRaises(TransformerBlockNumericalError) as raised:
                            operation(value)
                    self.assertEqual(raised.exception.details["invariant"], target)

        original_contract = block_module._require_contract

        for target in ("dropout.uniform.range", "dropout.keep_mask.structure"):
            with self.subTest(target=target):
                def fail_contract(
                    condition: bool,
                    invariant: str,
                    *,
                    expected: str = target,
                    **facts: object,
                ) -> None:
                    if invariant == expected:
                        raise TransformerBlockContractError(invariant)
                    original_contract(condition, invariant, **facts)

                with patch.object(
                    block_module,
                    "_require_contract",
                    side_effect=fail_contract,
                ):
                    with self.assertRaises(TransformerBlockContractError) as raised:
                        _dropout()(_input())
                self.assertEqual(raised.exception.details["invariant"], target)

        original_positive = block_module._require_positive

        def fail_positive(tensor: torch.Tensor, invariant: str) -> None:
            if invariant == "normalization.variance_plus_epsilon.positive":
                raise TransformerBlockNumericalError(invariant)
            original_positive(tensor, invariant)

        with patch.object(block_module, "_require_positive", side_effect=fail_positive):
            with self.assertRaises(TransformerBlockNumericalError) as raised:
                _norm()(_input())
        self.assertEqual(
            raised.exception.details["invariant"],
            "normalization.variance_plus_epsilon.positive",
        )

        static_orders = (
            (
                ExplicitLayerNorm._compute_normalization,
                (
                    "normalization.mean.finite",
                    "normalization.centered.finite",
                    "normalization.squared.finite",
                    "normalization.variance.finite",
                    "normalization.variance_plus_epsilon.finite",
                    "normalization.variance_plus_epsilon.positive",
                    "normalization.inverse_std.finite",
                    "normalization.normalized.finite",
                    "normalization.gamma_product.finite",
                    "normalization.output.finite",
                ),
            ),
            (
                explicit_gelu,
                (
                    "gelu.scaled.finite",
                    "gelu.erf.finite",
                    "gelu.one_plus_erf.finite",
                    "gelu.half_input.finite",
                    "gelu.output.finite",
                ),
            ),
            (
                PositionwiseFeedForward._compute_feed_forward,
                (
                    "ffn.pre_activation.finite",
                    "ffn.hidden_activation.finite",
                    "ffn.output.finite",
                ),
            ),
            (
                ExplicitDropout._compute_dropout,
                (
                    "dropout.uniform.finite",
                    "dropout.uniform.range",
                    "dropout.keep_mask.structure",
                    "dropout.output.finite",
                ),
            ),
            (
                TransformerBlock._compute_forward,
                (
                    "block.attention_dropout_output.finite",
                    "block.first_residual.finite",
                    "block.feed_forward_dropout_output.finite",
                    "block.output.finite",
                ),
            ),
        )
        for callable_object, invariants in static_orders:
            with self.subTest(callable=callable_object.__name__):
                source = inspect.getsource(callable_object)
                positions = tuple(source.index(invariant) for invariant in invariants)
                self.assertEqual(positions, tuple(sorted(positions)))

    def test_errors_are_content_safe_and_deterministic(self) -> None:
        module = _block()
        value = torch.full((1, 32), float("nan"))
        messages: list[str] = []
        for _ in range(2):
            with self.assertRaises(TransformerBlockNumericalError) as raised:
                module(value)
            messages.append(str(raised.exception))
            self.assertEqual(raised.exception.details["invariant"], "block.input.finite")
        self.assertEqual(messages[0], messages[1])
        self.assertNotIn("nan", messages[0].lower())
        with self.assertRaises(TypeError):
            raised.exception.details["invariant"] = "changed"  # type: ignore[index]

class TransactionTests(unittest.TestCase):
    def _assert_failed_call_restores_both(
        self,
        method_name: str,
        stage: str,
    ) -> None:
        failed = _block().train()
        control = _block().train()
        value = _input(3)
        global_before = torch.random.get_rng_state().clone()
        original_finite = block_module._require_finite

        def final_failure(tensor: torch.Tensor, invariant: str) -> None:
            if invariant == "block.output.finite":
                raise RuntimeError("after-both")
            original_finite(tensor, invariant)

        method = getattr(failed, method_name)
        if stage == "before":
            target = "inspect" if method_name == "inspect" else "forward"
            context = patch.object(failed.attention, target, side_effect=RuntimeError("before"))
        elif stage == "after_attention":
            target = "inspect" if method_name == "inspect" else "forward"
            context = patch.object(failed.norm2, target, side_effect=RuntimeError("after-one"))
        else:
            context = patch.object(block_module, "_require_finite", side_effect=final_failure)
        with context:
            with self.assertRaises(RuntimeError):
                method(value)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        failed_result = failed.inspect(value)
        control_result = control.inspect(value)
        self.assertTrue(
            torch.equal(
                failed_result.attention_dropout_keep_mask,
                control_result.attention_dropout_keep_mask,
            )
        )
        self.assertTrue(
            torch.equal(
                failed_result.feed_forward_dropout_keep_mask,
                control_result.feed_forward_dropout_keep_mask,
            )
        )

    def test_forward_transaction_restores_before_after_one_and_after_both(self) -> None:
        for stage in ("before", "after_attention", "after_both"):
            with self.subTest(stage=stage):
                self._assert_failed_call_restores_both("forward", stage)

    def test_inspect_transaction_restores_before_after_one_and_after_both(self) -> None:
        for stage in ("before", "after_attention", "after_both"):
            with self.subTest(stage=stage):
                self._assert_failed_call_restores_both("inspect", stage)

    def test_inspection_packaging_failure_restores_both(self) -> None:
        failed = _block().train()
        control = _block().train()
        value = _input(3)
        global_before = torch.random.get_rng_state().clone()
        with patch.object(block_module, "TransformerBlockInspection", side_effect=RuntimeError("package")):
            with self.assertRaisesRegex(RuntimeError, "package"):
                failed.inspect(value)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        failed_result = failed.inspect(value)
        control_result = control.inspect(value)
        self.assertTrue(torch.equal(failed_result.attention_dropout_keep_mask, control_result.attention_dropout_keep_mask))
        self.assertTrue(torch.equal(failed_result.feed_forward_dropout_keep_mask, control_result.feed_forward_dropout_keep_mask))

    def test_snapshot_failure_is_owned_and_draws_nothing(self) -> None:
        for stream, invariant in (
            ("attention", "block.dropout_transaction.snapshot.attention"),
            ("feed_forward", "block.dropout_transaction.snapshot.feed_forward"),
        ):
            with self.subTest(stream=stream):
                module = _block().train()
                before = _generator_states(module)
                global_before = torch.random.get_rng_state().clone()
                original_read = block_module._read_generator_state
                target = (
                    module.attention_dropout._generator
                    if stream == "attention"
                    else module.feed_forward_dropout._generator
                )

                def fail_target(generator: torch.Generator) -> torch.Tensor:
                    if generator is target:
                        raise RuntimeError("snapshot")
                    return original_read(generator)

                with patch.object(
                    block_module,
                    "_read_generator_state",
                    side_effect=fail_target,
                ):
                    with self.assertRaises(TransformerBlockTransactionError) as raised:
                        module(_input())
                self.assertEqual(raised.exception.details["invariant"], invariant)
                _assert_states_equal(self, _generator_states(module), before)
                self.assertTrue(
                    torch.equal(torch.random.get_rng_state(), global_before)
                )

    def test_rollback_attempts_both_and_owns_restore_failure(self) -> None:
        for failing_names in (
            ("attention",),
            ("feed_forward",),
            ("attention", "feed_forward"),
        ):
            with self.subTest(failing_names=failing_names):
                module = _block().train()
                value = _input()
                global_before = torch.random.get_rng_state().clone()
                original_write = block_module._write_generator_state
                calls: list[torch.Generator] = []

                def fail_selected(
                    generator: torch.Generator,
                    state: torch.Tensor,
                ) -> None:
                    calls.append(generator)
                    name = (
                        "attention"
                        if generator is module.attention_dropout._generator
                        else "feed_forward"
                    )
                    if name in failing_names:
                        raise RuntimeError("restore")
                    original_write(generator, state)

                with (
                    patch.object(
                        module.norm2,
                        "forward",
                        side_effect=RuntimeError("original"),
                    ),
                    patch.object(
                        block_module,
                        "_write_generator_state",
                        side_effect=fail_selected,
                    ),
                ):
                    with self.assertRaises(TransformerBlockTransactionError) as raised:
                        module(value)
                self.assertEqual(
                    raised.exception.details["invariant"],
                    "block.dropout_transaction.rollback",
                )
                self.assertEqual(
                    raised.exception.details["failed_streams"],
                    failing_names,
                )
                self.assertIsInstance(raised.exception.__cause__, RuntimeError)
                self.assertEqual(
                    calls,
                    [
                        module.attention_dropout._generator,
                        module.feed_forward_dropout._generator,
                    ],
                )
                self.assertTrue(
                    torch.equal(torch.random.get_rng_state(), global_before)
                )


if __name__ == "__main__":
    unittest.main()
