"""Independent contract tests for transparent Phase 5 causal self-attention."""

from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.model.self_attention as attention_module  # noqa: E402
from sebgpt.model import (  # noqa: E402
    AttentionContractError,
    AttentionInspection,
    AttentionNumericalError,
    AttentionTypeError,
    MultiHeadCausalSelfAttention,
    SingleHeadCausalSelfAttention,
)


class FloatTensorSubclass(torch.Tensor):
    pass


def _single(**overrides: object) -> SingleHeadCausalSelfAttention:
    arguments: dict[str, object] = {
        "model_width": 32,
        "max_sequence_length": 256,
        "seed": 5005,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments.update(overrides)
    return SingleHeadCausalSelfAttention(**arguments)  # type: ignore[arg-type]


def _multi(**overrides: object) -> MultiHeadCausalSelfAttention:
    arguments: dict[str, object] = {
        "model_width": 32,
        "number_of_heads": 4,
        "head_width": 8,
        "max_sequence_length": 256,
        "seed": 5005,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments.update(overrides)
    return MultiHeadCausalSelfAttention(**arguments)  # type: ignore[arg-type]


def _input(length: int = 3, *, requires_grad: bool = False) -> torch.Tensor:
    value = torch.arange(length * 32, dtype=torch.float32).reshape(length, 32)
    value = value / 97.0 - 0.5
    return value.requires_grad_(requires_grad)


def _reference_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor:
    scores = query @ key.transpose(0, 1)
    scaled = scores / math.sqrt(width)
    positions = torch.arange(query.shape[0], dtype=torch.long)
    allowed = positions[None, :] <= positions[:, None]
    masked = scaled.masked_fill(~allowed, -torch.inf)
    shifted = masked - masked.max(dim=-1, keepdim=True).values
    numerator = torch.exp(shifted)
    return numerator / numerator.sum(dim=-1, keepdim=True)


def _snapshot(module: nn.Module, value: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    return value.clone(), tuple(parameter.detach().clone() for parameter in module.parameters())


def _assert_unchanged(
    test: unittest.TestCase,
    module: nn.Module,
    value: torch.Tensor,
    snapshot: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
) -> None:
    input_before, parameters_before = snapshot
    torch.testing.assert_close(
        value,
        input_before,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    test.assertTrue(
        all(
            torch.equal(parameter, before)
            for parameter, before in zip(
                module.parameters(), parameters_before, strict=True
            )
        )
    )


def _set_uniform_single(module: SingleHeadCausalSelfAttention) -> None:
    with torch.no_grad():
        module.query_weight.zero_()
        module.key_weight.zero_()
        module.value_weight.copy_(torch.eye(32, dtype=torch.float32))


def _set_uniform_multi(module: MultiHeadCausalSelfAttention) -> None:
    with torch.no_grad():
        for index, head in enumerate(module.heads):
            head.query_weight.zero_()
            head.key_weight.zero_()
            head.value_weight.zero_()
            head.value_weight[index * 8 : (index + 1) * 8, :].copy_(
                torch.eye(8, dtype=torch.float32)
            )
        module.output_weight.copy_(torch.eye(32, dtype=torch.float32))


class ConstructorAndInitializationTests(unittest.TestCase):
    def test_constructor_type_failures_precede_allocation_with_literal_invariants(self) -> None:
        single_cases = (
            ("model_width", True, "single.constructor.model_width.type"),
            ("max_sequence_length", 256.0, "single.constructor.max_sequence_length.type"),
            ("seed", "5005", "single.constructor.seed.type"),
            ("device", "cpu", "single.constructor.device.type"),
            ("dtype", "float32", "single.constructor.dtype.type"),
        )
        multi_cases = (
            ("model_width", False, "multi.constructor.model_width.type"),
            ("number_of_heads", 4.0, "multi.constructor.number_of_heads.type"),
            ("head_width", "8", "multi.constructor.head_width.type"),
            ("max_sequence_length", None, "multi.constructor.max_sequence_length.type"),
            ("seed", True, "multi.constructor.seed.type"),
            ("device", object(), "multi.constructor.device.type"),
            ("dtype", object(), "multi.constructor.dtype.type"),
        )
        for name, value, invariant in single_cases:
            with self.subTest(module="single", name=name):
                with patch.object(attention_module.torch, "empty") as empty:
                    with self.assertRaises(AttentionTypeError) as raised:
                        _single(**{name: value})
                empty.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)
        for name, value, invariant in multi_cases:
            with self.subTest(module="multi", name=name):
                with patch.object(attention_module.torch, "empty") as empty:
                    with self.assertRaises(AttentionTypeError) as raised:
                        _multi(**{name: value})
                empty.assert_not_called()
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_constructor_value_failures_precede_allocation_with_literal_invariants(self) -> None:
        single_cases = (
            ("model_width", 31, "single.constructor.model_width.value"),
            ("max_sequence_length", 255, "single.constructor.max_sequence_length.value"),
            ("seed", 5004, "single.constructor.seed.value"),
            ("device", torch.device("meta"), "single.constructor.device.value"),
            ("dtype", torch.float64, "single.constructor.dtype.value"),
        )
        multi_cases = (
            ("model_width", 31, "multi.constructor.model_width.value"),
            ("number_of_heads", 3, "multi.constructor.number_of_heads.value"),
            ("head_width", 4, "multi.constructor.head_width.value"),
            ("max_sequence_length", 255, "multi.constructor.max_sequence_length.value"),
            ("seed", 5004, "multi.constructor.seed.value"),
            ("device", torch.device("meta"), "multi.constructor.device.value"),
            ("dtype", torch.float64, "multi.constructor.dtype.value"),
        )
        for factory, cases in ((_single, single_cases), (_multi, multi_cases)):
            for name, value, invariant in cases:
                with self.subTest(factory=factory.__name__, name=name):
                    with patch.object(attention_module.torch, "empty") as empty:
                        with self.assertRaises(AttentionContractError) as raised:
                            factory(**{name: value})
                    empty.assert_not_called()
                    self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_constructor_combined_defects_obey_all_types_then_values_order(self) -> None:
        with self.assertRaises(AttentionTypeError) as raised:
            _multi(model_width=31, number_of_heads="four", head_width="eight")
        self.assertEqual(
            raised.exception.details["invariant"],
            "multi.constructor.number_of_heads.type",
        )
        with self.assertRaises(AttentionContractError) as raised:
            _multi(model_width=31, number_of_heads=3, head_width=4)
        self.assertEqual(
            raised.exception.details["invariant"],
            "multi.constructor.model_width.value",
        )

    def test_exact_parameter_enumeration_counts_shapes_modules_and_no_buffers(self) -> None:
        single = _single()
        multi = _multi()
        self.assertEqual(
            tuple(name for name, _ in single.named_parameters()),
            ("query_weight", "key_weight", "value_weight"),
        )
        self.assertEqual(
            tuple(name for name, _ in multi.named_parameters()),
            (
                "output_weight",
                "heads.0.query_weight",
                "heads.0.key_weight",
                "heads.0.value_weight",
                "heads.1.query_weight",
                "heads.1.key_weight",
                "heads.1.value_weight",
                "heads.2.query_weight",
                "heads.2.key_weight",
                "heads.2.value_weight",
                "heads.3.query_weight",
                "heads.3.key_weight",
                "heads.3.value_weight",
            ),
        )
        self.assertEqual(sum(parameter.numel() for parameter in single.parameters()), 3072)
        self.assertEqual(sum(parameter.numel() for parameter in multi.parameters()), 4096)
        self.assertEqual(tuple(single.named_buffers()), ())
        self.assertEqual(tuple(multi.named_buffers()), ())
        self.assertEqual(len(multi.heads), 4)
        self.assertTrue(all(parameter.requires_grad for parameter in single.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in multi.parameters()))
        self.assertEqual(tuple(single.query_weight.shape), (32, 32))
        self.assertEqual(tuple(single.key_weight.shape), (32, 32))
        self.assertEqual(tuple(single.value_weight.shape), (32, 32))
        self.assertEqual(tuple(multi.output_weight.shape), (32, 32))
        self.assertTrue(
            all(
                tuple(parameter.shape) == (32, 8)
                for head in multi.heads
                for parameter in (head.query_weight, head.key_weight, head.value_weight)
            )
        )

    def test_single_initialization_matches_literal_independent_generator_bitwise(self) -> None:
        module = _single()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(5005)
        expected = []
        for _ in range(3):
            tensor = torch.empty((32, 32), device="cpu", dtype=torch.float32)
            torch.nn.init.normal_(
                tensor,
                mean=0.0,
                std=1.0 / math.sqrt(32),
                generator=generator,
            )
            expected.append(tensor)
        actual = (module.query_weight, module.key_weight, module.value_weight)
        self.assertTrue(all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True)))

    def test_multi_initialization_matches_literal_head_major_generator_bitwise(self) -> None:
        module = _multi()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(5005)
        expected_heads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(4):
            values = []
            for _ in range(3):
                tensor = torch.empty((32, 8), device="cpu", dtype=torch.float32)
                torch.nn.init.normal_(
                    tensor,
                    mean=0.0,
                    std=1.0 / math.sqrt(32),
                    generator=generator,
                )
                values.append(tensor)
            expected_heads.append((values[0], values[1], values[2]))
        expected_output = torch.empty((32, 32), device="cpu", dtype=torch.float32)
        torch.nn.init.normal_(
            expected_output,
            mean=0.0,
            std=1.0 / math.sqrt(32),
            generator=generator,
        )
        for head, expected in zip(module.heads, expected_heads, strict=True):
            actual = (head.query_weight, head.key_weight, head.value_weight)
            self.assertTrue(all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True)))
        self.assertTrue(torch.equal(module.output_weight, expected_output))
        equal_shaped = tuple(
            parameter
            for head in module.heads
            for parameter in (head.query_weight, head.key_weight, head.value_weight)
        )
        self.assertTrue(any(not torch.equal(equal_shaped[0], value) for value in equal_shaped[1:]))

    def test_construction_is_repeatable_and_isolated_from_global_rng(self) -> None:
        before = torch.get_rng_state().clone()
        try:
            first_single = _single()
            first_multi = _multi()
            after = torch.get_rng_state().clone()
            torch.rand(73)
            second_single = _single()
            second_multi = _multi()
        finally:
            torch.set_rng_state(before)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first_single.parameters(), second_single.parameters(), strict=True
                )
            )
        )
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    first_multi.parameters(), second_multi.parameters(), strict=True
                )
            )
        )


class ValidationAndSafetyTests(unittest.TestCase):
    def test_input_validation_order_and_literal_invariants(self) -> None:
        module = _single()
        cases = (
            (object(), AttentionTypeError, "attention.input.tensor"),
            (torch.tensor(1.0), AttentionContractError, "attention.input.rank"),
            (torch.zeros((1, 32), dtype=torch.float64), AttentionContractError, "attention.input.dtype"),
            (torch.empty((1, 32), device="meta"), AttentionContractError, "attention.input.device"),
            (torch.zeros((1, 31)), AttentionContractError, "attention.input.model_width"),
            (torch.zeros((0, 32)), AttentionContractError, "attention.input.sequence_length"),
            (torch.zeros((257, 32)), AttentionContractError, "attention.input.sequence_length"),
            (torch.full((1, 32), torch.nan), AttentionNumericalError, "attention.input.finite"),
        )
        for value, exception, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(exception) as raised:
                    module(value)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_combined_input_defects_obey_first_failure_order(self) -> None:
        module = _single()
        combined = (
            (torch.zeros((1, 1, 32), dtype=torch.float64), "attention.input.rank"),
            (torch.empty((1, 31), dtype=torch.float64, device="meta"), "attention.input.dtype"),
            (torch.empty((0, 31), device="meta"), "attention.input.device"),
            (torch.zeros((0, 31)), "attention.input.model_width"),
        )
        for value, invariant in combined:
            with self.subTest(invariant=invariant):
                with self.assertRaises(AttentionContractError) as raised:
                    module.inspect(value)
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_tensor_subclass_and_noncontiguous_input_are_accepted(self) -> None:
        module = _single()
        subclass = _input().as_subclass(FloatTensorSubclass)
        noncontiguous = torch.arange(64, dtype=torch.float32).reshape(32, 2).t()
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(tuple(module(subclass).shape), (3, 32))
        self.assertEqual(tuple(module(noncontiguous).shape), (2, 32))

    def test_maximum_length_output_device_dtype_and_nonaliasing(self) -> None:
        for module in (_single(), _multi()):
            value = torch.zeros((256, 32), dtype=torch.float32)
            output = module(value)
            self.assertEqual(tuple(output.shape), (256, 32))
            self.assertEqual(output.device, torch.device("cpu"))
            self.assertIs(output.dtype, torch.float32)
            self.assertNotEqual(
                output.untyped_storage().data_ptr(), value.untyped_storage().data_ptr()
            )
            for parameter in module.parameters():
                self.assertNotEqual(
                    output.untyped_storage().data_ptr(),
                    parameter.untyped_storage().data_ptr(),
                )

    def test_forward_and_inspect_do_not_mutate_inputs_or_parameters(self) -> None:
        for module in (_single(), _multi()):
            value = _input()
            before = _snapshot(module, value)
            module(value)
            module.inspect(value)
            _assert_unchanged(self, module, value, before)

    def test_structure_type_shape_dtype_device_and_finiteness_errors_are_stable(self) -> None:
        structure = _single()
        structure.extra = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        with self.assertRaises(AttentionContractError) as raised:
            structure(_input())
        self.assertEqual(raised.exception.details["invariant"], "single.parameters.structure")

        wrong_type = _single()
        wrong_type._parameters["query_weight"] = torch.zeros((32, 32))  # type: ignore[assignment]
        with self.assertRaises(AttentionTypeError) as raised:
            wrong_type(_input())
        self.assertEqual(raised.exception.details["invariant"], "single.parameter.query_weight.type")

        replacements = (
            (nn.Parameter(torch.zeros((31, 32))), AttentionContractError, "single.parameter.query_weight.shape"),
            (nn.Parameter(torch.zeros((32, 32), dtype=torch.float64)), AttentionContractError, "single.parameter.query_weight.dtype"),
            (nn.Parameter(torch.empty((32, 32), device="meta")), AttentionContractError, "single.parameter.query_weight.device"),
            (nn.Parameter(torch.full((32, 32), torch.nan)), AttentionNumericalError, "single.parameter.query_weight.finite"),
        )
        for replacement, exception, invariant in replacements:
            with self.subTest(invariant=invariant):
                module = _single()
                module.query_weight = replacement
                with self.assertRaises(exception) as raised:
                    module(_input())
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_multi_semantic_shape_validation_is_head_first_not_enumeration_order(self) -> None:
        module = _multi()
        module.output_weight = nn.Parameter(torch.zeros((31, 32)))
        module.heads[0].query_weight = nn.Parameter(torch.zeros((31, 8)))
        with self.assertRaises(AttentionContractError) as raised:
            module(_input())
        self.assertEqual(
            raised.exception.details["invariant"],
            "multi.parameter.heads.0.query_weight.shape",
        )

    def test_parameter_validation_is_category_major_then_semantic_order(self) -> None:
        single = _single()
        single.query_weight = nn.Parameter(torch.zeros((31, 32)))
        single._parameters["key_weight"] = torch.zeros((32, 32))  # type: ignore[assignment]
        with self.assertRaises(AttentionTypeError) as raised:
            single(_input())
        self.assertEqual(
            raised.exception.details["invariant"],
            "single.parameter.key_weight.type",
        )

        multi = _multi()
        multi.heads[0].query_weight = nn.Parameter(torch.zeros((31, 8)))
        multi.heads[0]._parameters["key_weight"] = torch.zeros((32, 8))  # type: ignore[assignment]
        with self.assertRaises(AttentionTypeError) as raised:
            multi(_input())
        self.assertEqual(
            raised.exception.details["invariant"],
            "multi.parameter.heads.0.key_weight.type",
        )

    def test_multi_parameter_failures_use_literal_stable_invariants(self) -> None:
        cases = (
            (
                lambda module: setattr(
                    module.heads[2],
                    "value_weight",
                    nn.Parameter(torch.zeros((32, 7))),
                ),
                AttentionContractError,
                "multi.parameter.heads.2.value_weight.shape",
            ),
            (
                lambda module: setattr(
                    module.heads[1],
                    "key_weight",
                    nn.Parameter(torch.zeros((32, 8), dtype=torch.float64)),
                ),
                AttentionContractError,
                "multi.parameter.heads.1.key_weight.dtype",
            ),
            (
                lambda module: setattr(
                    module.heads[3],
                    "query_weight",
                    nn.Parameter(torch.empty((32, 8), device="meta")),
                ),
                AttentionContractError,
                "multi.parameter.heads.3.query_weight.device",
            ),
            (
                lambda module: setattr(
                    module,
                    "output_weight",
                    nn.Parameter(torch.full((32, 32), torch.inf)),
                ),
                AttentionNumericalError,
                "multi.parameter.output_weight.finite",
            ),
        )
        for mutate, exception, invariant in cases:
            with self.subTest(invariant=invariant):
                module = _multi()
                mutate(module)
                with self.assertRaises(exception) as raised:
                    module(_input())
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_parameter_freezing_and_compatible_state_restore_are_permitted(self) -> None:
        for factory in (_single, _multi):
            original = factory()
            restored = factory()
            restored.load_state_dict(original.state_dict())
            for parameter in restored.parameters():
                parameter.requires_grad_(False)
            self.assertEqual(tuple(restored(_input()).shape), (3, 32))

    def test_rejected_call_is_nonmutating_and_diagnostics_are_content_safe(self) -> None:
        module = _single()
        value = torch.full((2, 32), 1234567.0)
        value[0, 0] = torch.nan
        before = _snapshot(module, value)
        with self.assertRaises(AttentionNumericalError) as raised:
            module(value)
        _assert_unchanged(self, module, value, before)
        self.assertEqual(raised.exception.details["invariant"], "attention.input.finite")
        self.assertNotIn("1234567", str(raised.exception))
        self.assertEqual(set(raised.exception.details), {"invariant", "shape"})


class MathematicsAndInspectionTests(unittest.TestCase):
    def test_single_head_matches_independent_qkv_score_weight_and_output_reference(self) -> None:
        module = _single()
        value = _input(4)
        inspection_result = module.inspect(value)
        query = value @ module.query_weight
        key = value @ module.key_weight
        projected_value = value @ module.value_weight
        expected_weights = _reference_weights(query, key, width=32)
        expected_output = expected_weights @ projected_value
        self.assertEqual(tuple(query.shape), (4, 32))
        self.assertEqual(tuple(key.shape), (4, 32))
        self.assertEqual(tuple(projected_value.shape), (4, 32))
        self.assertEqual(tuple(inspection_result.attention_weights.shape), (4, 4))
        self.assertEqual(tuple(inspection_result.output.shape), (4, 32))
        self.assertTrue(torch.allclose(inspection_result.attention_weights, expected_weights, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(inspection_result.output, expected_output, rtol=1e-5, atol=1e-6))

    def test_single_causal_weights_have_exact_zeros_and_normalized_rows(self) -> None:
        weights = _single().inspect(_input(5)).attention_weights
        future = torch.triu(torch.ones((5, 5), dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.equal(weights[future], torch.zeros_like(weights[future])))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(5), rtol=0, atol=1e-6))

    def test_single_length_one_is_exact(self) -> None:
        module = _single()
        value = _input(1)
        result = module.inspect(value)
        projected_value = value @ module.value_weight
        self.assertTrue(torch.equal(result.attention_weights, torch.tensor([[1.0]])))
        self.assertTrue(torch.equal(result.output, projected_value))

    def test_multi_head_matches_independent_per_head_concat_and_projection_reference(self) -> None:
        module = _multi()
        value = _input(4)
        result = module.inspect(value)
        expected_head_outputs = []
        expected_weights = []
        for head in module.heads:
            query = value @ head.query_weight
            key = value @ head.key_weight
            projected_value = value @ head.value_weight
            weights = _reference_weights(query, key, width=8)
            expected_weights.append(weights)
            expected_head_outputs.append(weights @ projected_value)
            self.assertEqual(tuple(query.shape), (4, 8))
            self.assertEqual(tuple(key.shape), (4, 8))
            self.assertEqual(tuple(projected_value.shape), (4, 8))
        concatenated = torch.cat(tuple(expected_head_outputs), dim=-1)
        expected_output = concatenated @ module.output_weight
        self.assertEqual(tuple(concatenated.shape), (4, 32))
        self.assertEqual(tuple(result.output.shape), (4, 32))
        self.assertEqual(tuple(result.attention_weights.shape), (4, 4, 4))
        self.assertTrue(torch.allclose(result.attention_weights, torch.stack(tuple(expected_weights)), rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(result.output, expected_output, rtol=1e-5, atol=1e-6))

    def test_multi_head_weights_preserve_heads_causality_and_normalization(self) -> None:
        weights = _multi().inspect(_input(5)).attention_weights
        future = torch.triu(torch.ones((5, 5), dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.equal(weights[:, future], torch.zeros_like(weights[:, future])))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones((4, 5)), rtol=0, atol=1e-6))

    def test_forward_returns_only_output_and_inspection_repr_hides_values(self) -> None:
        value = _input()
        for module, weight_shape in ((_single(), (3, 3)), (_multi(), (4, 3, 3))):
            output = module(value)
            result = module.inspect(value)
            self.assertIsInstance(output, torch.Tensor)
            self.assertNotIsInstance(output, AttentionInspection)
            self.assertEqual(tuple(result.attention_weights.shape), weight_shape)
            self.assertEqual(repr(result), "AttentionInspection()")

    def test_inspect_calls_private_boundary_once_and_preserves_returned_identities(self) -> None:
        for module in (_single(), _multi()):
            original = module._compute_attention
            captured: dict[str, object] = {"calls": 0}

            def wrapper(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                captured["calls"] = int(captured["calls"]) + 1
                if captured["calls"] > 1:
                    raise AssertionError("diagnostic recomputation")
                output, weights = original(value)
                captured["output"] = output
                captured["weights"] = weights
                return output, weights

            with patch.object(module, "_compute_attention", side_effect=wrapper) as boundary:
                result = module.inspect(_input())
            self.assertEqual(boundary.call_count, 1)
            self.assertEqual(captured["calls"], 1)
            self.assertIs(result.output, captured["output"])
            self.assertIs(result.attention_weights, captured["weights"])

    def test_inspection_weights_retain_qk_and_input_autograd_without_v_requirement(self) -> None:
        single = _single()
        single_input = _input(4, requires_grad=True)
        single_weights = single.inspect(single_input).attention_weights
        coefficients = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        (single_weights * coefficients).sum().backward()
        for gradient in (
            single_input.grad,
            single.query_weight.grad,
            single.key_weight.grad,
        ):
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.all(torch.isfinite(gradient)))
            self.assertTrue(torch.any(gradient != 0))
        self.assertIsNone(single.value_weight.grad)

        multi = _multi()
        multi_input = _input(4, requires_grad=True)
        multi_weights = multi.inspect(multi_input).attention_weights
        head_coefficients = torch.stack(
            tuple(coefficients + head_index * 17 for head_index in range(4))
        )
        (multi_weights * head_coefficients).sum().backward()
        self.assertIsNotNone(multi_input.grad)
        assert multi_input.grad is not None
        self.assertTrue(torch.all(torch.isfinite(multi_input.grad)))
        self.assertTrue(torch.any(multi_input.grad != 0))
        for head in multi.heads:
            for gradient in (head.query_weight.grad, head.key_weight.grad):
                self.assertIsNotNone(gradient)
                assert gradient is not None
                self.assertTrue(torch.all(torch.isfinite(gradient)))
                self.assertTrue(torch.any(gradient != 0))
            self.assertIsNone(head.value_weight.grad)
        self.assertIsNone(multi.output_weight.grad)

    def test_output_backward_reaches_input_and_every_attention_parameter(self) -> None:
        for module in (_single(), _multi()):
            value = _input(4, requires_grad=True)
            output = module(value)
            coefficients = torch.arange(output.numel(), dtype=torch.float32).reshape_as(output) + 1
            (output * coefficients).sum().backward()
            self.assertIsNotNone(value.grad)
            assert value.grad is not None
            self.assertTrue(torch.all(torch.isfinite(value.grad)))
            self.assertTrue(torch.any(value.grad != 0))
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                assert parameter.grad is not None
                self.assertEqual(tuple(parameter.grad.shape), tuple(parameter.shape))
                self.assertTrue(torch.all(torch.isfinite(parameter.grad)))
                self.assertTrue(torch.any(parameter.grad != 0))


class CausalityTests(unittest.TestCase):
    def test_single_future_perturbation_preserves_prefix_exactly_but_unmasked_reference_changes(self) -> None:
        module = _single()
        _set_uniform_single(module)
        original = _input(4)
        changed = original.clone()
        changed[3] += 10000.0
        original_output = module(original)
        changed_output = module(changed)
        self.assertTrue(torch.equal(original_output[:3], changed_output[:3]))
        original_unmasked = original.mean(dim=0, keepdim=True).expand(4, 32)
        changed_unmasked = changed.mean(dim=0, keepdim=True).expand(4, 32)
        self.assertFalse(torch.equal(original_unmasked[:3], changed_unmasked[:3]))

    def test_multi_future_perturbation_preserves_prefix_exactly_but_unmasked_reference_changes(self) -> None:
        module = _multi()
        _set_uniform_multi(module)
        original = _input(4)
        changed = original.clone()
        changed[3] -= 10000.0
        original_output = module(original)
        changed_output = module(changed)
        self.assertTrue(torch.equal(original_output[:3], changed_output[:3]))
        original_unmasked = original.mean(dim=0, keepdim=True).expand(4, 32)
        changed_unmasked = changed.mean(dim=0, keepdim=True).expand(4, 32)
        self.assertFalse(torch.equal(original_unmasked[:3], changed_unmasked[:3]))

    def test_allowed_earlier_change_affects_later_single_and_multi_outputs(self) -> None:
        original = _input(4)
        changed = original.clone()
        changed[0] += 50.0
        for module, configure in (
            (_single(), _set_uniform_single),
            (_multi(), _set_uniform_multi),
        ):
            configure(module)  # type: ignore[arg-type]
            before = module(original)
            after = module(changed)
            self.assertFalse(torch.equal(before[3], after[3]))


class NumericalGuardTests(unittest.TestCase):
    def _assert_numerical_failure(
        self,
        module: nn.Module,
        value: torch.Tensor,
        invariant: str,
    ) -> None:
        before = _snapshot(module, value)
        with self.assertRaises(AttentionNumericalError) as raised:
            module(value)
        self.assertEqual(raised.exception.details["invariant"], invariant)
        self.assertNotIn("tensor", str(raised.exception).lower())
        _assert_unchanged(self, module, value, before)

    def test_single_projection_overflow_stops_at_exact_q_k_v_stage(self) -> None:
        for target, invariant in (
            ("query_weight", "single.projection.query.finite"),
            ("key_weight", "single.projection.key.finite"),
            ("value_weight", "single.projection.value.finite"),
        ):
            with self.subTest(target=target):
                module = _single()
                with torch.no_grad():
                    module.query_weight.zero_()
                    module.key_weight.zero_()
                    module.value_weight.zero_()
                    getattr(module, target).fill_(1.0)
                self._assert_numerical_failure(
                    module,
                    torch.full((1, 32), torch.finfo(torch.float32).max),
                    invariant,
                )

    def test_each_multi_head_projection_overflow_has_exact_stage(self) -> None:
        for head_index in range(4):
            for target, role in (
                ("query_weight", "query"),
                ("key_weight", "key"),
                ("value_weight", "value"),
            ):
                with self.subTest(head=head_index, target=target):
                    module = _multi()
                    with torch.no_grad():
                        for head in module.heads:
                            head.query_weight.zero_()
                            head.key_weight.zero_()
                            head.value_weight.zero_()
                        module.output_weight.zero_()
                        getattr(module.heads[head_index], target).fill_(1.0)
                    self._assert_numerical_failure(
                        module,
                        torch.full((1, 32), torch.finfo(torch.float32).max),
                        f"multi.head.{head_index}.projection.{role}.finite",
                    )

    def test_finite_projections_can_fail_at_raw_score_overflow(self) -> None:
        module = _single()
        with torch.no_grad():
            module.query_weight.fill_(1.0e18)
            module.key_weight.fill_(1.0e18)
            module.value_weight.zero_()
        value = torch.ones((1, 32), dtype=torch.float32)
        query = value @ module.query_weight
        key = value @ module.key_weight
        self.assertTrue(torch.all(torch.isfinite(query)))
        self.assertTrue(torch.all(torch.isfinite(key)))
        self._assert_numerical_failure(module, value, "single.scores.finite")

    def test_multi_output_projection_overflow_is_detected(self) -> None:
        module = _multi()
        with torch.no_grad():
            for head in module.heads:
                head.query_weight.zero_()
                head.key_weight.zero_()
                head.value_weight.fill_(1.0e18)
            module.output_weight.fill_(1.0e20)
        value = torch.ones((1, 32), dtype=torch.float32)
        self._assert_numerical_failure(module, value, "multi.output_projection.finite")

    def test_mask_created_negative_infinity_is_expected_and_future_weights_are_zero(self) -> None:
        original_checker = attention_module._require_finite
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []

        def checker(
            value: torch.Tensor,
            invariant: str,
            *,
            permitted: torch.Tensor | None = None,
        ) -> None:
            if invariant == "single.masked_scores.permitted_finite":
                assert permitted is not None
                captured.append((value.detach().clone(), permitted.detach().clone()))
            original_checker(value, invariant, permitted=permitted)

        with patch.object(attention_module, "_require_finite", side_effect=checker):
            weights = _single().inspect(_input(4)).attention_weights
        self.assertEqual(len(captured), 1)
        masked, allowed = captured[0]
        self.assertTrue(torch.all(torch.isneginf(masked[~allowed])))
        self.assertTrue(torch.all(torch.isfinite(masked[allowed])))
        self.assertTrue(torch.equal(weights[~allowed], torch.zeros_like(weights[~allowed])))

    def test_shifted_score_overflow_uses_exact_invariant(self) -> None:
        maximum = torch.finfo(torch.float32).max
        scores = torch.tensor([[0.0, 0.0], [-maximum, maximum]], dtype=torch.float32)
        allowed = torch.tensor([[True, False], [True, True]])
        with self.assertRaises(AttentionNumericalError) as raised:
            attention_module._stable_masked_softmax(scores, allowed, prefix="single")
        self.assertEqual(
            raised.exception.details["invariant"],
            "single.shifted_scores.permitted_finite",
        )

    def test_successful_numerical_guard_sequence_and_defensive_checker_branches(self) -> None:
        original_finite = attention_module._require_finite
        original_positive = attention_module._require_positive
        observed: list[str] = []

        def finite(
            value: torch.Tensor,
            invariant: str,
            *,
            permitted: torch.Tensor | None = None,
        ) -> None:
            observed.append(invariant)
            original_finite(value, invariant, permitted=permitted)

        def positive(value: torch.Tensor, invariant: str) -> None:
            observed.append(invariant)
            original_positive(value, invariant)

        with (
            patch.object(attention_module, "_require_finite", side_effect=finite),
            patch.object(attention_module, "_require_positive", side_effect=positive),
        ):
            _single()(_input(3))
        expected_sequence = (
            "attention.input.finite",
            "single.parameter.query_weight.finite",
            "single.parameter.key_weight.finite",
            "single.parameter.value_weight.finite",
            "single.projection.query.finite",
            "single.projection.key.finite",
            "single.projection.value.finite",
            "single.scores.finite",
            "single.scaled_scores.finite",
            "single.masked_scores.permitted_finite",
            "single.row_max.finite",
            "single.shifted_scores.permitted_finite",
            "single.exponentials.finite",
            "single.denominator.finite",
            "single.denominator.positive",
            "single.weights.finite",
            "single.output.finite",
        )
        self.assertEqual(tuple(observed), expected_sequence)

        defensive = (
            "single.scaled_scores.finite",
            "single.masked_scores.permitted_finite",
            "single.row_max.finite",
            "single.exponentials.finite",
            "single.denominator.finite",
            "single.weights.finite",
            "single.output.finite",
            "multi.concatenated.finite",
        )
        for invariant in defensive:
            with self.subTest(invariant=invariant):
                with self.assertRaises(AttentionNumericalError) as raised:
                    attention_module._require_finite(torch.tensor([torch.nan]), invariant)
                self.assertEqual(raised.exception.details["invariant"], invariant)
        with self.assertRaises(AttentionNumericalError) as raised:
            attention_module._require_positive(
                torch.tensor([0.0]), "single.denominator.positive"
            )
        self.assertEqual(
            raised.exception.details["invariant"], "single.denominator.positive"
        )

    def test_multi_head_guard_sequence_is_head_major_then_output(self) -> None:
        original_finite = attention_module._require_finite
        original_positive = attention_module._require_positive
        observed: list[str] = []

        def finite(
            value: torch.Tensor,
            invariant: str,
            *,
            permitted: torch.Tensor | None = None,
        ) -> None:
            observed.append(invariant)
            original_finite(value, invariant, permitted=permitted)

        def positive(value: torch.Tensor, invariant: str) -> None:
            observed.append(invariant)
            original_positive(value, invariant)

        with (
            patch.object(attention_module, "_require_finite", side_effect=finite),
            patch.object(attention_module, "_require_positive", side_effect=positive),
        ):
            _multi()(_input(2))

        parameter_checks = tuple(
            f"multi.parameter.heads.{head}.{role}_weight.finite"
            for head in range(4)
            for role in ("query", "key", "value")
        ) + ("multi.parameter.output_weight.finite",)
        arithmetic_checks = tuple(
            invariant
            for head in range(4)
            for invariant in (
                f"multi.head.{head}.projection.query.finite",
                f"multi.head.{head}.projection.key.finite",
                f"multi.head.{head}.projection.value.finite",
                f"multi.head.{head}.scores.finite",
                f"multi.head.{head}.scaled_scores.finite",
                f"multi.head.{head}.masked_scores.permitted_finite",
                f"multi.head.{head}.row_max.finite",
                f"multi.head.{head}.shifted_scores.permitted_finite",
                f"multi.head.{head}.exponentials.finite",
                f"multi.head.{head}.denominator.finite",
                f"multi.head.{head}.denominator.positive",
                f"multi.head.{head}.weights.finite",
                f"multi.head.{head}.output.finite",
            )
        )
        expected = (
            "attention.input.finite",
            *parameter_checks,
            *arithmetic_checks,
            "multi.concatenated.finite",
            "multi.output_projection.finite",
        )
        self.assertEqual(tuple(observed), expected)

    def test_every_multi_head_defensive_sentinel_fails_safely_without_mutation(self) -> None:
        finite_suffixes = (
            "scaled_scores.finite",
            "masked_scores.permitted_finite",
            "row_max.finite",
            "shifted_scores.permitted_finite",
            "exponentials.finite",
            "denominator.finite",
            "weights.finite",
            "output.finite",
        )
        for head_index in range(4):
            for suffix in finite_suffixes:
                invariant = f"multi.head.{head_index}.{suffix}"
                with self.subTest(head=head_index, invariant=invariant):
                    module = _multi()
                    value = _input(2)
                    snapshot = _snapshot(module, value)
                    sentinel = torch.tensor([torch.nan], dtype=torch.float32)
                    messages = []
                    for _ in range(2):
                        with self.assertRaises(AttentionNumericalError) as raised:
                            if suffix == "masked_scores.permitted_finite":
                                attention_module._require_finite(
                                    sentinel,
                                    invariant,
                                    permitted=torch.tensor([True]),
                                )
                            else:
                                attention_module._require_finite(sentinel, invariant)
                        self.assertEqual(raised.exception.details["invariant"], invariant)
                        self.assertEqual(
                            set(raised.exception.details), {"invariant", "shape"}
                        )
                        self.assertNotIn("nan", str(raised.exception).lower())
                        messages.append(str(raised.exception))
                    self.assertEqual(messages[0], messages[1])
                    torch.testing.assert_close(
                        sentinel,
                        torch.tensor([torch.nan], dtype=torch.float32),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                    _assert_unchanged(self, module, value, snapshot)

            positive_invariant = f"multi.head.{head_index}.denominator.positive"
            with self.subTest(head=head_index, invariant=positive_invariant):
                module = _multi()
                value = _input(2)
                snapshot = _snapshot(module, value)
                sentinel = torch.tensor([0.0], dtype=torch.float32)
                messages = []
                for _ in range(2):
                    with self.assertRaises(AttentionNumericalError) as raised:
                        attention_module._require_positive(
                            sentinel,
                            positive_invariant,
                        )
                    self.assertEqual(
                        raised.exception.details["invariant"], positive_invariant
                    )
                    self.assertEqual(
                        set(raised.exception.details), {"invariant", "shape"}
                    )
                    self.assertNotIn("0.0", str(raised.exception))
                    messages.append(str(raised.exception))
                self.assertEqual(messages[0], messages[1])
                self.assertTrue(torch.equal(sentinel, torch.tensor([0.0])))
                _assert_unchanged(self, module, value, snapshot)


class BoundaryAndExportTests(unittest.TestCase):
    def test_source_has_no_forbidden_phase_or_high_level_attention_mechanism(self) -> None:
        source = inspect.getsource(attention_module)
        forbidden = (
            "MultiheadAttention",
            "scaled_dot_product_attention",
            "torch.optim",
            "LayerNorm",
            "Dropout",
            "Transformer",
            "checkpoint",
            "generate",
            "sebgpt.data",
            "tokenization",
        )
        for text in forbidden:
            with self.subTest(text=text):
                self.assertNotIn(text, source)

    def test_public_exports_are_exactly_available(self) -> None:
        self.assertIs(attention_module.AttentionContractError, AttentionContractError)
        self.assertIs(attention_module.AttentionInspection, AttentionInspection)
        self.assertIs(attention_module.AttentionNumericalError, AttentionNumericalError)
        self.assertIs(attention_module.AttentionTypeError, AttentionTypeError)
        self.assertIs(attention_module.MultiHeadCausalSelfAttention, MultiHeadCausalSelfAttention)
        self.assertIs(attention_module.SingleHeadCausalSelfAttention, SingleHeadCausalSelfAttention)


if __name__ == "__main__":
    unittest.main()
