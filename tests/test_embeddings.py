"""Contract tests for explicit token and learned position embeddings."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.model.embeddings import (  # noqa: E402
    EmbeddingContractError,
    EmbeddingTypeError,
    TokenPositionEmbedding,
)
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)


class LongTensorSubclass(torch.Tensor):
    pass


class TokenPositionEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = load_accepted_vocabulary_binding(REPOSITORY_ROOT)

    def _module(self, **overrides) -> TokenPositionEmbedding:
        arguments = {
            "embedding_dim": 32,
            "max_positions": 256,
            "seed": 1337,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        arguments.update(overrides)
        return TokenPositionEmbedding(self.binding, **arguments)

    def test_08_exact_parameter_names_order_and_no_buffers(self) -> None:
        module = self._module()

        self.assertEqual(
            tuple(name for name, _ in module.named_parameters()),
            ("token_embeddings", "position_embeddings"),
        )
        self.assertEqual(tuple(module.named_buffers()), ())

    def test_09_parameter_shapes_are_exact(self) -> None:
        module = self._module()

        self.assertEqual(tuple(module.token_embeddings.shape), (81, 32))
        self.assertEqual(tuple(module.position_embeddings.shape), (256, 32))

    def test_10_parameter_counts_are_exact(self) -> None:
        module = self._module()
        token_count = module.token_embeddings.numel()
        position_count = module.position_embeddings.numel()

        self.assertEqual(token_count, 81 * 32)
        self.assertEqual(position_count, 256 * 32)
        self.assertEqual(token_count + position_count, 10784)

    def test_11_parameters_are_cpu_float32_and_learnable(self) -> None:
        module = self._module()

        for parameter in module.parameters():
            self.assertEqual(parameter.device, torch.device("cpu"))
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(parameter.requires_grad)

    def test_12_no_hidden_parameters_and_invalid_configuration_precedes_init(self) -> None:
        module = self._module()
        self.assertEqual(len(tuple(module.parameters())), 2)
        self.assertFalse(hasattr(module, "bias"))
        self.assertFalse(hasattr(module, "projection"))
        self.assertFalse(hasattr(module, "dropout"))
        self.assertFalse(hasattr(module, "normalization"))

        cases = (
            {"embedding_dim": 16},
            {"embedding_dim": True},
            {"max_positions": 255},
            {"max_positions": True},
            {"seed": 1},
            {"seed": True},
            {"device": "cpu"},
            {"device": torch.device("meta")},
            {"dtype": torch.float64},
        )
        for override in cases:
            with self.subTest(override=override):
                with patch("torch.nn.init.normal_") as initializer:
                    with self.assertRaises(EmbeddingContractError):
                        self._module(**override)
                initializer.assert_not_called()
        with patch("torch.nn.init.normal_") as initializer:
            with self.assertRaises(EmbeddingContractError) as binding_error:
                TokenPositionEmbedding(  # type: ignore[arg-type]
                    object(),
                    embedding_dim=32,
                    max_positions=256,
                    seed=1337,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
        initializer.assert_not_called()
        self.assertEqual(binding_error.exception.details["invariant"], "vocabulary.binding")

    def test_13_test_local_generator_reproduces_both_tables(self) -> None:
        module = self._module()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1337)
        standard_deviation = 1.0 / math.sqrt(32)
        expected_tokens = torch.empty((81, 32), dtype=torch.float32, device="cpu")
        torch.nn.init.normal_(
            expected_tokens,
            mean=0.0,
            std=standard_deviation,
            generator=generator,
        )
        expected_positions = torch.empty((256, 32), dtype=torch.float32, device="cpu")
        torch.nn.init.normal_(
            expected_positions,
            mean=0.0,
            std=standard_deviation,
            generator=generator,
        )

        self.assertTrue(torch.equal(module.token_embeddings, expected_tokens))
        self.assertTrue(torch.equal(module.position_embeddings, expected_positions))

    def test_14_repeated_constructions_are_bitwise_equal(self) -> None:
        first = self._module()
        second = self._module()

        self.assertTrue(torch.equal(first.token_embeddings, second.token_embeddings))
        self.assertTrue(torch.equal(first.position_embeddings, second.position_embeddings))

    def test_15_global_rng_consumption_does_not_change_construction(self) -> None:
        global_state = torch.get_rng_state().clone()
        try:
            first = self._module()
            torch.rand(4096)
            second = self._module()
        finally:
            torch.set_rng_state(global_state)

        self.assertTrue(torch.equal(first.token_embeddings, second.token_embeddings))
        self.assertTrue(torch.equal(first.position_embeddings, second.position_embeddings))

    def test_16_construction_preserves_global_cpu_rng_state(self) -> None:
        before = torch.get_rng_state().clone()
        self._module()
        after = torch.get_rng_state()

        self.assertTrue(torch.equal(before, after))

    def test_17_reversing_generator_draw_order_changes_tables(self) -> None:
        module = self._module()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1337)
        standard_deviation = 1.0 / math.sqrt(32)
        reversed_positions = torch.empty((256, 32), dtype=torch.float32)
        torch.nn.init.normal_(
            reversed_positions,
            mean=0.0,
            std=standard_deviation,
            generator=generator,
        )
        reversed_tokens = torch.empty((81, 32), dtype=torch.float32)
        torch.nn.init.normal_(
            reversed_tokens,
            mean=0.0,
            std=standard_deviation,
            generator=generator,
        )

        self.assertFalse(torch.equal(module.token_embeddings, reversed_tokens))
        self.assertFalse(torch.equal(module.position_embeddings, reversed_positions))

    def test_18_rank_one_and_rank_two_output_shapes(self) -> None:
        module = self._module()

        self.assertEqual(tuple(module(torch.tensor([0, 1, 2])).shape), (3, 32))
        self.assertEqual(
            tuple(module(torch.tensor([[0, 1, 2], [3, 4, 5]])).shape),
            (2, 3, 32),
        )

    def test_19_repeated_and_boundary_ids_select_exact_token_rows(self) -> None:
        module = self._module()
        token_ids = torch.tensor([0, 80, 3, 3], dtype=torch.long)
        selected = module.token_embeddings[token_ids]

        self.assertTrue(torch.equal(selected[0], module.token_embeddings[0]))
        self.assertTrue(torch.equal(selected[1], module.token_embeddings[80]))
        self.assertTrue(torch.equal(selected[2], selected[3]))

    def test_20_negative_and_equal_size_ids_are_rejected(self) -> None:
        module = self._module()
        for token_id in (-1, 81):
            with self.subTest(token_id=token_id):
                with self.assertRaises(EmbeddingContractError) as raised:
                    module(torch.tensor([0, token_id], dtype=torch.long))
                self.assertEqual(raised.exception.details["invariant"], "input.token_id_range")

    def test_21_rank_dtype_and_wrong_device_errors_are_ordered(self) -> None:
        module = self._module()
        cases = (
            (torch.tensor(0, dtype=torch.long), "input.rank"),
            (torch.zeros((1, 1, 1), dtype=torch.long), "input.rank"),
            (torch.tensor([0.0]), "input.dtype"),
            (torch.tensor([True]), "input.dtype"),
            (torch.tensor([0], dtype=torch.int32), "input.dtype"),
            (torch.empty((1,), dtype=torch.long, device="meta"), "input.device"),
        )
        for value, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(EmbeddingContractError) as raised:
                    module(value)
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_22_non_tensor_inputs_fail_without_coercion(self) -> None:
        class NoTensorCoercion:
            called = False

            def __torch_function__(self, *args, **kwargs):
                self.called = True
                raise AssertionError("must not coerce")

        value = NoTensorCoercion()
        module = self._module()

        with self.assertRaises(EmbeddingTypeError) as raised:
            module(value)  # type: ignore[arg-type]

        self.assertFalse(value.called)
        self.assertEqual(raised.exception.details["invariant"], "input.tensor")

    def test_23_non_contiguous_and_tensor_subclass_inputs_are_accepted(self) -> None:
        module = self._module()
        non_contiguous = torch.tensor(
            [[0, 1], [2, 3], [4, 5]],
            dtype=torch.long,
        ).t()
        subclass = torch.tensor([0, 1, 2], dtype=torch.long).as_subclass(
            LongTensorSubclass
        )

        self.assertFalse(non_contiguous.is_contiguous())
        self.assertEqual(tuple(module(non_contiguous).shape), (2, 3, 32))
        self.assertEqual(tuple(module(subclass).shape), (3, 32))

    def test_24_forward_does_not_mutate_input(self) -> None:
        module = self._module()
        token_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        before = token_ids.clone()

        module(token_ids)

        self.assertTrue(torch.equal(token_ids, before))

    def test_25_internal_positions_are_exactly_zero_through_t_minus_one(self) -> None:
        module = self._module()
        with torch.no_grad():
            module.token_embeddings.zero_()
        token_ids = torch.tensor([4, 4, 4], dtype=torch.long)
        output = module(token_ids)
        expected_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        self.assertTrue(torch.equal(output, module.position_embeddings[expected_ids]))

    def test_26_batch_members_share_position_vectors(self) -> None:
        module = self._module()
        with torch.no_grad():
            module.token_embeddings.zero_()
        token_ids = torch.tensor([[5, 5, 5], [5, 5, 5]], dtype=torch.long)
        output = module(token_ids)

        self.assertTrue(torch.equal(output[0], output[1]))
        self.assertTrue(torch.equal(output[0], module.position_embeddings[:3]))

    def test_27_maximum_position_boundary_accepts_256_and_rejects_257(self) -> None:
        module = self._module()
        accepted = torch.zeros((256,), dtype=torch.long)
        rejected = torch.zeros((257,), dtype=torch.long)

        self.assertEqual(tuple(module(accepted).shape), (256, 32))
        with self.assertRaises(EmbeddingContractError) as raised:
            module(rejected)
        self.assertEqual(raised.exception.details["invariant"], "input.sequence_length")

    def test_28_empty_shapes_are_supported_exactly(self) -> None:
        module = self._module()
        cases = (
            ((0,), (0, 32)),
            ((3, 0), (3, 0, 32)),
            ((0, 3), (0, 3, 32)),
            ((0, 0), (0, 0, 32)),
        )
        for input_shape, output_shape in cases:
            with self.subTest(input_shape=input_shape):
                token_ids = torch.empty(input_shape, dtype=torch.long)
                self.assertEqual(tuple(module(token_ids).shape), output_shape)

    def test_29_output_is_exact_token_plus_generated_position_lookup(self) -> None:
        module = self._module()
        token_ids = torch.tensor([[3, 1, 3], [2, 1, 2]], dtype=torch.long)
        position_ids = torch.arange(3, dtype=torch.long, device="cpu")
        expected = (
            module.token_embeddings[token_ids]
            + module.position_embeddings[position_ids]
        )

        self.assertTrue(torch.equal(module(token_ids), expected))
        self.assertEqual(module(token_ids).dtype, torch.float32)
        self.assertEqual(module(token_ids).device, torch.device("cpu"))

    def test_30_output_is_new_and_autograd_connects_both_tables(self) -> None:
        module = self._module()
        token_ids = torch.tensor([0, 1], dtype=torch.long)
        before = token_ids.clone()
        output = module(token_ids)

        self.assertIsNotNone(output.grad_fn)
        self.assertNotEqual(output.data_ptr(), module.token_embeddings.data_ptr())
        self.assertNotEqual(output.data_ptr(), module.position_embeddings.data_ptr())
        self.assertTrue(torch.equal(token_ids, before))
        output.sum().backward()
        self.assertIsNotNone(module.token_embeddings.grad)
        self.assertIsNotNone(module.position_embeddings.grad)

    def test_31_repeated_token_rows_accumulate_exact_gradients(self) -> None:
        module = self._module()
        module(torch.tensor([3, 1, 3], dtype=torch.long)).sum().backward()
        token_gradient = module.token_embeddings.grad

        self.assertTrue(torch.equal(token_gradient[3], torch.full((32,), 2.0)))
        self.assertTrue(torch.equal(token_gradient[1], torch.ones(32)))
        self.assertTrue(torch.equal(token_gradient[0], torch.zeros(32)))

    def test_32_shared_position_rows_accumulate_across_batch(self) -> None:
        module = self._module()
        token_ids = torch.tensor([[3, 1, 3], [2, 1, 2]], dtype=torch.long)
        module(token_ids).sum().backward()
        position_gradient = module.position_embeddings.grad

        for position in range(3):
            self.assertTrue(
                torch.equal(position_gradient[position], torch.full((32,), 2.0))
            )
        self.assertTrue(torch.equal(position_gradient[3], torch.zeros(32)))

    def test_33_backward_reaches_both_tables_without_updating_values(self) -> None:
        module = self._module()
        tokens_before = module.token_embeddings.detach().clone()
        positions_before = module.position_embeddings.detach().clone()
        module(torch.tensor([0, 1], dtype=torch.long)).sum().backward()

        self.assertIsNotNone(module.token_embeddings.grad)
        self.assertIsNotNone(module.position_embeddings.grad)
        self.assertTrue(torch.equal(module.token_embeddings, tokens_before))
        self.assertTrue(torch.equal(module.position_embeddings, positions_before))

    def test_34_one_hot_matrix_multiplication_equals_direct_lookup(self) -> None:
        table = torch.tensor(
            [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]],
            dtype=torch.float32,
        )
        token_id = 1
        one_hot = torch.nn.functional.one_hot(
            torch.tensor(token_id),
            num_classes=3,
        ).to(torch.float32)

        self.assertTrue(torch.equal(one_hot @ table, table[token_id]))

    def test_35_errors_and_details_are_content_safe_and_immutable(self) -> None:
        module = self._module()
        secret_ids = torch.tensor([17, -777, 33], dtype=torch.long)

        with self.assertRaises(EmbeddingContractError) as range_error:
            module(secret_ids)
        with self.assertRaises(EmbeddingTypeError) as type_error:
            module("DO_NOT_RENDER")  # type: ignore[arg-type]

        rendered = (
            str(range_error.exception)
            + repr(range_error.exception)
            + str(type_error.exception)
            + repr(type_error.exception)
        )
        self.assertNotIn("-777", rendered)
        self.assertNotIn("17", rendered)
        self.assertNotIn("33", rendered)
        self.assertNotIn("DO_NOT_RENDER", rendered)
        self.assertEqual(range_error.exception.details["lower_bound"], 0)
        self.assertEqual(range_error.exception.details["upper_bound_exclusive"], 81)
        with self.assertRaises(TypeError):
            range_error.exception.details["new"] = "value"  # type: ignore[index]
