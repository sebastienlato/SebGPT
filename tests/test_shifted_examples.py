"""Independent contract tests for document-local shifted examples."""

from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.data.shifted_examples import (  # noqa: E402
    Phase4ContractError,
    Phase4TypeError,
    ShiftedTokenExample,
    iter_shifted_examples,
)


TEST_CONTEXT_LENGTH = 64
TEST_STRIDE = 64
TEST_VOCABULARY_SIZE = 81
BOUNDARY_EXPECTATIONS = {
    0: ((), ()),
    1: ((), ()),
    2: ((0,), (1,)),
    63: ((0,), (62,)),
    64: ((0,), (63,)),
    65: ((0,), (64,)),
    66: ((0, 64), (64, 1)),
    128: ((0, 64), (64, 63)),
    129: ((0, 64), (64, 64)),
    130: ((0, 64, 128), (64, 64, 1)),
}


def _ids(length: int) -> tuple[int, ...]:
    return tuple(position % TEST_VOCABULARY_SIZE for position in range(length))


def _examples(token_ids: tuple[int, ...]):
    return tuple(
        iter_shifted_examples(
            token_ids,
            context_length=TEST_CONTEXT_LENGTH,
            stride=TEST_STRIDE,
        )
    )


class ShiftedExampleConstructionTests(unittest.TestCase):
    def test_public_function_is_not_a_generator_and_validates_at_call_time(
        self,
    ) -> None:
        self.assertFalse(inspect.isgeneratorfunction(iter_shifted_examples))

        invalid = _ids(130)[:-1] + (TEST_VOCABULARY_SIZE,)
        with self.assertRaises(Phase4ContractError) as raised:
            iter_shifted_examples(
                invalid,
                context_length=TEST_CONTEXT_LENGTH,
                stride=TEST_STRIDE,
            )

        self.assertEqual(raised.exception.details["position"], 129)

    def test_examples_are_constructed_only_when_each_is_requested(self) -> None:
        token_ids = _ids(130)
        with patch(
            "sebgpt.data.shifted_examples.ShiftedTokenExample",
            wraps=ShiftedTokenExample,
        ) as constructor:
            iterator = iter_shifted_examples(
                token_ids,
                context_length=TEST_CONTEXT_LENGTH,
                stride=TEST_STRIDE,
            )
            self.assertEqual(constructor.call_count, 0)

            first = next(iterator)
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(first.start_index, 0)
            self.assertEqual(first.input_ids, token_ids[0:64])
            self.assertEqual(first.target_ids, token_ids[1:65])

            second = next(iterator)
            self.assertEqual(constructor.call_count, 2)
            self.assertEqual(second.start_index, 64)
            self.assertEqual(second.input_ids, token_ids[64:128])
            self.assertEqual(second.target_ids, token_ids[65:129])

            self.assertEqual(constructor.call_count, 2)
            third = next(iterator)
            self.assertEqual(constructor.call_count, 3)
            self.assertEqual(third.start_index, 128)
            self.assertEqual(third.input_ids, token_ids[128:129])
            self.assertEqual(third.target_ids, token_ids[129:130])

            with self.assertRaises(StopIteration):
                next(iterator)
            self.assertEqual(constructor.call_count, 3)

    def test_empty_and_one_token_documents_produce_no_examples(self) -> None:
        self.assertEqual(_examples(()), ())
        self.assertEqual(_examples((3,)), ())

    def test_all_required_boundary_lengths_are_exact(self) -> None:
        for length, (expected_starts, expected_lengths) in (
            BOUNDARY_EXPECTATIONS.items()
        ):
            with self.subTest(length=length):
                examples = _examples(_ids(length))
                self.assertEqual(
                    tuple(example.start_index for example in examples),
                    expected_starts,
                )
                self.assertEqual(
                    tuple(example.length for example in examples),
                    expected_lengths,
                )

    def test_inputs_and_targets_are_exact_shifted_slices(self) -> None:
        token_ids = _ids(130)
        for example in _examples(token_ids):
            start = example.start_index
            length = example.length
            self.assertEqual(example.input_ids, token_ids[start : start + length])
            self.assertEqual(
                example.target_ids,
                token_ids[start + 1 : start + length + 1],
            )
            self.assertEqual(len(example.input_ids), len(example.target_ids))
            self.assertGreaterEqual(length, 1)
            self.assertLessEqual(length, TEST_CONTEXT_LENGTH)

    def test_transition_indices_form_an_exact_partition(self) -> None:
        for length in (2, 3, 64, 65, 66, 130, 257, 514):
            with self.subTest(length=length):
                covered = tuple(
                    example.start_index + offset
                    for example in _examples(_ids(length))
                    for offset in range(example.length)
                )
                self.assertEqual(covered, tuple(range(length - 1)))
                self.assertEqual(len(covered), len(set(covered)))

    def test_boundary_source_token_does_not_duplicate_a_target(self) -> None:
        token_ids = _ids(66)
        first, tail = _examples(token_ids)

        self.assertEqual(first.target_ids[-1], token_ids[64])
        self.assertEqual(tail.input_ids[0], token_ids[64])
        self.assertEqual(tail.target_ids, (token_ids[65],))
        self.assertEqual(
            tuple(
                example.start_index + offset
                for example in (first, tail)
                for offset in range(example.length)
            ),
            tuple(range(65)),
        )

    def test_two_documents_are_never_joined(self) -> None:
        first_document = (1, 2, 3)
        second_document = (7, 8, 9)
        first = _examples(first_document)
        second = _examples(second_document)

        observed_pairs = tuple(
            zip(example.input_ids, example.target_ids, strict=True)
            for example in first + second
        )
        flattened = tuple(pair for group in observed_pairs for pair in group)
        self.assertEqual(flattened, ((1, 2), (2, 3), (7, 8), (8, 9)))
        self.assertNotIn((3, 7), flattened)

    def test_construction_is_deterministic(self) -> None:
        token_ids = _ids(257)
        self.assertEqual(_examples(token_ids), _examples(token_ids))

    def test_examples_are_frozen_and_hide_id_sequences(self) -> None:
        example = _examples((3, 4, 5))[0]

        self.assertIsInstance(example, ShiftedTokenExample)
        self.assertIs(type(example.input_ids), tuple)
        self.assertIs(type(example.target_ids), tuple)
        self.assertNotIn(str(example.input_ids), repr(example))
        self.assertNotIn(str(example.target_ids), repr(example))
        self.assertIn("length=2", repr(example))
        with self.assertRaises(FrozenInstanceError):
            example.start_index = 1  # type: ignore[misc]


class ShiftedExampleValidationTests(unittest.TestCase):
    def test_outer_container_must_be_an_exact_tuple(self) -> None:
        class TupleSubclass(tuple):
            pass

        for value in ([1, 2], TupleSubclass((1, 2)), "12"):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(Phase4TypeError) as raised:
                    iter_shifted_examples(  # type: ignore[arg-type]
                        value,
                        context_length=TEST_CONTEXT_LENGTH,
                        stride=TEST_STRIDE,
                    )
                self.assertEqual(
                    raised.exception.details["invariant"],
                    "example.token_ids.exact_tuple",
                )

    def test_context_and_stride_type_then_value_order_is_exact(self) -> None:
        cases = (
            (True, TEST_STRIDE, Phase4TypeError, "example.context_length.exact_int"),
            (63, False, Phase4ContractError, "example.context_length.value"),
            (TEST_CONTEXT_LENGTH, True, Phase4TypeError, "example.stride.exact_int"),
            (TEST_CONTEXT_LENGTH, 1, Phase4ContractError, "example.stride.value"),
        )
        for context_length, stride, error_type, invariant in cases:
            with self.subTest(invariant=invariant):
                with self.assertRaises(error_type) as raised:
                    iter_shifted_examples(
                        (1, 2),
                        context_length=context_length,  # type: ignore[arg-type]
                        stride=stride,  # type: ignore[arg-type]
                    )
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_token_bool_and_non_integer_fail_at_first_position(self) -> None:
        for token_id in (True, 1.0, "1"):
            with self.subTest(token_type=type(token_id).__name__):
                with self.assertRaises(Phase4TypeError) as raised:
                    iter_shifted_examples(
                        (1, token_id, 81),  # type: ignore[arg-type]
                        context_length=TEST_CONTEXT_LENGTH,
                        stride=TEST_STRIDE,
                    )
                self.assertEqual(raised.exception.details["position"], 1)

    def test_token_range_failure_is_first_and_content_safe(self) -> None:
        for token_id in (-1, TEST_VOCABULARY_SIZE, 1000):
            with self.subTest(token_id=token_id):
                with self.assertRaises(Phase4ContractError) as raised:
                    iter_shifted_examples(
                        (1, token_id, 2),
                        context_length=TEST_CONTEXT_LENGTH,
                        stride=TEST_STRIDE,
                    )
                error = raised.exception
                self.assertEqual(error.details["position"], 1)
                self.assertNotIn(str((1, token_id, 2)), str(error))
                self.assertEqual(repr(error), str(error))
                with self.assertRaises(TypeError):
                    error.details["invariant"] = "changed"  # type: ignore[index]

    def test_multi_fault_input_has_one_deterministic_first_failure(self) -> None:
        with self.assertRaises(Phase4TypeError) as raised:
            iter_shifted_examples(  # type: ignore[arg-type]
                [81, True],
                context_length=63,
                stride=True,
            )
        self.assertEqual(
            raised.exception.details["invariant"],
            "example.token_ids.exact_tuple",
        )


if __name__ == "__main__":
    unittest.main()
