"""Tests for the corpus-neutral code-point tokenizer and membership layer."""

from __future__ import annotations

import sys
import unittest
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.tokenization.code_point import (  # noqa: E402
    CodePointTokenizer,
    InvalidTokenIdError,
    TokenizationTypeError,
    TokenizerStateError,
    UnknownCodePointError,
    build_code_point_vocabulary,
)


class FixedSequence(Sequence[int]):
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class ReverseIterSequence(FixedSequence):
    def __iter__(self):
        return reversed(self._values)


class StringSubclass(str):
    pass


class TensorLike:
    """Synthetic non-Sequence object; no PyTorch dependency is introduced."""


class CodePointTokenizerStateTests(unittest.TestCase):
    def test_valid_sorted_state_and_vocabulary_size(self) -> None:
        tokenizer = CodePointTokenizer((9, 10, 65, 0x0301, 0xD800, 0x1F642))

        self.assertEqual(tokenizer.vocabulary_size, 6)
        self.assertEqual(tokenizer.code_points, (9, 10, 65, 0x0301, 0xD800, 0x1F642))
        self.assertEqual(repr(tokenizer), "CodePointTokenizer(vocabulary_size=6)")

    def test_empty_state_is_rejected(self) -> None:
        with self.assertRaises(TokenizerStateError) as raised:
            CodePointTokenizer(())

        self.assertEqual(raised.exception.invariant, "code_points.nonempty")

    def test_non_tuple_state_is_rejected_without_coercion(self) -> None:
        with self.assertRaises(TokenizerStateError) as raised:
            CodePointTokenizer([65])  # type: ignore[arg-type]

        self.assertEqual(raised.exception.invariant, "code_points.exact_tuple")

    def test_non_int_and_bool_elements_are_rejected(self) -> None:
        for value in (65.0, "65", True):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TokenizerStateError) as raised:
                    CodePointTokenizer((value,))  # type: ignore[arg-type]
                self.assertEqual(
                    raised.exception.invariant,
                    "code_points.element.exact_int",
                )
                self.assertEqual(raised.exception.position, 0)

    def test_code_point_range_is_exact_and_includes_surrogates(self) -> None:
        for value in (-1, 0x110000):
            with self.subTest(value=value):
                with self.assertRaises(TokenizerStateError) as raised:
                    CodePointTokenizer((value,))
                self.assertEqual(
                    raised.exception.invariant,
                    "code_points.element.range",
                )

        tokenizer = CodePointTokenizer((0xD800,))
        self.assertEqual(tokenizer.code_points, (0xD800,))

    def test_duplicate_and_decreasing_state_are_distinct_failures(self) -> None:
        with self.assertRaises(TokenizerStateError) as duplicate:
            CodePointTokenizer((65, 65))
        with self.assertRaises(TokenizerStateError) as decreasing:
            CodePointTokenizer((66, 65))

        self.assertEqual(
            duplicate.exception.invariant,
            "code_points.element.duplicate",
        )
        self.assertEqual(
            decreasing.exception.invariant,
            "code_points.strictly_increasing",
        )
        self.assertEqual(duplicate.exception.position, 1)
        self.assertEqual(decreasing.exception.position, 1)


class CodePointTokenizerEncodeTests(unittest.TestCase):
    def test_ascii_punctuation_and_whitespace_encode_exactly(self) -> None:
        text = "A b!\t\n\nA"
        tokenizer = CodePointTokenizer(build_code_point_vocabulary((text,)))

        token_ids = tokenizer.encode(text)

        self.assertIsInstance(token_ids, tuple)
        self.assertEqual(len(token_ids), len(text))
        self.assertEqual(token_ids[0], token_ids[-1])
        self.assertEqual(tokenizer.decode(token_ids), text)

    def test_empty_and_repeated_code_points(self) -> None:
        tokenizer = CodePointTokenizer((65,))

        self.assertEqual(tokenizer.encode(""), ())
        self.assertEqual(tokenizer.encode("AAA"), (0, 0, 0))

    def test_non_ascii_combining_supplementary_and_surrogate_encode(self) -> None:
        text = "é" + "e\u0301" + "🙂" + chr(0xD800)
        tokenizer = CodePointTokenizer(build_code_point_vocabulary((text,)))

        token_ids = tokenizer.encode(text)

        self.assertEqual(len(token_ids), len(text))
        self.assertEqual(tokenizer.decode(token_ids), text)

    def test_string_subclass_is_accepted(self) -> None:
        tokenizer = CodePointTokenizer((65,))

        self.assertEqual(tokenizer.encode(StringSubclass("AA")), (0, 0))

    def test_non_string_is_rejected_without_coercion(self) -> None:
        class NoStringCoercion:
            called = False

            def __str__(self) -> str:
                self.called = True
                raise AssertionError("must not coerce")

        value = NoStringCoercion()
        tokenizer = CodePointTokenizer((65,))

        with self.assertRaises(TokenizationTypeError) as raised:
            tokenizer.encode(value)  # type: ignore[arg-type]

        self.assertFalse(value.called)
        self.assertEqual(raised.exception.invariant, "encode.text.isinstance_str")

    def test_first_unknown_is_reported_without_partial_output(self) -> None:
        tokenizer = CodePointTokenizer((65, 66))
        result = (999,)

        try:
            result = tokenizer.encode("AB🙂Z")
        except UnknownCodePointError as error:
            observed = error
        else:
            self.fail("unknown input did not raise")

        self.assertEqual(result, (999,))
        self.assertEqual(observed.code_point, 0x1F642)
        self.assertEqual(observed.u_plus, "U+1F642")
        self.assertEqual(observed.position, 2)

    def test_unknown_error_string_and_repr_are_content_safe(self) -> None:
        surrounding = "LEFT"
        literal = "🙂"
        tokenizer = CodePointTokenizer(
            build_code_point_vocabulary((surrounding + "RIGHT",))
        )

        with self.assertRaises(UnknownCodePointError) as raised:
            tokenizer.encode(surrounding + literal + "RIGHT")

        error = raised.exception
        rendered = str(error) + repr(error)
        self.assertNotIn(literal, rendered)
        self.assertNotIn(surrounding, rendered)
        self.assertNotIn("RIGHT", rendered)
        self.assertIn("128578", rendered)
        self.assertIn("U+1F642", rendered)
        self.assertIn("position 4", rendered)
        with self.assertRaises(AttributeError):
            error.code_point = 65  # type: ignore[misc]


class CodePointTokenizerDecodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CodePointTokenizer((65, 66, 67))

    def test_tuple_list_and_custom_sequence_are_accepted(self) -> None:
        self.assertEqual(self.tokenizer.decode((0, 1, 2)), "ABC")
        self.assertEqual(self.tokenizer.decode([2, 1, 0]), "CBA")
        self.assertEqual(self.tokenizer.decode(FixedSequence((0, 2))), "AC")
        self.assertEqual(
            self.tokenizer.decode(ReverseIterSequence((0, 1, 2))),
            "ABC",
        )

    def test_empty_sequence_and_repeated_ids(self) -> None:
        self.assertEqual(self.tokenizer.decode(()), "")
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(self.tokenizer.decode((1, 1, 1)), "BBB")

    def test_negative_equal_size_and_larger_ids_are_rejected(self) -> None:
        for value in (-1, self.tokenizer.vocabulary_size, 99):
            with self.subTest(value=value):
                with self.assertRaises(InvalidTokenIdError) as raised:
                    self.tokenizer.decode((0, value, 1))
                self.assertEqual(raised.exception.token_id, value)
                self.assertEqual(raised.exception.position, 1)
                self.assertEqual(raised.exception.lower_bound, 0)
                self.assertEqual(
                    raised.exception.upper_bound_exclusive,
                    self.tokenizer.vocabulary_size,
                )

    def test_non_exact_integer_elements_fail_at_first_position(self) -> None:
        for value in (True, 1.0, "1", object()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TokenizationTypeError) as raised:
                    self.tokenizer.decode((0, value, 2))  # type: ignore[arg-type]
                self.assertEqual(raised.exception.position, 1)
                self.assertEqual(
                    raised.exception.invariant,
                    "decode.token_id.exact_int_and_range",
                )

    def test_invalid_outer_containers_are_rejected(self) -> None:
        for value in ("012", b"012", bytearray(b"012"), TensorLike()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TokenizationTypeError) as raised:
                    self.tokenizer.decode(value)  # type: ignore[arg-type]
                self.assertEqual(
                    raised.exception.invariant,
                    "decode.token_ids.sequence",
                )

    def test_generator_is_rejected_without_consumption(self) -> None:
        generator = (value for value in (0, 1))

        with self.assertRaises(TokenizationTypeError):
            self.tokenizer.decode(generator)  # type: ignore[arg-type]

        self.assertEqual(next(generator), 0)

    def test_decode_returns_no_partial_result_and_reports_first_failure(self) -> None:
        result = "UNCHANGED"

        try:
            result = self.tokenizer.decode((0, 3, -1))
        except InvalidTokenIdError as error:
            observed = error
        else:
            self.fail("invalid ID did not raise")

        self.assertEqual(result, "UNCHANGED")
        self.assertEqual(observed.position, 1)
        self.assertEqual(observed.token_id, 3)

    def test_decode_does_not_mutate_input(self) -> None:
        token_ids = [0, 1, 2]
        before = tuple(token_ids)

        self.tokenizer.decode(token_ids)

        self.assertEqual(tuple(token_ids), before)

    def test_decode_error_representations_are_content_safe(self) -> None:
        secret_value = "DO_NOT_RENDER"

        with self.assertRaises(TokenizationTypeError) as wrong_type:
            self.tokenizer.decode((0, secret_value, 2))  # type: ignore[arg-type]
        with self.assertRaises(InvalidTokenIdError) as wrong_range:
            self.tokenizer.decode((0, 9, 2))

        self.assertNotIn(secret_value, str(wrong_type.exception))
        self.assertNotIn(secret_value, repr(wrong_type.exception))
        self.assertIn("position 1", str(wrong_type.exception))
        self.assertIn("token ID 9", str(wrong_range.exception))
        self.assertNotIn("ABC", str(wrong_range.exception))


class CodePointRoundTripTests(unittest.TestCase):
    def test_representative_supported_text_round_trips_exactly(self) -> None:
        samples = (
            "",
            "ASCII punctuation: yes!",
            "spaces\tand\n\nblank lines\n",
            "é and e\u0301",
            "supplementary 🙂",
            chr(0xD800),
        )
        tokenizer = CodePointTokenizer(build_code_point_vocabulary(samples))

        for sample in samples:
            with self.subTest(code_point_count=len(sample)):
                self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)


class CodePointMembershipTests(unittest.TestCase):
    def test_membership_is_union_in_numeric_order(self) -> None:
        vocabulary = build_code_point_vocabulary(("ba", "ac", "bb"))

        self.assertEqual(vocabulary, (ord("a"), ord("b"), ord("c")))

    def test_document_order_and_duplicate_membership_do_not_change_result(self) -> None:
        first = build_code_point_vocabulary(("ab", "bc", "a"))
        second = build_code_point_vocabulary(("a", "bc", "ab"))

        self.assertEqual(first, second)
        self.assertEqual(first, build_code_point_vocabulary(("abc",)))

    def test_empty_inputs_are_allowed_at_neutral_layer(self) -> None:
        self.assertEqual(build_code_point_vocabulary(()), ())
        self.assertEqual(build_code_point_vocabulary(("", "")), ())

    def test_membership_rejects_outer_string_and_non_string_document(self) -> None:
        with self.assertRaises(TokenizationTypeError) as outer:
            build_code_point_vocabulary("abc")
        with self.assertRaises(TokenizationTypeError) as element:
            build_code_point_vocabulary(("abc", 3))  # type: ignore[arg-type]

        self.assertEqual(outer.exception.invariant, "membership.documents.sequence")
        self.assertEqual(
            element.exception.invariant,
            "membership.document.isinstance_str",
        )
        self.assertEqual(element.exception.position, 1)

    def test_repeated_construction_is_deterministic(self) -> None:
        documents = ("cab", "é", "e\u0301", "🙂")
        expected = build_code_point_vocabulary(documents)

        for _ in range(5):
            self.assertEqual(build_code_point_vocabulary(documents), expected)
