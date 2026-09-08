"""Synthetic tests for Shakespeare tokenization orchestration governance."""

from __future__ import annotations

import hashlib
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.tokenization.shakespeare import (  # noqa: E402
    EXPECTED_NORMALIZATION as PRODUCTION_EXPECTED_NORMALIZATION,
    EXPECTED_TRAINING_WORKS as PRODUCTION_EXPECTED_TRAINING_WORKS,
    ShakespeareTokenizationError,
    build_shakespeare_training_vocabulary,
)


TEST_EXPECTED_TRAINING_WORKS = (
    (1, "hamlet"),
    (2, "romeo-and-juliet"),
    (3, "macbeth"),
    (4, "a-midsummer-nights-dream"),
    (5, "much-ado-about-nothing"),
    (6, "henry-v"),
)
TEST_EXPECTED_NORMALIZATION = "crlf_to_lf_only"


@dataclass(frozen=True)
class SyntheticIdentity:
    sha256: str
    byte_count: int


class GuardedWork:
    def __init__(
        self,
        *,
        work_id: str,
        manifest_order: int,
        split: str,
        normalization: str,
        sha256: str,
        byte_count: int,
        code_point_count: int,
        text: str,
        allow_text_access: bool = True,
    ) -> None:
        self.work_id = work_id
        self.manifest_order = manifest_order
        self.split = split
        self.normalization = normalization
        self.processed = SyntheticIdentity(sha256, byte_count)
        self.processed_code_point_count = code_point_count
        self._text = text
        self._allow_text_access = allow_text_access
        self.text_access_count = 0

    @property
    def processed_text(self) -> str:
        self.text_access_count += 1
        if not self._allow_text_access:
            raise AssertionError("processed_text must not be accessed")
        return self._text


def _identity(text: str) -> tuple[str, int, int]:
    data = text.encode("utf-8", errors="strict")
    return hashlib.sha256(data).hexdigest(), len(data), len(text)


def _manifest(texts: tuple[str, ...]) -> dict[str, object]:
    works = [
        {"order": order, "work_id": work_id, "split": "train"}
        for order, work_id in TEST_EXPECTED_TRAINING_WORKS
    ]
    works.extend(
        (
            {"order": 7, "work_id": "validation-fixture", "split": "validation"},
            {"order": 8, "work_id": "test-fixture", "split": "test"},
        )
    )
    results = []
    for (order, work_id), text in zip(
        TEST_EXPECTED_TRAINING_WORKS,
        texts,
        strict=True,
    ):
        sha256, byte_count, code_point_count = _identity(text)
        results.append(
            {
                "work_id": work_id,
                "manifest_order": order,
                "split": "train",
                "normalization": TEST_EXPECTED_NORMALIZATION,
                "processed": {
                    "sha256": sha256,
                    "byte_count": byte_count,
                    "code_point_count": code_point_count,
                },
            }
        )
    return {
        "schema_version": 4,
        "dataset_id": "shakespeare-eight-play",
        "works": works,
        "processing": {"per_work_results": results},
    }


def _works(
    texts: tuple[str, ...],
    *,
    allow_text_access: bool = True,
) -> tuple[GuardedWork, ...]:
    result = []
    for (order, work_id), text in zip(
        TEST_EXPECTED_TRAINING_WORKS,
        texts,
        strict=True,
    ):
        sha256, byte_count, code_point_count = _identity(text)
        result.append(
            GuardedWork(
                work_id=work_id,
                manifest_order=order,
                split="train",
                normalization=TEST_EXPECTED_NORMALIZATION,
                sha256=sha256,
                byte_count=byte_count,
                code_point_count=code_point_count,
                text=text,
                allow_text_access=allow_text_access,
            )
        )
    return tuple(result)


def _replace_work(work: GuardedWork, **changes) -> GuardedWork:
    values = {
        "work_id": work.work_id,
        "manifest_order": work.manifest_order,
        "split": work.split,
        "normalization": work.normalization,
        "sha256": work.processed.sha256,
        "byte_count": work.processed.byte_count,
        "code_point_count": work.processed_code_point_count,
        "text": work._text,
        "allow_text_access": work._allow_text_access,
    }
    values.update(changes)
    return GuardedWork(**values)


class ShakespeareVocabularyOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.texts = ("A", "B", "C", "é", "e\u0301", "🙂")
        self.manifest = _manifest(self.texts)

    def test_permitted_synthetic_training_works_are_verified_before_membership(self) -> None:
        self.assertEqual(
            PRODUCTION_EXPECTED_TRAINING_WORKS,
            TEST_EXPECTED_TRAINING_WORKS,
        )
        self.assertEqual(
            PRODUCTION_EXPECTED_NORMALIZATION,
            TEST_EXPECTED_NORMALIZATION,
        )
        works = _works(self.texts)

        vocabulary = build_shakespeare_training_vocabulary(works, self.manifest)

        expected = tuple(sorted({ord(character) for text in self.texts for character in text}))
        self.assertEqual(vocabulary, expected)
        self.assertEqual(tuple(work.text_access_count for work in works), (1,) * 6)

    def test_validation_and_test_are_rejected_before_text_access(self) -> None:
        for split in ("validation", "test"):
            with self.subTest(split=split):
                works = list(_works(self.texts, allow_text_access=False))
                works[0] = _replace_work(works[0], split=split)

                with self.assertRaises(ShakespeareTokenizationError) as raised:
                    build_shakespeare_training_vocabulary(tuple(works), self.manifest)

                self.assertEqual(raised.exception.details["invariant"], "work.split.train")
                self.assertEqual(works[0].text_access_count, 0)

    def test_metadata_failures_precede_all_text_access(self) -> None:
        changes_and_invariants = (
            ({"work_id": "wrong"}, "work.id"),
            ({"manifest_order": 2}, "work.order"),
            ({"manifest_order": True}, "work.order"),
            ({"normalization": "wrong"}, "work.normalization"),
            ({"sha256": "0" * 64}, "work.processed.sha256"),
            ({"byte_count": 999}, "work.processed.byte_count"),
            ({"byte_count": True}, "work.processed.byte_count"),
            ({"code_point_count": 999}, "work.processed_code_point_count"),
            ({"code_point_count": True}, "work.processed_code_point_count"),
        )
        for changes, invariant in changes_and_invariants:
            with self.subTest(invariant=invariant):
                works = list(_works(self.texts, allow_text_access=False))
                works[0] = _replace_work(works[0], **changes)

                with self.assertRaises(ShakespeareTokenizationError) as raised:
                    build_shakespeare_training_vocabulary(tuple(works), self.manifest)

                self.assertEqual(raised.exception.details["invariant"], invariant)
                self.assertEqual(sum(work.text_access_count for work in works), 0)

    def test_reordered_works_are_rejected_before_text_access(self) -> None:
        works = list(_works(self.texts, allow_text_access=False))
        works[0], works[1] = works[1], works[0]

        with self.assertRaises(ShakespeareTokenizationError) as raised:
            build_shakespeare_training_vocabulary(tuple(works), self.manifest)

        self.assertEqual(raised.exception.details["invariant"], "work.id")
        self.assertEqual(sum(work.text_access_count for work in works), 0)

    def test_manifest_identity_failure_precedes_text_access(self) -> None:
        works = _works(self.texts, allow_text_access=False)
        changed_manifest = dict(self.manifest)
        changed_manifest["dataset_id"] = "wrong"

        with self.assertRaises(ShakespeareTokenizationError) as raised:
            build_shakespeare_training_vocabulary(works, changed_manifest)

        self.assertEqual(raised.exception.details["invariant"], "manifest.dataset_id")
        self.assertEqual(sum(work.text_access_count for work in works), 0)

    def test_recomputed_sha256_mismatch_is_rejected_safely(self) -> None:
        works = list(_works(self.texts))
        works[0] = _replace_work(works[0], text="Z")

        with self.assertRaises(ShakespeareTokenizationError) as raised:
            build_shakespeare_training_vocabulary(tuple(works), self.manifest)

        self.assertEqual(raised.exception.details["invariant"], "work.recomputed.sha256")
        self.assertEqual(works[0].text_access_count, 1)
        self.assertEqual(sum(work.text_access_count for work in works[1:]), 0)

    def test_recomputed_byte_and_code_point_counts_are_checked(self) -> None:
        byte_manifest = _manifest(self.texts)
        byte_works = list(_works(self.texts))
        byte_works[0] = _replace_work(byte_works[0], byte_count=2)
        byte_manifest["processing"]["per_work_results"][0]["processed"][  # type: ignore[index]
            "byte_count"
        ] = 2

        with self.assertRaises(ShakespeareTokenizationError) as byte_error:
            build_shakespeare_training_vocabulary(tuple(byte_works), byte_manifest)

        code_point_texts = ("é",) + self.texts[1:]
        code_point_manifest = _manifest(code_point_texts)
        code_point_works = list(_works(code_point_texts))
        code_point_works[0] = _replace_work(code_point_works[0], code_point_count=2)
        code_point_manifest["processing"]["per_work_results"][0][  # type: ignore[index]
            "processed"
        ]["code_point_count"] = 2

        with self.assertRaises(ShakespeareTokenizationError) as code_point_error:
            build_shakespeare_training_vocabulary(
                tuple(code_point_works),
                code_point_manifest,
            )

        self.assertEqual(
            byte_error.exception.details["invariant"],
            "work.recomputed.byte_count",
        )
        self.assertEqual(
            code_point_error.exception.details["invariant"],
            "work.recomputed.code_point_count",
        )

    def test_strict_utf8_recomputation_rejects_surrogate_safely(self) -> None:
        works = list(_works(self.texts))
        works[0] = _replace_work(works[0], text=chr(0xD800))

        with self.assertRaises(ShakespeareTokenizationError) as raised:
            build_shakespeare_training_vocabulary(tuple(works), self.manifest)

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.processed_text.strict_utf8",
        )

    def test_orchestration_error_does_not_render_guarded_text(self) -> None:
        secret = "SYNTHETIC SECRET MUST NOT RENDER"
        works = list(_works(self.texts, allow_text_access=False))
        works[0] = _replace_work(works[0], work_id="wrong", text=secret)

        with self.assertRaises(ShakespeareTokenizationError) as raised:
            build_shakespeare_training_vocabulary(tuple(works), self.manifest)

        rendered = str(raised.exception) + repr(raised.exception)
        self.assertNotIn(secret, rendered)
        self.assertEqual(works[0].text_access_count, 0)
        with self.assertRaises(TypeError):
            raised.exception.details["invariant"] = "changed"  # type: ignore[index]


class StatisticsAccessBoundaryTests(unittest.TestCase):
    def test_statistics_implementation_is_deferred(self) -> None:
        import sebgpt.tokenization.shakespeare as module

        public_statistics_functions = tuple(
            name
            for name in dir(module)
            if not name.startswith("_") and "statistic" in name.lower()
        )

        self.assertEqual(public_statistics_functions, ())
