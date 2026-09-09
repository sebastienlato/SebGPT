"""Independent tests for Shakespeare Phase 4 example governance."""

from __future__ import annotations

import hashlib
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.data.shakespeare_examples import (  # noqa: E402
    DocumentShiftedExample,
    iter_shakespeare_phase_4_examples,
)
from sebgpt.data.shakespeare_extract import ByteStage, ExtractedWork  # noqa: E402
from sebgpt.data.shifted_examples import (  # noqa: E402
    Phase4ContractError,
    Phase4GovernanceError,
    Phase4TypeError,
)
from sebgpt.tokenization.code_point import CodePointTokenizer  # noqa: E402
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    VocabularyBinding,
    load_accepted_vocabulary_binding,
)


TEST_SCHEMA_VERSION = 4
TEST_DATASET_ID = "shakespeare-eight-play"
TEST_NORMALIZATION = "crlf_to_lf_only"
TRAINING_IDENTITIES = (
    (1, "hamlet"),
    (2, "romeo-and-juliet"),
    (3, "macbeth"),
    (4, "a-midsummer-nights-dream"),
    (5, "much-ado-about-nothing"),
    (6, "henry-v"),
)
VALIDATION_IDENTITY = (7, "the-tempest")
TEST_IDENTITY = (8, "twelfth-night")
TRAINING_TEXTS = ("A\n", "B\n", "C\n", "D\n", "E\n", "F\n")
VALIDATION_TEXT = "G\n"
SEALED_SENTINEL_TEXT = "SEALED SENTINEL"


def _stage(text: str) -> ByteStage:
    content = text.encode("utf-8")
    return ByteStage(hashlib.sha256(content).hexdigest(), len(content))


def _make_work(
    order: int,
    work_id: str,
    split: str,
    text: str,
    *,
    guarded: bool = False,
    counting: bool = False,
    normalization: str = TEST_NORMALIZATION,
    processed_sha256: str | None = None,
    processed_byte_count: int | None = None,
    processed_code_point_count: int | None = None,
) -> ExtractedWork:
    if guarded:
        work_type = GuardedExtractedWork
    elif counting:
        work_type = CountingExtractedWork
    else:
        work_type = ExtractedWork
    stage = _stage(text)
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
        processed=ByteStage(
            processed_sha256
            if processed_sha256 is not None
            else stage.sha256,
            processed_byte_count
            if processed_byte_count is not None
            else stage.byte_count,
        ),
        removed_local_contents_byte_count=0,
        processed_code_point_count=(
            processed_code_point_count
            if processed_code_point_count is not None
            else len(text)
        ),
        processed_line_count=text.count("\n"),
        processed_word_count=len(text.split()),
        normalization=normalization,
    )
    if guarded or counting:
        object.__setattr__(work, "_text_access_count", 0)
    return work


class GuardedExtractedWork(ExtractedWork):
    """An accepted-boundary work that fails if its text is accessed."""

    def __getattribute__(self, name: str) -> Any:
        if name == "processed_text":
            count = object.__getattribute__(self, "_text_access_count")
            object.__setattr__(self, "_text_access_count", count + 1)
            raise AssertionError("sealed-test content access is forbidden")
        return super().__getattribute__(name)

    @property
    def text_access_count(self) -> int:
        return object.__getattribute__(self, "_text_access_count")


class CountingExtractedWork(ExtractedWork):
    """An accepted-boundary work that counts permitted text access."""

    def __getattribute__(self, name: str) -> Any:
        if name == "processed_text":
            count = object.__getattribute__(self, "_text_access_count")
            object.__setattr__(self, "_text_access_count", count + 1)
        return super().__getattribute__(name)

    @property
    def text_access_count(self) -> int:
        return object.__getattribute__(self, "_text_access_count")


def _manifest() -> dict[str, Any]:
    identities = (
        *((order, work_id, "train", text) for (order, work_id), text in zip(
            TRAINING_IDENTITIES,
            TRAINING_TEXTS,
            strict=True,
        )),
        (*VALIDATION_IDENTITY, "validation", VALIDATION_TEXT),
        (*TEST_IDENTITY, "test", SEALED_SENTINEL_TEXT),
    )
    return {
        "schema_version": TEST_SCHEMA_VERSION,
        "dataset_id": TEST_DATASET_ID,
        "works": [
            {"order": order, "work_id": work_id, "split": split}
            for order, work_id, split, _ in identities
        ],
        "processing": {
            "per_work_results": [
                {
                    "work_id": work_id,
                    "manifest_order": order,
                    "split": split,
                    "normalization": TEST_NORMALIZATION,
                    "processed": {
                        "sha256": _stage(text).sha256,
                        "byte_count": _stage(text).byte_count,
                        "code_point_count": len(text),
                    },
                }
                for order, work_id, split, text in identities
            ]
        },
    }


def _training_works(*, guarded: bool = False) -> tuple[ExtractedWork, ...]:
    return tuple(
        _make_work(order, work_id, "train", text, guarded=guarded)
        for (order, work_id), text in zip(
            TRAINING_IDENTITIES,
            TRAINING_TEXTS,
            strict=True,
        )
    )


def _validation_work(*, guarded: bool = False) -> ExtractedWork:
    return _make_work(
        *VALIDATION_IDENTITY,
        "validation",
        VALIDATION_TEXT,
        guarded=guarded,
    )


def _sealed_work() -> GuardedExtractedWork:
    work = _make_work(
        *TEST_IDENTITY,
        "test",
        SEALED_SENTINEL_TEXT,
        guarded=True,
    )
    assert isinstance(work, GuardedExtractedWork)
    return work


class ShakespeareExampleGovernanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
        cls.tokenizer = CodePointTokenizer(cls.vocabulary.code_points)
        cls.manifest = _manifest()

    def assert_guarded_works_unread(
        self,
        works: tuple[ExtractedWork, ...] | list[ExtractedWork],
    ) -> None:
        self.assertTrue(
            all(
                isinstance(work, GuardedExtractedWork)
                and work.text_access_count == 0
                for work in works
            )
        )

    def test_six_training_works_are_accepted_in_manifest_order(self) -> None:
        examples = tuple(
            iter_shakespeare_phase_4_examples(
                _training_works(),
                self.vocabulary,
                self.manifest,
                split="train",
            )
        )

        self.assertEqual(len(examples), 6)
        self.assertEqual(
            tuple((item.manifest_order, item.work_id) for item in examples),
            TRAINING_IDENTITIES,
        )
        self.assertTrue(all(item.split == "train" for item in examples))

    def test_validation_accepts_only_the_tempest_through_validation(self) -> None:
        examples = tuple(
            iter_shakespeare_phase_4_examples(
                (_validation_work(),),
                self.vocabulary,
                self.manifest,
                split="validation",
            )
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(
            (examples[0].manifest_order, examples[0].work_id),
            VALIDATION_IDENTITY,
        )
        self.assertEqual(examples[0].split, "validation")

    def test_documents_are_encoded_and_windowed_independently(self) -> None:
        examples = tuple(
            iter_shakespeare_phase_4_examples(
                _training_works(),
                self.vocabulary,
                self.manifest,
                split="train",
            )
        )
        expected_pairs = tuple(
            tuple(zip(ids[:-1], ids[1:], strict=True))
            for ids in (self.tokenizer.encode(text) for text in TRAINING_TEXTS)
        )
        observed_pairs = tuple(
            tuple(
                zip(
                    item.example.input_ids,
                    item.example.target_ids,
                    strict=True,
                )
            )
            for item in examples
        )

        self.assertEqual(observed_pairs, expected_pairs)
        self.assertTrue(all(item.example.start_index == 0 for item in examples))

    def test_traversal_is_deterministic(self) -> None:
        def run() -> tuple[DocumentShiftedExample, ...]:
            return tuple(
                iter_shakespeare_phase_4_examples(
                    _training_works(),
                    self.vocabulary,
                    self.manifest,
                    split="train",
                )
            )

        self.assertEqual(run(), run())

    def test_valid_text_access_is_on_demand_and_one_document_at_a_time(self) -> None:
        works = tuple(
            _make_work(
                order,
                work_id,
                "train",
                text,
                counting=True,
            )
            for (order, work_id), text in zip(
                TRAINING_IDENTITIES,
                TRAINING_TEXTS,
                strict=True,
            )
        )
        self.assertTrue(all(isinstance(work, CountingExtractedWork) for work in works))

        iterator = iter_shakespeare_phase_4_examples(
            works,
            self.vocabulary,
            self.manifest,
            split="train",
        )
        self.assertTrue(
            all(
                isinstance(work, CountingExtractedWork)
                and work.text_access_count == 0
                for work in works
            )
        )

        first = next(iterator)
        self.assertEqual(first.work_id, TRAINING_IDENTITIES[0][1])
        self.assertEqual(
            tuple(
                work.text_access_count
                if isinstance(work, CountingExtractedWork)
                else -1
                for work in works
            ),
            (1, 0, 0, 0, 0, 0),
        )

    def test_document_example_is_frozen_and_repr_hides_ids(self) -> None:
        item = next(
            iter_shakespeare_phase_4_examples(
                (_validation_work(),),
                self.vocabulary,
                self.manifest,
                split="validation",
            )
        )

        self.assertNotIn(str(item.example.input_ids), repr(item))
        self.assertNotIn(str(item.example.target_ids), repr(item))
        with self.assertRaises(FrozenInstanceError):
            item.split = "train"  # type: ignore[misc]

    def test_sealed_split_is_refused_before_sentinel_content_access(self) -> None:
        sealed = _sealed_work()

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                (sealed,),
                self.vocabulary,
                self.manifest,
                split="test",
            )

        self.assertEqual(raised.exception.details["invariant"], "split.permitted")
        self.assertEqual(sealed.text_access_count, 0)

    def test_all_eight_then_filter_is_refused_before_any_content_access(self) -> None:
        guarded = _training_works(guarded=True) + (
            _validation_work(guarded=True),
            _sealed_work(),
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                guarded,
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "works.count")
        self.assert_guarded_works_unread(guarded)

    def test_missing_training_work_is_refused_before_content_access(self) -> None:
        works = _training_works(guarded=True)[:-1]

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                works,
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "works.count")
        self.assert_guarded_works_unread(works)

    def test_duplicate_training_work_is_refused_before_content_access(self) -> None:
        works = list(_training_works(guarded=True))
        works[1] = _make_work(
            *TRAINING_IDENTITIES[0],
            "train",
            TRAINING_TEXTS[0],
            guarded=True,
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "work.id")
        self.assertEqual(raised.exception.details["position"], 1)
        self.assert_guarded_works_unread(works)

    def test_reordered_training_works_are_refused_before_content_access(self) -> None:
        works = list(_training_works(guarded=True))
        works[0], works[1] = works[1], works[0]

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "work.id")
        self.assertEqual(raised.exception.details["position"], 0)
        self.assert_guarded_works_unread(works)

    def test_count_correct_sealed_replacement_is_refused_before_content_access(
        self,
    ) -> None:
        works = list(_training_works(guarded=True))
        works[-1] = _sealed_work()

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "work.id")
        self.assertEqual(raised.exception.details["position"], 5)
        self.assert_guarded_works_unread(works)

    def test_wrong_stored_normalization_is_refused_before_content_access(self) -> None:
        works = list(_training_works(guarded=True))
        works[0] = _make_work(
            *TRAINING_IDENTITIES[0],
            "train",
            TRAINING_TEXTS[0],
            guarded=True,
            normalization="wrong",
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.normalization",
        )
        self.assertEqual(raised.exception.details["position"], 0)
        self.assert_guarded_works_unread(works)

    def test_wrong_stored_sha256_is_refused_before_content_access(self) -> None:
        works = list(_training_works(guarded=True))
        works[0] = _make_work(
            *TRAINING_IDENTITIES[0],
            "train",
            TRAINING_TEXTS[0],
            guarded=True,
            processed_sha256="0" * 64,
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.processed.sha256",
        )
        self.assertEqual(raised.exception.details["position"], 0)
        self.assert_guarded_works_unread(works)

    def test_wrong_stored_byte_count_is_refused_before_content_access(self) -> None:
        works = list(_training_works(guarded=True))
        works[0] = _make_work(
            *TRAINING_IDENTITIES[0],
            "train",
            TRAINING_TEXTS[0],
            guarded=True,
            processed_byte_count=len(TRAINING_TEXTS[0].encode("utf-8")) + 1,
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.processed.byte_count",
        )
        self.assertEqual(raised.exception.details["position"], 0)
        self.assert_guarded_works_unread(works)

    def test_wrong_stored_code_point_count_is_refused_before_content_access(
        self,
    ) -> None:
        works = list(_training_works(guarded=True))
        works[0] = _make_work(
            *TRAINING_IDENTITIES[0],
            "train",
            TRAINING_TEXTS[0],
            guarded=True,
            processed_code_point_count=len(TRAINING_TEXTS[0]) + 1,
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "work.processed.code_point_count",
        )
        self.assertEqual(raised.exception.details["position"], 0)
        self.assert_guarded_works_unread(works)

    def test_all_metadata_is_checked_before_any_permitted_text_access(self) -> None:
        works = list(_training_works(guarded=True))
        works[1] = _make_work(
            2,
            "unexpected",
            "train",
            TRAINING_TEXTS[1],
            guarded=True,
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "work.id")
        self.assertEqual(raised.exception.details["position"], 1)
        self.assertTrue(
            all(
                isinstance(work, GuardedExtractedWork)
                and work.text_access_count == 0
                for work in works
            )
        )

    def test_cross_split_work_is_refused_before_content_access(self) -> None:
        works = list(_training_works(guarded=True))
        works[0] = _make_work(
            1,
            "hamlet",
            "validation",
            TRAINING_TEXTS[0],
            guarded=True,
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                tuple(works),
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "work.split")
        self.assertTrue(
            all(
                isinstance(work, GuardedExtractedWork)
                and work.text_access_count == 0
                for work in works
            )
        )

    def test_recomputed_provenance_failure_is_safe_and_on_demand(self) -> None:
        works = list(_training_works())
        works[0] = replace(works[0], processed_text="changed")
        iterator = iter_shakespeare_phase_4_examples(
            tuple(works),
            self.vocabulary,
            self.manifest,
            split="train",
        )

        with self.assertRaises(Phase4GovernanceError) as raised:
            next(iterator)

        rendered = str(raised.exception) + repr(raised.exception)
        self.assertEqual(
            raised.exception.details["invariant"],
            "work.recomputed.sha256",
        )
        self.assertNotIn("changed", rendered)
        with self.assertRaises(TypeError):
            raised.exception.details["invariant"] = "changed"  # type: ignore[index]

    def test_manifest_identity_precedes_count_and_content(self) -> None:
        sealed = _sealed_work()
        changed_manifest = dict(self.manifest)
        changed_manifest["dataset_id"] = "wrong"

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                (sealed,),
                self.vocabulary,
                changed_manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "manifest.dataset_id")
        self.assertEqual(sealed.text_access_count, 0)

    def test_manifest_schema_is_refused_before_count_and_content(self) -> None:
        sealed = _sealed_work()
        changed_manifest = dict(self.manifest)
        changed_manifest["schema_version"] = TEST_SCHEMA_VERSION + 1

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                (sealed,),
                self.vocabulary,
                changed_manifest,
                split="train",
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "manifest.schema_version",
        )
        self.assertEqual(sealed.text_access_count, 0)

    def test_wrong_manifest_object_type_precedes_later_inputs(self) -> None:
        works = _training_works(guarded=True)

        with self.assertRaises(Phase4TypeError) as raised:
            iter_shakespeare_phase_4_examples(
                works,
                object(),  # type: ignore[arg-type]
                "not a manifest",  # type: ignore[arg-type]
                split=3,  # type: ignore[arg-type]
            )

        self.assertEqual(raised.exception.details["invariant"], "manifest.mapping")
        self.assert_guarded_works_unread(works)

    def test_wrong_split_object_type_precedes_vocabulary(self) -> None:
        works = _training_works(guarded=True)

        with self.assertRaises(Phase4TypeError) as raised:
            iter_shakespeare_phase_4_examples(
                works,
                object(),  # type: ignore[arg-type]
                self.manifest,
                split=3,  # type: ignore[arg-type]
            )

        self.assertEqual(raised.exception.details["invariant"], "split.exact_str")
        self.assert_guarded_works_unread(works)

    def test_wrong_vocabulary_object_type_is_refused_before_content(self) -> None:
        works = _training_works(guarded=True)

        with self.assertRaises(Phase4TypeError) as raised:
            iter_shakespeare_phase_4_examples(
                works,
                object(),  # type: ignore[arg-type]
                self.manifest,
                split="train",
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "vocabulary.binding.type",
        )
        self.assert_guarded_works_unread(works)

    def test_forged_vocabulary_binding_is_refused_before_content(self) -> None:
        works = _training_works(guarded=True)
        forged = object.__new__(VocabularyBinding)

        with self.assertRaises(Phase4ContractError) as raised:
            iter_shakespeare_phase_4_examples(
                works,
                forged,
                self.manifest,
                split="train",
            )

        self.assertEqual(
            raised.exception.details["invariant"],
            "vocabulary.binding.verified",
        )
        self.assert_guarded_works_unread(works)

    def test_invalid_split_value_is_refused_before_content(self) -> None:
        works = _training_works(guarded=True)

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                works,
                self.vocabulary,
                self.manifest,
                split="development",
            )

        self.assertEqual(raised.exception.details["invariant"], "split.permitted")
        self.assert_guarded_works_unread(works)

    def test_invalid_work_object_type_is_refused_before_content(self) -> None:
        guarded = list(_training_works(guarded=True))
        works: tuple[Any, ...] = (object(), *guarded[1:])

        with self.assertRaises(Phase4TypeError) as raised:
            iter_shakespeare_phase_4_examples(
                works,  # type: ignore[arg-type]
                self.vocabulary,
                self.manifest,
                split="train",
            )

        self.assertEqual(raised.exception.details["invariant"], "work.extracted_work")
        self.assertEqual(raised.exception.details["position"], 0)
        self.assert_guarded_works_unread(guarded[1:])

    def test_training_work_through_validation_is_refused_before_content(self) -> None:
        work = _make_work(
            *TRAINING_IDENTITIES[0],
            "train",
            TRAINING_TEXTS[0],
            guarded=True,
        )
        assert isinstance(work, GuardedExtractedWork)

        with self.assertRaises(Phase4GovernanceError) as raised:
            iter_shakespeare_phase_4_examples(
                (work,),
                self.vocabulary,
                self.manifest,
                split="validation",
            )

        self.assertEqual(raised.exception.details["invariant"], "work.id")
        self.assertEqual(raised.exception.details["position"], 0)
        self.assertEqual(work.text_access_count, 0)

    def test_public_type_order_precedes_governance(self) -> None:
        sealed = _sealed_work()

        with self.assertRaises(Phase4TypeError) as raised:
            iter_shakespeare_phase_4_examples(
                [sealed],  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                "not a manifest",  # type: ignore[arg-type]
                split=3,  # type: ignore[arg-type]
            )

        self.assertEqual(raised.exception.details["invariant"], "works.exact_tuple")
        self.assertEqual(sealed.text_access_count, 0)


if __name__ == "__main__":
    unittest.main()
