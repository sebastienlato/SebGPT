"""Independent synthetic tests for Phase 8 data construction."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.data.shakespeare_extract import ByteStage, ExtractedWork  # noqa: E402
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)
from sebgpt.training import (  # noqa: E402
    Phase8ContractError,
    Phase8Window,
    build_phase8_windows,
    configure_phase8_runtime,
    create_phase8_epoch_order,
    iter_phase8_logical_batches,
)


TRAINING_IDENTITIES = (
    (1, "hamlet"),
    (2, "romeo-and-juliet"),
    (3, "macbeth"),
    (4, "a-midsummer-nights-dream"),
    (5, "much-ado-about-nothing"),
    (6, "henry-v"),
)


def _stage(text: str) -> ByteStage:
    value = text.encode("utf-8")
    return ByteStage(hashlib.sha256(value).hexdigest(), len(value))


def _work(order: int, work_id: str, split: str, text: str) -> ExtractedWork:
    return ExtractedWork(
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
        normalization="crlf_to_lf_only",
    )


class _GuardedWork(ExtractedWork):
    def __getattribute__(self, name: str):
        if name == "processed_text":
            raise AssertionError("text accessed before metadata refusal")
        return super().__getattribute__(name)


def _fixtures(first_text: str = "A" * 514):
    texts = (first_text, "B!", "C!", "D!", "E!", "F!")
    records = (
        *((order, work_id, "train", text) for (order, work_id), text in zip(
            TRAINING_IDENTITIES,
            texts,
            strict=True,
        )),
        (7, "the-tempest", "validation", "G!"),
    )
    manifest = {
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
    training = tuple(
        _work(order, work_id, "train", text)
        for (order, work_id), text in zip(TRAINING_IDENTITIES, texts, strict=True)
    )
    validation = (_work(7, "the-tempest", "validation", "G!"),)
    return training, validation, manifest


class Phase8DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_phase8_runtime()
        cls.vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)

    def test_context_stride_tail_and_exact_transition_coverage(self) -> None:
        training, _, manifest = _fixtures()
        windows = build_phase8_windows(
            training,
            self.vocabulary,
            manifest,
            split="train",
        )
        hamlet = tuple(window for window in windows if window.work_id == "hamlet")
        self.assertEqual(
            tuple((window.start_index, window.length) for window in hamlet),
            ((0, 256), (256, 256), (512, 1)),
        )
        for window in windows:
            self.assertEqual(window.target_ids[:-1], window.input_ids[1:])
            self.assertLessEqual(window.length, 256)
        covered = tuple(
            index
            for window in hamlet
            for index in range(window.start_index, window.start_index + window.length)
        )
        self.assertEqual(covered, tuple(range(513)))

    def test_work_and_validation_boundaries_remain_separate(self) -> None:
        training, validation, manifest = _fixtures()
        train_windows = build_phase8_windows(
            training,
            self.vocabulary,
            manifest,
            split="train",
        )
        validation_windows = build_phase8_windows(
            validation,
            self.vocabulary,
            manifest,
            split="validation",
        )
        self.assertEqual({window.split for window in train_windows}, {"train"})
        self.assertEqual(
            tuple(window.work_id for window in validation_windows),
            ("the-tempest",),
        )
        self.assertNotIn("twelfth-night", repr((train_windows, validation_windows)))

    def test_empty_and_one_token_documents_produce_no_windows(self) -> None:
        training, _, manifest = _fixtures("A")
        windows = build_phase8_windows(
            training,
            self.vocabulary,
            manifest,
            split="train",
        )
        self.assertFalse(any(window.work_id == "hamlet" for window in windows))

    def test_exact_boundary_lengths(self) -> None:
        expected = {
            2: ((0, 1),),
            257: ((0, 256),),
            258: ((0, 256), (256, 1)),
            513: ((0, 256), (256, 256)),
        }
        for size, expected_windows in expected.items():
            with self.subTest(size=size):
                training, _, manifest = _fixtures("A" * size)
                windows = build_phase8_windows(
                    training,
                    self.vocabulary,
                    manifest,
                    split="train",
                )
                hamlet = tuple(
                    (window.start_index, window.length)
                    for window in windows
                    if window.work_id == "hamlet"
                )
                self.assertEqual(hamlet, expected_windows)

    def test_epoch_shuffle_has_literal_supported_permutations_and_rng_isolation(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8001)
        global_before = torch.get_rng_state().clone()
        first = create_phase8_epoch_order(5, epoch=1, generator=generator)
        state_after_first = generator.get_state().clone()
        second = create_phase8_epoch_order(5, epoch=2, generator=generator)
        self.assertEqual(first, (4, 3, 0, 2, 1))
        self.assertEqual(second, (4, 2, 1, 0, 3))
        self.assertFalse(torch.equal(state_after_first, generator.get_state()))
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_same_generator_state_reproduces_order(self) -> None:
        first = torch.Generator(device="cpu")
        second = torch.Generator(device="cpu")
        first.manual_seed(8001)
        second.manual_seed(8001)
        self.assertEqual(
            create_phase8_epoch_order(9, epoch=1, generator=first),
            create_phase8_epoch_order(9, epoch=1, generator=second),
        )
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))

    def test_logical_batches_retain_final_partial_without_duplication(self) -> None:
        windows = tuple(
            Phase8Window("hamlet", 1, "train", index, (1,), (2,))
            for index in range(9)
        )
        batches = tuple(
            iter_phase8_logical_batches(
                windows,
                tuple(range(9)),
                epoch=1,
            )
        )
        self.assertEqual(tuple(len(batch.window_indices) for batch in batches), (8, 1))
        self.assertEqual(
            tuple(index for batch in batches for index in batch.window_indices),
            tuple(range(9)),
        )
        self.assertEqual(tuple(batch.target_count for batch in batches), (8, 1))

    def test_logical_batch_boundaries_for_one_seven_and_eight(self) -> None:
        for count in (1, 7, 8):
            with self.subTest(count=count):
                windows = tuple(
                    Phase8Window("hamlet", 1, "train", index, (1,), (2,))
                    for index in range(count)
                )
                batches = tuple(
                    iter_phase8_logical_batches(
                        windows,
                        tuple(range(count)),
                        epoch=1,
                    )
                )
                self.assertEqual(len(batches), 1)
                self.assertEqual(len(batches[0].window_indices), count)

    def test_literal_batch_grouping_for_zero_one_seven_eight_nine_and_larger(self) -> None:
        global_before = torch.get_rng_state().clone()
        for count in (0, 1, 7, 8, 9, 17):
            with self.subTest(count=count):
                windows = tuple(
                    Phase8Window(
                        "hamlet",
                        1,
                        "train",
                        index * 256,
                        (1,) * ((index % 3) + 1),
                        (2,) * ((index % 3) + 1),
                    )
                    for index in range(count)
                )
                order = tuple(reversed(range(count)))
                batches = tuple(
                    iter_phase8_logical_batches(windows, order, epoch=4)
                )
                expected_groups = tuple(
                    order[start : start + 8]
                    for start in range(0, count, 8)
                )
                self.assertEqual(len(batches), len(expected_groups))
                self.assertEqual(
                    tuple(batch.window_indices for batch in batches),
                    expected_groups,
                )
                self.assertEqual(
                    tuple(batch.epoch for batch in batches),
                    (4,) * len(expected_groups),
                )
                self.assertEqual(
                    tuple(batch.batch_index for batch in batches),
                    tuple(range(len(expected_groups))),
                )
                self.assertEqual(
                    tuple(batch.target_count for batch in batches),
                    tuple(
                        sum(windows[index].length for index in group)
                        for group in expected_groups
                    ),
                )
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_empty_batching_requires_a_paired_empty_order(self) -> None:
        self.assertEqual(tuple(iter_phase8_logical_batches((), (), epoch=1)), ())
        with self.assertRaises(Phase8ContractError):
            iter_phase8_logical_batches((), (0,), epoch=1)
        window = Phase8Window("hamlet", 1, "train", 0, (1,), (2,))
        with self.assertRaises(Phase8ContractError):
            iter_phase8_logical_batches((window,), (), epoch=1)

    def test_metadata_failure_precedes_text_access(self) -> None:
        training, _, manifest = _fixtures()
        first = training[0]
        guarded = _GuardedWork(**{**first.__dict__, "manifest_order": 99})
        with self.assertRaises(Exception) as caught:
            build_phase8_windows(
                (guarded, *training[1:]),
                self.vocabulary,
                manifest,
                split="train",
            )
        self.assertNotIsInstance(caught.exception, AssertionError)

    def test_order_rejects_non_permutation(self) -> None:
        window = Phase8Window("hamlet", 1, "train", 0, (1,), (2,))
        with self.assertRaises(Phase8ContractError):
            iter_phase8_logical_batches((window,), (1,), epoch=1)

    def test_split_override_is_rejected(self) -> None:
        training, _, manifest = _fixtures()
        with self.assertRaises(Exception) as caught:
            build_phase8_windows(
                training,
                self.vocabulary,
                manifest,
                split="test",
            )
        self.assertEqual(
            caught.exception.details["invariant"],
            "phase8.governance.split",
        )


if __name__ == "__main__":
    unittest.main()
