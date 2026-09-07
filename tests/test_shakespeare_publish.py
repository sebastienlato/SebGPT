"""Tests for deterministic, transactional Shakespeare publication."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.data.shakespeare_extract import (  # noqa: E402
    ByteStage,
    ExtractedWork,
    OuterRawRange,
    extract_shakespeare_in_memory,
)
import sebgpt.data.shakespeare_publish as publish_module  # noqa: E402
from sebgpt.data.shakespeare_preflight import Position  # noqa: E402
from sebgpt.data.shakespeare_inventory import inventory_extracted_works  # noqa: E402
from sebgpt.data.shakespeare_publish import (  # noqa: E402
    EXPECTED_RAW_SHA256,
    EXPECTED_OUTPUT_PATHS,
    PROCESSING_MANIFEST_PATH,
    PUBLICATION_LOCK_PATH,
    PUBLISHER_SCRIPT_PATH,
    STAGING_ROOT,
    PublicationError,
    main,
    publish_extracted_works,
)


SECRET_TEST_PROSE = "SEALED TEST PROSE MUST NEVER APPEAR"
RAW_FIXTURE = b"immutable raw fixture\r\n"
WORK_SPECS = (
    (1, "hamlet", "Hamlet", "train"),
    (2, "romeo-and-juliet", "Romeo and Juliet", "train"),
    (3, "macbeth", "Macbeth", "train"),
    (
        4,
        "a-midsummer-nights-dream",
        "A Midsummer Night's Dream",
        "train",
    ),
    (5, "much-ado-about-nothing", "Much Ado About Nothing", "train"),
    (6, "henry-v", "Henry V", "train"),
    (7, "the-tempest", "The Tempest", "validation"),
    (8, "twelfth-night", "Twelfth Night", "test"),
)
ZERO_CANDIDATE_AUDIT = {
    "status": "completed",
    "validation_not_in_train_cardinality": 0,
    "test_not_in_train_cardinality": 0,
    "test_not_in_train_or_validation_cardinality": 0,
    "occurrence_location_candidate_count": 0,
    "occurrence_location_entries": [],
    "occurrence_location_reporting": "not_applicable_for_pinned_corpus",
    "test_only_code_point_identities_emitted": False,
    "test_prose_emitted": False,
}


def _stage(data: bytes) -> ByteStage:
    return ByteStage(hashlib.sha256(data).hexdigest(), len(data))


def _work(
    order: int,
    work_id: str,
    title: str,
    split: str,
    *,
    text: str | None = None,
) -> ExtractedWork:
    processed_text = (
        text
        if text is not None
        else (
            SECRET_TEST_PROSE + "\n"
            if split == "test"
            else f"SYNTHETIC WORK {order}\n"
        )
    )
    processed_bytes = processed_text.encode("utf-8")
    outer_bytes = f"outer-{order}".encode("ascii")
    retained_bytes = f"retained-{order}".encode("ascii")
    base = order * 10_000
    outer_start = Position(order * 10, base, base)
    contents = Position(order * 10 + 2, base + 20, base + 20)
    dramatis = Position(order * 10 + 4, base + 40, base + 40)
    successor = Position(order * 10 + 20, base + 500, base + 500)
    return ExtractedWork(
        work_id=work_id,
        title=title,
        split=split,
        manifest_order=order,
        body_marker=f"WORK {order}",
        successor_marker=f"NEXT {order}",
        processed_text=processed_text,
        outer_range=OuterRawRange(
            start_position=outer_start,
            end_byte_offset=base + 400,
            end_code_point_offset=base + 400,
        ),
        successor_position=successor,
        local_contents_position=contents,
        dramatis_position=dramatis,
        outer_raw=_stage(outer_bytes),
        retained_raw=_stage(retained_bytes),
        processed=_stage(processed_bytes),
        removed_local_contents_byte_count=20,
        processed_code_point_count=len(processed_text),
        processed_line_count=processed_text.count("\n"),
        processed_word_count=len(processed_text.split()),
    )


def _works() -> tuple[ExtractedWork, ...]:
    return tuple(_work(*spec) for spec in WORK_SPECS)


def _position(position: Position) -> dict[str, int]:
    return {
        "line_number": position.line_number,
        "byte_offset": position.byte_offset,
        "code_point_offset": position.code_point_offset,
    }


def _work_result(work: ExtractedWork) -> dict[str, object]:
    return {
        "work_id": work.work_id,
        "title": work.title,
        "manifest_order": work.manifest_order,
        "split": work.split,
        "output_path": EXPECTED_OUTPUT_PATHS[work.work_id],
        "outer_raw": {
            "sha256": work.outer_raw.sha256,
            "byte_count": work.outer_raw.byte_count,
        },
        "retained_raw": {
            "sha256": work.retained_raw.sha256,
            "byte_count": work.retained_raw.byte_count,
        },
        "processed": {
            "sha256": work.processed.sha256,
            "byte_count": work.processed.byte_count,
            "code_point_count": work.processed_code_point_count,
            "line_count": work.processed_line_count,
            "word_count": work.processed_word_count,
        },
        "normalization": work.normalization,
    }


def _excluded_range(work: ExtractedWork) -> dict[str, object]:
    return {
        "work_id": work.work_id,
        "reason": "remove_play_local_contents",
        "start": _position(work.local_contents_position),
        "end_exclusive": _position(work.dramatis_position),
        "byte_count": work.removed_local_contents_byte_count,
    }


def _split_results(works: tuple[ExtractedWork, ...]) -> list[dict[str, object]]:
    results = []
    for split in ("train", "validation", "test"):
        selected = tuple(work for work in works if work.split == split)
        results.append(
            {
                "split": split,
                "work_count": len(selected),
                "processed_byte_count": sum(
                    work.processed.byte_count for work in selected
                ),
                "processed_code_point_count": sum(
                    work.processed_code_point_count for work in selected
                ),
                "processed_line_count": sum(
                    work.processed_line_count for work in selected
                ),
                "processed_word_count": sum(
                    work.processed_word_count for work in selected
                ),
            }
        )
    return results


def _manifest(
    works: tuple[ExtractedWork, ...],
    *,
    raw_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 4,
        "dataset_id": "shakespeare-eight-play",
        "source": {
            "raw_sha256": raw_sha256 or EXPECTED_RAW_SHA256,
        },
        "works": [
            {
                "order": work.manifest_order,
                "work_id": work.work_id,
                "title": work.title,
                "split": work.split,
                "processed_path": EXPECTED_OUTPUT_PATHS[work.work_id],
            }
            for work in works
        ],
        "processing": {
            "script_path": PUBLISHER_SCRIPT_PATH,
            "git_commit": "1" * 40,
            "python_version": "3.14.4",
            "operating_platform": {"system": "Darwin", "machine": "arm64"},
            "transformations": ["crlf_to_lf_only"],
            "excluded_ranges": [_excluded_range(work) for work in works],
            "per_work_results": [_work_result(work) for work in works],
            "per_split_results": _split_results(works),
            "duplicate_and_boundary_audit": {
                "status": "passed",
                "work_count": 8,
                "outer_ranges_non_overlapping": True,
                "processed_hashes_distinct": True,
            },
            "character_inventory_audit": deepcopy(ZERO_CANDIDATE_AUDIT),
        },
    }


def _write_repository(
    root: Path,
    works: tuple[ExtractedWork, ...],
    *,
    raw_bytes: bytes = RAW_FIXTURE,
) -> Path:
    manifest_path = root / "docs/data/shakespeare-eight-play-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            _manifest(works, raw_sha256=EXPECTED_RAW_SHA256),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    raw_path = root / "data/raw/gutenberg-ebook-100/complete-works.txt"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw_bytes)
    return raw_path


def _published_file_paths(root: Path) -> set[str]:
    processed_root = root / "data/processed"
    return {
        path.relative_to(root).as_posix()
        for path in processed_root.rglob("*")
        if path.is_file()
    }


def _expected_published_paths() -> set[str]:
    return {
        *EXPECTED_OUTPUT_PATHS.values(),
        PROCESSING_MANIFEST_PATH.as_posix(),
    }


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            child_key
            for child in value.values()
            for child_key in _all_mapping_keys(child)
        }
    if isinstance(value, list):
        return {
            child_key
            for child in value
            for child_key in _all_mapping_keys(child)
        }
    return set()


class PublisherTests(unittest.TestCase):
    def test_first_publication_has_exact_tree_bytes_and_safe_manifest(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = _write_repository(root, works)
            raw_before = raw_path.read_bytes()

            report = publish_extracted_works(works, root)

            self.assertEqual(report.status, "created")
            self.assertEqual(report.work_count, 8)
            self.assertEqual(_published_file_paths(root), _expected_published_paths())
            for work in works:
                published = root / EXPECTED_OUTPUT_PATHS[work.work_id]
                expected_bytes = work.processed_text.encode("utf-8")
                self.assertEqual(published.read_bytes(), expected_bytes)
                self.assertEqual(published.read_text(encoding="utf-8"), work.processed_text)
                self.assertNotIn(b"\r", expected_bytes)
                self.assertTrue(expected_bytes.endswith(b"\n"))
                self.assertFalse(expected_bytes.endswith(b"\n\n"))
            manifest_bytes = (root / PROCESSING_MANIFEST_PATH).read_bytes()
            generated = json.loads(manifest_bytes.decode("utf-8"))
            self.assertTrue(manifest_bytes.endswith(b"\n"))
            self.assertFalse(manifest_bytes.endswith(b"\n\n"))
            self.assertNotIn(b"\r", manifest_bytes)
            self.assertEqual(generated["schema_version"], 1)
            self.assertEqual(generated["dataset_id"], "shakespeare-eight-play")
            self.assertEqual(
                tuple(item["manifest_order"] for item in generated["works"]),
                tuple(range(1, 9)),
            )
            self.assertEqual(
                tuple(item["split"] for item in generated["splits"]),
                ("train", "validation", "test"),
            )
            self.assertNotIn(SECRET_TEST_PROSE, manifest_bytes.decode("utf-8"))
            self.assertNotIn(str(root), manifest_bytes.decode("utf-8"))
            self.assertNotIn(STAGING_ROOT.as_posix(), manifest_bytes.decode("utf-8"))
            self.assertTrue(
                _all_mapping_keys(generated).isdisjoint(
                    {
                        "absolute_path",
                        "character_identity",
                        "character_name",
                        "code_point",
                        "dirty_state",
                        "generated_at",
                        "hostname",
                        "mtime",
                        "timestamp",
                        "user",
                        "username",
                    }
                )
            )
            self.assertNotIn(SECRET_TEST_PROSE, repr(report))
            self.assertNotIn(SECRET_TEST_PROSE, repr(report.safe_summary()))
            with self.assertRaises(FrozenInstanceError):
                report.safe_summary().status = "changed"  # type: ignore[misc]
            self.assertNotIn(
                "work_id",
                generated["works"][0]["excluded_local_contents"],
            )
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertFalse((root / STAGING_ROOT).exists())
            self.assertFalse((root / PUBLICATION_LOCK_PATH).exists())
            with self.assertRaises(FrozenInstanceError):
                report.status = "changed"  # type: ignore[misc]

    def test_generated_manifest_is_deterministic_across_roots(self) -> None:
        works = _works()
        generated_manifests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_repository(root, works)
                publish_extracted_works(works, root)
                generated_manifests.append(
                    (root / PROCESSING_MANIFEST_PATH).read_bytes()
                )

        self.assertEqual(generated_manifests[0], generated_manifests[1])

    def test_cli_emits_only_safe_summary_in_temporary_root(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch(
                    "sebgpt.data.shakespeare_publish.REPOSITORY_ROOT",
                    root,
                ),
                patch(
                    "sebgpt.data.shakespeare_extract.extract_shakespeare_in_memory",
                    return_value=works,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = main()

            output = stdout.getvalue()
            summary = json.loads(output)
            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(summary["status"], "created")
            self.assertEqual(summary["work_count"], 8)
            self.assertNotIn(SECRET_TEST_PROSE, output)
            self.assertNotIn(str(root), output)

    def test_symlinked_data_root_is_refused_without_external_write(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repository"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            manifest_path = root / "docs/data/shakespeare-eight-play-manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(_manifest(works), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (root / "data").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.data_root.symlink",
            )
            self.assertEqual(tuple(outside.iterdir()), ())

    def test_exact_rerun_is_already_current_and_performs_no_writes(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            first = publish_extracted_works(works, root)
            before = {
                path: (root / path).stat().st_mtime_ns
                for path in _expected_published_paths()
            }

            with patch.object(
                publish_module,
                "_create_staged_file",
                side_effect=AssertionError("rerun attempted a write"),
            ):
                second = publish_extracted_works(works, root)

            after = {
                path: (root / path).stat().st_mtime_ns
                for path in _expected_published_paths()
            }
            self.assertEqual(first.processing_manifest_sha256, second.processing_manifest_sha256)
            self.assertEqual(second.status, "already_current")
            self.assertEqual(before, after)
            self.assertFalse((root / PUBLICATION_LOCK_PATH).exists())

    def test_changed_work_file_is_refused_without_overwrite(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            publish_extracted_works(works, root)
            changed_path = root / EXPECTED_OUTPUT_PATHS["twelfth-night"]
            changed_bytes = b"changed\n"
            changed_path.write_bytes(changed_bytes)

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.file.byte_identity",
            )
            self.assertEqual(changed_path.read_bytes(), changed_bytes)
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
            self.assertFalse((root / STAGING_ROOT).exists())
            self.assertFalse((root / PUBLICATION_LOCK_PATH).exists())

    def test_changed_processing_manifest_is_refused(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            publish_extracted_works(works, root)
            manifest_path = root / PROCESSING_MANIFEST_PATH
            manifest_path.write_bytes(b"{}\n")

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.file.byte_identity",
            )
            self.assertEqual(manifest_path.read_bytes(), b"{}\n")
            self.assertFalse((root / PUBLICATION_LOCK_PATH).exists())

    def test_missing_expected_file_and_partial_target_are_refused(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            partial = root / EXPECTED_OUTPUT_PATHS["hamlet"]
            partial.parent.mkdir(parents=True)
            partial.write_bytes(works[0].processed_text.encode("utf-8"))

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.tree.entries",
            )
            self.assertEqual(_published_file_paths(root), {EXPECTED_OUTPUT_PATHS["hamlet"]})
            self.assertFalse((root / PUBLICATION_LOCK_PATH).exists())

    def test_missing_file_from_complete_publication_is_refused_without_repair(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            publish_extracted_works(works, root)
            missing = root / EXPECTED_OUTPUT_PATHS["macbeth"]
            missing.unlink()

            with self.assertRaises(PublicationError):
                publish_extracted_works(works, root)

            self.assertFalse(missing.exists())
            self.assertFalse((root / PUBLICATION_LOCK_PATH).exists())

    def test_unexpected_file_and_directory_are_refused(self) -> None:
        works = _works()
        for unexpected_kind in ("file", "directory"):
            with self.subTest(unexpected_kind=unexpected_kind):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _write_repository(root, works)
                    publish_extracted_works(works, root)
                    unexpected = root / "data/processed/unexpected"
                    if unexpected_kind == "file":
                        unexpected.write_bytes(b"unexpected")
                    else:
                        unexpected.mkdir()

                    with self.assertRaises(PublicationError) as raised:
                        publish_extracted_works(works, root)

                    self.assertEqual(
                        raised.exception.details["invariant"],
                        "publication.tree.entries",
                    )
                    self.assertTrue(unexpected.exists())

    def test_symlink_is_refused(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            publish_extracted_works(works, root)
            path = root / EXPECTED_OUTPUT_PATHS["hamlet"]
            original = root / "outside.txt"
            original.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(original)

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertIn("symlink", raised.exception.details["invariant"])
            self.assertTrue(path.is_symlink())

    def test_stale_staging_tree_is_refused_and_preserved(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            stale_marker = root / STAGING_ROOT / "stale.marker"
            stale_marker.parent.mkdir()
            stale_marker.write_bytes(b"stale")

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.staging.absent",
            )
            self.assertEqual(stale_marker.read_bytes(), b"stale")
            self.assertFalse((root / "data/processed").exists())

    def test_injected_write_failure_preserves_stage_and_publishes_nothing(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            original_create_file = publish_module._create_staged_file

            def fail_during_stage(parent_fd, name, expected) -> None:
                if expected.work_id == "macbeth":
                    raise OSError("injected failure")
                original_create_file(parent_fd, name, expected)

            with patch.object(
                publish_module,
                "_create_staged_file",
                new=fail_during_stage,
            ):
                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.transaction",
            )
            self.assertNotIn("injected failure", str(raised.exception))
            self.assertEqual(
                raised.exception.details["observed"]["staging_cleanup"],
                "preserved_identity_bound_deletion_unavailable",
            )
            self.assertTrue((root / STAGING_ROOT).is_dir())
            self.assertTrue((root / PUBLICATION_LOCK_PATH).is_file())
            self.assertEqual((root / PUBLICATION_LOCK_PATH).read_bytes(), b"")
            self.assertFalse((root / "data/processed").exists())

    def test_raced_target_is_not_clobbered_by_final_promotion(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = _write_repository(root, works)
            raw_before = raw_path.read_bytes()
            original_promote = publish_module._promote_staging_exclusive
            raced_bytes = b"competing actor state\n"

            def create_target_then_promote(data_root, ownership) -> None:
                raced_target = data_root / "processed"
                raced_target.mkdir()
                (raced_target / "raced.marker").write_bytes(raced_bytes)
                original_promote(data_root, ownership)

            with patch.object(
                publish_module,
                "_promote_staging_exclusive",
                new=create_target_then_promote,
            ):
                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

            raced_target = root / "data/processed"
            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.promotion.exclusive",
            )
            self.assertEqual(
                _published_file_paths(root),
                {"data/processed/raced.marker"},
            )
            self.assertEqual(
                (raced_target / "raced.marker").read_bytes(),
                raced_bytes,
            )
            self.assertTrue((root / STAGING_ROOT).is_dir())
            self.assertTrue((root / PUBLICATION_LOCK_PATH).is_file())
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))

    def test_replaced_staging_identity_is_preserved_during_failure_cleanup(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = _write_repository(root, works)
            raw_before = raw_path.read_bytes()
            original_create_file = publish_module._create_staged_file
            replacement_bytes = b"unrelated replacement state\n"

            def replace_stage_then_fail(parent_fd, name, expected) -> None:
                if expected.work_id == "macbeth":
                    staging = root / STAGING_ROOT
                    moved_owned_stage = root / "data/.moved-owned-stage"
                    staging.rename(moved_owned_stage)
                    staging.mkdir()
                    (staging / "unrelated-user-file.txt").write_bytes(replacement_bytes)
                    raise OSError("injected failure after replacement")
                original_create_file(parent_fd, name, expected)

            with patch.object(
                publish_module,
                "_create_staged_file",
                new=replace_stage_then_fail,
            ):
                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

            replacement = root / STAGING_ROOT / "unrelated-user-file.txt"
            self.assertEqual(replacement.read_bytes(), replacement_bytes)
            self.assertTrue((root / "data/.moved-owned-stage").is_dir())
            self.assertFalse((root / "data/processed").exists())
            self.assertTrue((root / PUBLICATION_LOCK_PATH).is_file())
            self.assertEqual(
                raised.exception.details["observed"]["staging_cleanup"],
                "preserved_staging_identity_mismatch",
            )
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
            self.assertNotIn("injected failure after replacement", str(raised.exception))

    def test_replaced_nested_file_is_never_unlinked_during_failure_cleanup(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = _write_repository(root, works)
            raw_before = raw_path.read_bytes()
            original_create_file = publish_module._create_staged_file
            replacement_bytes = b"unrelated nested replacement\n"

            def replace_nested_file_then_fail(parent_fd, name, expected) -> None:
                if expected.work_id == "macbeth":
                    nested = root / STAGING_ROOT / Path(
                        EXPECTED_OUTPUT_PATHS["hamlet"]
                    ).relative_to("data/processed")
                    nested.unlink()
                    nested.write_bytes(replacement_bytes)
                    raise OSError("injected failure after nested replacement")
                original_create_file(parent_fd, name, expected)

            with patch.object(
                publish_module,
                "_create_staged_file",
                new=replace_nested_file_then_fail,
            ):
                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

            nested = root / STAGING_ROOT / Path(
                EXPECTED_OUTPUT_PATHS["hamlet"]
            ).relative_to("data/processed")
            self.assertEqual(nested.read_bytes(), replacement_bytes)
            self.assertFalse((root / "data/processed").exists())
            self.assertTrue((root / PUBLICATION_LOCK_PATH).is_file())
            self.assertEqual(
                raised.exception.details["observed"]["staging_cleanup"],
                "preserved_identity_bound_deletion_unavailable",
            )
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
            self.assertNotIn("injected failure after nested replacement", str(raised.exception))

    def test_replaced_final_staging_root_is_never_removed(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = _write_repository(root, works)
            raw_before = raw_path.read_bytes()
            original_create_file = publish_module._create_staged_file
            original_matches = publish_module._staging_identity_matches
            moved_owned_stage = root / "data/.moved-owned-stage-before-root-removal"
            replaced = False

            def fail_during_stage(parent_fd, name, expected) -> None:
                if expected.work_id == "macbeth":
                    raise OSError("injected failure before final root removal")
                original_create_file(parent_fd, name, expected)

            def replace_after_identity_check(data_fd, ownership) -> bool:
                nonlocal replaced
                matches = original_matches(data_fd, ownership)
                if matches and not replaced:
                    staging = root / STAGING_ROOT
                    staging.rename(moved_owned_stage)
                    staging.mkdir()
                    replaced = True
                return matches

            with (
                patch.object(
                    publish_module,
                    "_create_staged_file",
                    new=fail_during_stage,
                ),
                patch.object(
                    publish_module,
                    "_staging_identity_matches",
                    new=replace_after_identity_check,
                ),
            ):
                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

            replacement_root = root / STAGING_ROOT
            self.assertTrue(replaced)
            self.assertTrue(replacement_root.is_dir())
            self.assertEqual(tuple(replacement_root.iterdir()), ())
            self.assertTrue(moved_owned_stage.is_dir())
            self.assertFalse((root / "data/processed").exists())
            self.assertTrue((root / PUBLICATION_LOCK_PATH).is_file())
            self.assertEqual(
                raised.exception.details["observed"]["staging_cleanup"],
                "preserved_identity_bound_deletion_unavailable",
            )
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))

    def test_second_cooperating_publisher_is_refused_while_lock_is_held(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            first_lock = publish_module._acquire_publication_lock(root / "data")
            try:
                self.assertTrue((root / PUBLICATION_LOCK_PATH).is_file())
                self.assertEqual((root / PUBLICATION_LOCK_PATH).read_bytes(), b"")

                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

                self.assertEqual(
                    raised.exception.details["invariant"],
                    "publication.lock.acquire",
                )
                self.assertFalse((root / STAGING_ROOT).exists())
                self.assertFalse((root / "data/processed").exists())
                self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
            finally:
                publish_module._release_publication_lock(first_lock)

            self.assertFalse((root / PUBLICATION_LOCK_PATH).exists())

    def test_stale_lock_is_refused_and_never_removed(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            stale_lock = root / PUBLICATION_LOCK_PATH
            stale_lock.write_bytes(b"")

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.lock.acquire",
            )
            self.assertTrue(stale_lock.is_file())
            self.assertEqual(stale_lock.read_bytes(), b"")
            self.assertFalse((root / STAGING_ROOT).exists())
            self.assertFalse((root / "data/processed").exists())

    def test_stale_lock_and_staging_residue_block_next_publication(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            stale_lock = root / PUBLICATION_LOCK_PATH
            stale_lock.write_bytes(b"")
            stale_staging_file = root / STAGING_ROOT / "uncertain.state"
            stale_staging_file.parent.mkdir()
            stale_staging_file.write_bytes(b"preserve me")

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.lock.acquire",
            )
            self.assertEqual(stale_lock.read_bytes(), b"")
            self.assertEqual(stale_staging_file.read_bytes(), b"preserve me")
            self.assertFalse((root / "data/processed").exists())

    def test_symlinked_lock_is_refused_without_following_or_modifying_target(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_repository(root, works)
            external = root / "external-lock-target.txt"
            external_bytes = b"external lock target\n"
            external.write_bytes(external_bytes)
            lock_path = root / PUBLICATION_LOCK_PATH
            lock_path.symlink_to(external)

            with self.assertRaises(PublicationError) as raised:
                publish_extracted_works(works, root)

            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.lock.acquire",
            )
            self.assertTrue(lock_path.is_symlink())
            self.assertEqual(external.read_bytes(), external_bytes)
            self.assertFalse((root / STAGING_ROOT).exists())
            self.assertFalse((root / "data/processed").exists())
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
            self.assertNotIn("injected failure before final root removal", str(raised.exception))

    def test_raced_symlink_at_staged_file_is_never_followed(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = _write_repository(root, works)
            raw_before = raw_path.read_bytes()
            external_file = root / "external-target.txt"
            external_bytes = b"external state must remain unchanged\n"
            external_file.write_bytes(external_bytes)
            original_create_file = publish_module._create_staged_file

            def insert_symlink_then_create(parent_fd, name, expected) -> None:
                if expected.work_id == "hamlet":
                    os.symlink(str(external_file), name, dir_fd=parent_fd)
                original_create_file(parent_fd, name, expected)

            with patch.object(
                publish_module,
                "_create_staged_file",
                new=insert_symlink_then_create,
            ):
                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

            staged_symlink = root / STAGING_ROOT / Path(
                EXPECTED_OUTPUT_PATHS["hamlet"]
            ).relative_to("data/processed")
            self.assertEqual(external_file.read_bytes(), external_bytes)
            self.assertTrue(staged_symlink.is_symlink())
            self.assertFalse((root / "data/processed").exists())
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))

    def test_substituted_intermediate_directory_cannot_redirect_writes(self) -> None:
        works = _works()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = _write_repository(root, works)
            raw_before = raw_path.read_bytes()
            external_directory = root / "external-directory"
            external_directory.mkdir()
            sentinel = external_directory / "sentinel.txt"
            sentinel_bytes = b"external directory state\n"
            sentinel.write_bytes(sentinel_bytes)
            original_create_directory = publish_module._create_owned_directory

            def substitute_after_open(parent_fd, name, relative_path) -> int:
                directory_fd = original_create_directory(
                    parent_fd,
                    name,
                    relative_path,
                )
                if relative_path == "shakespeare-eight-play/train":
                    os.rename(
                        name,
                        ".owned-train-moved",
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    os.symlink(str(external_directory), name, dir_fd=parent_fd)
                return directory_fd

            with patch.object(
                publish_module,
                "_create_owned_directory",
                new=substitute_after_open,
            ):
                with self.assertRaises(PublicationError) as raised:
                    publish_extracted_works(works, root)

            staged_dataset = root / STAGING_ROOT / "shakespeare-eight-play"
            replacement = staged_dataset / "train"
            moved_owned = staged_dataset / ".owned-train-moved"
            self.assertTrue(replacement.is_symlink())
            self.assertTrue(moved_owned.is_dir())
            self.assertEqual(tuple(external_directory.iterdir()), (sentinel,))
            self.assertEqual(sentinel.read_bytes(), sentinel_bytes)
            self.assertFalse((root / "data/processed").exists())
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertEqual(
                raised.exception.details["invariant"],
                "publication.staging.directory_reachable",
            )
            self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))

    def test_invalid_work_invariants_fail_before_staging(self) -> None:
        base_works = _works()
        invalid_cases = {
            "wrong_order": (
                base_works[1],
                base_works[0],
                *base_works[2:],
            ),
            "wrong_hash": (
                replace(
                    base_works[0],
                    processed=replace(base_works[0].processed, sha256="0" * 64),
                ),
                *base_works[1:],
            ),
            "wrong_byte_count": (
                replace(
                    base_works[0],
                    processed=replace(
                        base_works[0].processed,
                        byte_count=base_works[0].processed.byte_count + 1,
                    ),
                ),
                *base_works[1:],
            ),
            "carriage_return": (
                _work(*WORK_SPECS[0], text="invalid\r\n"),
                *base_works[1:],
            ),
            "missing_terminal_lf": (
                _work(*WORK_SPECS[0], text="invalid"),
                *base_works[1:],
            ),
        }
        duplicate_hash_works = list(base_works)
        duplicate_hash_works[1] = _work(
            *WORK_SPECS[1],
            text=base_works[0].processed_text,
        )
        invalid_cases["duplicate_processed_hash"] = tuple(duplicate_hash_works)

        for case_name, invalid_works in invalid_cases.items():
            with self.subTest(case_name=case_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _write_repository(root, invalid_works)
                    with self.assertRaises(PublicationError) as raised:
                        publish_extracted_works(invalid_works, root)
                    self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
                    self.assertFalse((root / STAGING_ROOT).exists())
                    self.assertFalse((root / "data/processed").exists())

    def test_manifest_contract_disagreement_fails_before_staging(self) -> None:
        works = _works()
        mutations = {
            "path": lambda manifest: manifest["works"][0].update(
                {"processed_path": "data/processed/wrong.txt"}
            ),
            "script": lambda manifest: manifest["processing"].update(
                {"script_path": "wrong.py"}
            ),
            "audit": lambda manifest: manifest["processing"][
                "character_inventory_audit"
            ].update({"occurrence_location_candidate_count": 1}),
            "ledger": lambda manifest: manifest["processing"][
                "per_work_results"
            ][0]["processed"].update({"byte_count": 1}),
        }
        for case_name, mutate in mutations.items():
            with self.subTest(case_name=case_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _write_repository(root, works)
                    manifest_path = (
                        root / "docs/data/shakespeare-eight-play-manifest.json"
                    )
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    mutate(manifest)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                    with self.assertRaises(PublicationError) as raised:
                        publish_extracted_works(works, root)

                    self.assertNotIn(SECRET_TEST_PROSE, str(raised.exception))
                    self.assertFalse((root / STAGING_ROOT).exists())
                    self.assertFalse((root / "data/processed").exists())


class ProductionPublisherCompatibilityTests(unittest.TestCase):
    def test_authoritative_production_ledger_matches_accepted_pipeline(self) -> None:
        production_processed_root = REPOSITORY_ROOT / "data/processed"
        self.assertFalse(production_processed_root.exists())
        raw_path = (
            REPOSITORY_ROOT
            / "data/raw/gutenberg-ebook-100/complete-works.txt"
        )
        raw_bytes = raw_path.read_bytes()
        works = extract_shakespeare_in_memory(REPOSITORY_ROOT)
        inventory = inventory_extracted_works(works)
        manifest = json.loads(
            (
                REPOSITORY_ROOT
                / "docs/data/shakespeare-eight-play-manifest.json"
            ).read_text(encoding="utf-8")
        )
        processing = manifest["processing"]
        character_inventory = manifest["character_inventory"]

        expected = publish_module._prepare_publication(works, REPOSITORY_ROOT)

        self.assertEqual(
            hashlib.sha256(raw_bytes).hexdigest(),
            manifest["source"]["raw_sha256"],
        )
        self.assertEqual(processing["status"], "expected_results_populated_publication_not_authorized")
        self.assertEqual(
            processing["git_commit"],
            "cd4cd8159b420920aa66755630fe26f9633a5373",
        )
        self.assertEqual(processing["python_version"], "3.14.4")
        self.assertEqual(
            processing["operating_platform"],
            {"system": "Darwin", "machine": "arm64"},
        )
        self.assertEqual(processing["transformations"], ["crlf_to_lf_only"])
        self.assertEqual(
            processing["global_results"],
            {
                "work_count": len(works),
                "processed_byte_count": sum(
                    work.processed.byte_count for work in works
                ),
                "processed_code_point_count": sum(
                    work.processed_code_point_count for work in works
                ),
                "processed_line_count": sum(
                    work.processed_line_count for work in works
                ),
                "processed_word_count": sum(
                    work.processed_word_count for work in works
                ),
            },
        )
        self.assertEqual(
            character_inventory["per_work_results"],
            [
                {
                    "work_id": item.work_id,
                    "manifest_order": item.manifest_order,
                    "split": item.split,
                    "total_code_point_count": item.total_code_point_count,
                    "distinct_code_point_count": item.distinct_code_point_count,
                }
                for item in inventory.work_inventories
            ],
        )
        self.assertEqual(
            character_inventory["per_split_results"],
            [
                {
                    "split": item.split,
                    "work_count": item.work_count,
                    "total_code_point_count": item.total_code_point_count,
                    "distinct_code_point_count": item.distinct_code_point_count,
                }
                for item in inventory.split_inventories
            ],
        )
        self.assertEqual(
            character_inventory["global_result"],
            {
                "work_count": inventory.global_inventory.work_count,
                "total_code_point_count": (
                    inventory.global_inventory.total_code_point_count
                ),
                "distinct_code_point_count": (
                    inventory.global_inventory.distinct_code_point_count
                ),
            },
        )
        generated = processing["generated_processing_manifest"]
        self.assertEqual(generated["schema_version"], 1)
        self.assertEqual(
            generated["relative_path"],
            PROCESSING_MANIFEST_PATH.as_posix(),
        )
        self.assertEqual(
            generated["sha256"],
            expected.processing_manifest_sha256,
        )
        self.assertEqual(len(expected.report_works), 8)
        self.assertEqual(len(expected.files), 9)
        self.assertEqual(
            publish_module._prepare_publication(works, REPOSITORY_ROOT),
            expected,
        )
        self.assertFalse(production_processed_root.exists())

    def test_production_works_publish_only_inside_temporary_root(self) -> None:
        production_processed_root = REPOSITORY_ROOT / "data/processed"
        self.assertFalse(production_processed_root.exists())
        raw_path = (
            REPOSITORY_ROOT
            / "data/raw/gutenberg-ebook-100/complete-works.txt"
        )
        raw_before = raw_path.read_bytes()
        works = extract_shakespeare_in_memory(REPOSITORY_ROOT)

        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            temporary_manifest = (
                temporary_root
                / "docs/data/shakespeare-eight-play-manifest.json"
            )
            temporary_manifest.parent.mkdir(parents=True)
            temporary_manifest.write_bytes(
                (
                    REPOSITORY_ROOT
                    / "docs/data/shakespeare-eight-play-manifest.json"
                ).read_bytes()
            )
            (temporary_root / "data").mkdir()

            report = publish_extracted_works(works, temporary_root)
            repeated = publish_extracted_works(works, temporary_root)

            self.assertEqual(report.status, "created")
            self.assertEqual(repeated.status, "already_current")
            self.assertEqual(report.work_count, 8)
            self.assertEqual(
                _published_file_paths(temporary_root),
                _expected_published_paths(),
            )
            generated_manifest = (
                temporary_root / PROCESSING_MANIFEST_PATH
            ).read_bytes()
            self.assertNotIn(SECRET_TEST_PROSE.encode("utf-8"), generated_manifest)
            authoritative = json.loads(
                temporary_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(
                hashlib.sha256(generated_manifest).hexdigest(),
                authoritative["processing"]["generated_processing_manifest"][
                    "sha256"
                ],
            )

        self.assertEqual(raw_path.read_bytes(), raw_before)
        self.assertFalse(production_processed_root.exists())


if __name__ == "__main__":
    unittest.main()
