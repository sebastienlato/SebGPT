"""Deterministic, transactional publication of accepted extracted works."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from sebgpt.data.shakespeare_extract import ExtractedWork, ExtractionError
from sebgpt.data.shakespeare_preflight import (
    MANIFEST_PATH,
    REPOSITORY_ROOT,
    PreflightError,
)


DATASET_ID = "shakespeare-eight-play"
EXPECTED_RAW_SHA256 = (
    "3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37"
)
PROCESSED_ROOT = Path("data/processed")
DATASET_ROOT = PROCESSED_ROOT / DATASET_ID
PROCESSING_MANIFEST_PATH = DATASET_ROOT / "processing-manifest.json"
STAGING_ROOT = Path("data/.processed-staging")
PUBLICATION_LOCK_PATH = Path("data/.publish-lock")
PROCESSING_MANIFEST_SCHEMA_VERSION = 1
PUBLISHER_SCRIPT_PATH = "src/sebgpt/data/shakespeare_publish.py"
PUBLISHER_TEST_PATH = "tests/test_shakespeare_publish.py"
NORMALIZATION_ID = "crlf_to_lf_only"
SPLIT_ORDER = ("train", "validation", "test")
EXPECTED_SPLIT_COUNTS = {"train": 6, "validation": 1, "test": 1}
_AT_FDCWD = -2
_RENAME_EXCL = 0x00000004
_RENAME_NOFOLLOW_ANY = 0x00000010
EXPECTED_OUTPUT_PATHS = {
    "hamlet": "data/processed/shakespeare-eight-play/train/hamlet.txt",
    "romeo-and-juliet": (
        "data/processed/shakespeare-eight-play/train/romeo-and-juliet.txt"
    ),
    "macbeth": "data/processed/shakespeare-eight-play/train/macbeth.txt",
    "a-midsummer-nights-dream": (
        "data/processed/shakespeare-eight-play/train/"
        "a-midsummer-nights-dream.txt"
    ),
    "much-ado-about-nothing": (
        "data/processed/shakespeare-eight-play/train/much-ado-about-nothing.txt"
    ),
    "henry-v": "data/processed/shakespeare-eight-play/train/henry-v.txt",
    "the-tempest": (
        "data/processed/shakespeare-eight-play/validation/the-tempest.txt"
    ),
    "twelfth-night": (
        "data/processed/shakespeare-eight-play/test/twelfth-night.txt"
    ),
}


class PublicationError(Exception):
    """A publication failure containing no source or processed prose."""

    def __init__(
        self,
        invariant: str,
        *,
        expected: Any,
        observed: Any,
        work_id: str | None = None,
        split: str | None = None,
        manifest_order: int | None = None,
        relative_path: str | None = None,
    ) -> None:
        self.details = {
            "invariant": invariant,
            "expected": expected,
            "observed": observed,
        }
        if work_id is not None:
            self.details["work_id"] = work_id
        if split is not None:
            self.details["split"] = split
        if manifest_order is not None:
            self.details["manifest_order"] = manifest_order
        if relative_path is not None:
            self.details["relative_path"] = relative_path
        super().__init__(json.dumps(self.details, ensure_ascii=True, sort_keys=True))


@dataclass(frozen=True)
class PublishedWork:
    work_id: str
    split: str
    manifest_order: int
    relative_path: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class PublicationSummary:
    status: str
    dataset_id: str
    work_count: int
    total_processed_byte_count: int
    processing_manifest_path: str
    processing_manifest_sha256: str


@dataclass(frozen=True)
class PublicationReport:
    status: str
    dataset_id: str
    work_count: int
    total_processed_byte_count: int
    processing_manifest_path: str
    processing_manifest_sha256: str
    works: tuple[PublishedWork, ...] = field(repr=False)

    def safe_summary(self) -> PublicationSummary:
        """Return structural, numeric, and hash facts only."""

        return PublicationSummary(
            status=self.status,
            dataset_id=self.dataset_id,
            work_count=self.work_count,
            total_processed_byte_count=self.total_processed_byte_count,
            processing_manifest_path=self.processing_manifest_path,
            processing_manifest_sha256=self.processing_manifest_sha256,
        )


@dataclass(frozen=True)
class _ExpectedFile:
    relative_path: str
    sha256: str
    byte_count: int
    content: bytes = field(repr=False)
    work_id: str | None = None


@dataclass(frozen=True)
class _ExpectedPublication:
    dataset_id: str
    files: tuple[_ExpectedFile, ...]
    directories: tuple[str, ...]
    report_works: tuple[PublishedWork, ...]
    processing_manifest_sha256: str


@dataclass(frozen=True)
class _StagingOwnership:
    data_device: int
    data_inode: int
    staging_device: int
    staging_inode: int


@dataclass(frozen=True)
class _PublicationLock:
    data_fd: int
    lock_fd: int
    data_device: int
    data_inode: int
    lock_device: int
    lock_inode: int


@dataclass(frozen=True)
class _DirectoryBinding:
    parent_fd: int
    name: str
    directory_fd: int
    relative_path: str


def _require_equal(
    invariant: str,
    expected: Any,
    observed: Any,
    **context: Any,
) -> None:
    if observed != expected:
        raise PublicationError(
            invariant,
            expected=expected,
            observed=observed,
            **context,
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_read_json(path: Path, relative_path: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PublicationError(
            "authoritative_manifest.read",
            expected={"readable": True},
            observed={"readable": False, "error_type": type(error).__name__},
            relative_path=relative_path,
        ) from None
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError(
            "authoritative_manifest.json",
            expected={"valid": True},
            observed={"valid": False, "error_type": type(error).__name__},
            relative_path=relative_path,
        ) from None
    _require_equal(
        "authoritative_manifest.root_type",
        "dict",
        type(value).__name__,
        relative_path=relative_path,
    )
    return value


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    _require_equal(
        f"authoritative_manifest.{key}.type",
        "dict",
        type(value).__name__,
    )
    return value


def _require_list(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    _require_equal(
        f"authoritative_manifest.{key}.type",
        "list",
        type(value).__name__,
    )
    return value


def _validate_relative_output_path(
    work_id: str,
    split: str,
    manifest_order: int,
    value: Any,
) -> str:
    _require_equal(
        "publication.output_path.type",
        "str",
        type(value).__name__,
        work_id=work_id,
        split=split,
        manifest_order=manifest_order,
    )
    path = PurePosixPath(value)
    _require_equal(
        "publication.output_path.relative",
        True,
        not path.is_absolute() and ".." not in path.parts and "." not in path.parts,
        work_id=work_id,
        split=split,
        manifest_order=manifest_order,
    )
    expected = EXPECTED_OUTPUT_PATHS.get(work_id)
    _require_equal(
        "publication.output_path.contract",
        expected,
        value,
        work_id=work_id,
        split=split,
        manifest_order=manifest_order,
        relative_path=value,
    )
    return value


def _position_dict(position: Any) -> dict[str, int]:
    return {
        "line_number": position.line_number,
        "byte_offset": position.byte_offset,
        "code_point_offset": position.code_point_offset,
    }


def _excluded_range_result(work: ExtractedWork) -> dict[str, Any]:
    return {
        "work_id": work.work_id,
        "reason": "remove_play_local_contents",
        "start": _position_dict(work.local_contents_position),
        "end_exclusive": _position_dict(work.dramatis_position),
        "byte_count": work.removed_local_contents_byte_count,
    }


def _work_result(
    work: ExtractedWork,
    title: str,
    output_path: str,
) -> dict[str, Any]:
    return {
        "work_id": work.work_id,
        "title": title,
        "manifest_order": work.manifest_order,
        "split": work.split,
        "output_path": output_path,
        "outer_raw": asdict(work.outer_raw),
        "retained_raw": asdict(work.retained_raw),
        "processed": {
            "sha256": work.processed.sha256,
            "byte_count": work.processed.byte_count,
            "code_point_count": work.processed_code_point_count,
            "line_count": work.processed_line_count,
            "word_count": work.processed_word_count,
        },
        "normalization": work.normalization,
    }


def _split_results(works: tuple[ExtractedWork, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
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


def _validate_work_bytes(work: ExtractedWork) -> bytes:
    try:
        processed_bytes = work.processed_text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise PublicationError(
            "work.processed_text.strict_utf8",
            expected={"encodes": True},
            observed={"encodes": False, "error_type": type(error).__name__},
            work_id=work.work_id,
            split=work.split,
            manifest_order=work.manifest_order,
        ) from None
    context = {
        "work_id": work.work_id,
        "split": work.split,
        "manifest_order": work.manifest_order,
    }
    _require_equal(
        "work.processed.byte_count",
        work.processed.byte_count,
        len(processed_bytes),
        **context,
    )
    _require_equal(
        "work.processed.sha256",
        work.processed.sha256,
        _sha256(processed_bytes),
        **context,
    )
    _require_equal(
        "work.processed.code_point_count",
        work.processed_code_point_count,
        len(work.processed_text),
        **context,
    )
    _require_equal(
        "work.processed.line_count",
        work.processed_line_count,
        work.processed_text.count("\n"),
        **context,
    )
    _require_equal(
        "work.processed.word_count",
        work.processed_word_count,
        len(work.processed_text.split()),
        **context,
    )
    _require_equal(
        "work.processed.carriage_returns",
        0,
        work.processed_text.count("\r"),
        **context,
    )
    _require_equal(
        "work.processed.single_terminal_lf",
        True,
        work.processed_text.endswith("\n")
        and not work.processed_text.endswith("\n\n"),
        **context,
    )
    _require_equal(
        "work.normalization",
        NORMALIZATION_ID,
        work.normalization,
        **context,
    )
    return processed_bytes


def _validate_ranges(works: tuple[ExtractedWork, ...]) -> None:
    ranges = sorted(
        (
            work.outer_range.start_position.byte_offset,
            work.outer_range.end_byte_offset,
            work.work_id,
        )
        for work in works
    )
    for current, following in zip(ranges, ranges[1:]):
        _require_equal(
            "work.outer_ranges_non_overlapping",
            True,
            current[1] <= following[0],
            work_id=current[2],
        )


def _validate_audit(audit: dict[str, Any]) -> dict[str, Any]:
    expected = {
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
    _require_equal(
        "authoritative_manifest.character_inventory_audit",
        expected,
        audit,
    )
    return expected


def _prepare_publication(
    works: tuple[ExtractedWork, ...],
    repository_root: Path,
) -> _ExpectedPublication:
    _require_equal("collection.input_type_tuple", True, isinstance(works, tuple))
    _require_equal("collection.work_count", 8, len(works))
    _require_equal(
        "collection.unique_work_ids",
        len(works),
        len({work.work_id for work in works}),
    )
    _require_equal(
        "collection.unique_manifest_orders",
        len(works),
        len({work.manifest_order for work in works}),
    )
    _require_equal(
        "collection.unique_processed_hashes",
        len(works),
        len({work.processed.sha256 for work in works}),
    )
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        _require_equal(
            "collection.split_work_count",
            expected_count,
            sum(work.split == split for work in works),
            split=split,
        )
    _validate_ranges(works)

    manifest_relative_path = MANIFEST_PATH.as_posix()
    manifest = _safe_read_json(
        repository_root / MANIFEST_PATH,
        manifest_relative_path,
    )
    _require_equal(
        "authoritative_manifest.dataset_id",
        DATASET_ID,
        manifest.get("dataset_id"),
        relative_path=manifest_relative_path,
    )
    source = _require_mapping(manifest, "source")
    raw_sha256 = source.get("raw_sha256")
    _require_equal(
        "authoritative_manifest.raw_sha256",
        EXPECTED_RAW_SHA256,
        raw_sha256,
    )
    manifest_works = _require_list(manifest, "works")
    _require_equal("authoritative_manifest.work_count", 8, len(manifest_works))
    processing = _require_mapping(manifest, "processing")

    script_path = processing.get("script_path")
    _require_equal(
        "authoritative_manifest.processing.script_path",
        PUBLISHER_SCRIPT_PATH,
        script_path,
    )
    code_revision = processing.get("git_commit")
    _require_equal(
        "authoritative_manifest.processing.git_commit.format",
        True,
        isinstance(code_revision, str)
        and len(code_revision) == 40
        and all(character in "0123456789abcdef" for character in code_revision),
    )
    python_version = processing.get("python_version")
    _require_equal(
        "authoritative_manifest.processing.python_version.type",
        "str",
        type(python_version).__name__,
    )
    operating_platform = processing.get("operating_platform")
    _require_equal(
        "authoritative_manifest.processing.operating_platform.type",
        "dict",
        type(operating_platform).__name__,
    )
    _require_equal(
        "authoritative_manifest.processing.operating_platform.keys",
        ("machine", "system"),
        tuple(sorted(operating_platform)),
    )
    for key in ("machine", "system"):
        _require_equal(
            f"authoritative_manifest.processing.operating_platform.{key}.type",
            "str",
            type(operating_platform.get(key)).__name__,
        )
    _require_equal(
        "authoritative_manifest.processing.transformations",
        [NORMALIZATION_ID],
        processing.get("transformations"),
    )
    audit = _validate_audit(_require_mapping(processing, "character_inventory_audit"))

    for index, manifest_work in enumerate(manifest_works):
        _require_equal(
            "authoritative_manifest.work.type",
            "dict",
            type(manifest_work).__name__,
            manifest_order=index + 1,
        )
    ordered_manifest_works = sorted(manifest_works, key=lambda item: item.get("order", 0))
    _require_equal(
        "authoritative_manifest.orders",
        tuple(range(1, 9)),
        tuple(item.get("order") for item in ordered_manifest_works),
    )
    _require_equal(
        "collection.manifest_order",
        tuple(range(1, 9)),
        tuple(work.manifest_order for work in works),
    )

    document_files: list[_ExpectedFile] = []
    report_works: list[PublishedWork] = []
    actual_work_results: list[dict[str, Any]] = []
    actual_excluded_ranges: list[dict[str, Any]] = []
    for work, manifest_work in zip(works, ordered_manifest_works):
        expected_work_id = manifest_work.get("work_id")
        expected_split = manifest_work.get("split")
        expected_order = manifest_work.get("order")
        context = {
            "work_id": work.work_id,
            "split": work.split,
            "manifest_order": work.manifest_order,
        }
        _require_equal(
            "work.work_id", expected_work_id, work.work_id, **context
        )
        _require_equal("work.split", expected_split, work.split, **context)
        _require_equal(
            "work.manifest_order", expected_order, work.manifest_order, **context
        )
        _require_equal(
            "work.title.matches_manifest",
            True,
            work.title == manifest_work.get("title"),
            **context,
        )
        output_path = _validate_relative_output_path(
            work.work_id,
            work.split,
            work.manifest_order,
            manifest_work.get("processed_path"),
        )
        processed_bytes = _validate_work_bytes(work)
        relative_to_processed = PurePosixPath(output_path).relative_to(
            PurePosixPath(PROCESSED_ROOT.as_posix())
        ).as_posix()
        document_files.append(
            _ExpectedFile(
                relative_path=relative_to_processed,
                sha256=work.processed.sha256,
                byte_count=work.processed.byte_count,
                content=processed_bytes,
                work_id=work.work_id,
            )
        )
        report_works.append(
            PublishedWork(
                work_id=work.work_id,
                split=work.split,
                manifest_order=work.manifest_order,
                relative_path=output_path,
                sha256=work.processed.sha256,
                byte_count=work.processed.byte_count,
            )
        )
        actual_work_results.append(
            _work_result(work, manifest_work.get("title"), output_path)
        )
        actual_excluded_ranges.append(_excluded_range_result(work))

    _require_equal(
        "collection.unique_output_paths",
        len(document_files),
        len({item.relative_path for item in document_files}),
    )
    actual_split_results = _split_results(works)
    _require_equal(
        "authoritative_manifest.processing.per_work_results",
        actual_work_results,
        processing.get("per_work_results"),
    )
    _require_equal(
        "authoritative_manifest.processing.per_split_results",
        actual_split_results,
        processing.get("per_split_results"),
    )
    _require_equal(
        "authoritative_manifest.processing.excluded_ranges",
        actual_excluded_ranges,
        processing.get("excluded_ranges"),
    )
    expected_duplicate_audit = {
        "status": "passed",
        "work_count": 8,
        "outer_ranges_non_overlapping": True,
        "processed_hashes_distinct": True,
    }
    _require_equal(
        "authoritative_manifest.processing.duplicate_and_boundary_audit",
        expected_duplicate_audit,
        processing.get("duplicate_and_boundary_audit"),
    )

    generated_manifest = {
        "schema_version": PROCESSING_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "raw_sha256": raw_sha256,
        "processing": {
            "script_path": script_path,
            "code_revision": code_revision,
            "python_version": python_version,
            "operating_system": operating_platform["system"],
            "machine": operating_platform["machine"],
            "normalization": NORMALIZATION_ID,
        },
        "unseen_character_audit": {
            key: value
            for key, value in audit.items()
            if key not in {"test_only_code_point_identities_emitted", "test_prose_emitted"}
        },
        "splits": actual_split_results,
        "works": [
            {
                **work_result,
                "excluded_local_contents": {
                    key: value
                    for key, value in excluded_range.items()
                    if key != "work_id"
                },
            }
            for work_result, excluded_range in zip(
                actual_work_results, actual_excluded_ranges
            )
        ],
    }
    manifest_bytes = (
        json.dumps(generated_manifest, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    manifest_relative = PurePosixPath(PROCESSING_MANIFEST_PATH.as_posix()).relative_to(
        PurePosixPath(PROCESSED_ROOT.as_posix())
    ).as_posix()
    manifest_file = _ExpectedFile(
        relative_path=manifest_relative,
        sha256=_sha256(manifest_bytes),
        byte_count=len(manifest_bytes),
        content=manifest_bytes,
    )
    all_files = tuple((*document_files, manifest_file))
    directories = tuple(
        sorted(
            {
                PurePosixPath(item.relative_path).parent.as_posix()
                for item in all_files
            }
            | {DATASET_ID}
        )
    )
    return _ExpectedPublication(
        dataset_id=DATASET_ID,
        files=all_files,
        directories=directories,
        report_works=tuple(report_works),
        processing_manifest_sha256=manifest_file.sha256,
    )


def _entry_inventory(root: Path) -> tuple[set[str], set[str]]:
    directories: set[str] = set()
    files: set[str] = set()
    try:
        entries = tuple(root.rglob("*"))
    except OSError as error:
        raise PublicationError(
            "publication.tree.scan",
            expected={"readable": True},
            observed={"readable": False, "error_type": type(error).__name__},
            relative_path=PROCESSED_ROOT.as_posix(),
        ) from None
    for entry in entries:
        relative_path = entry.relative_to(root).as_posix()
        _require_equal(
            "publication.tree.symlink",
            False,
            entry.is_symlink(),
            relative_path=(PROCESSED_ROOT / relative_path).as_posix(),
        )
        if entry.is_dir():
            directories.add(relative_path)
        elif entry.is_file():
            files.add(relative_path)
        else:
            raise PublicationError(
                "publication.tree.entry_type",
                expected="regular_file_or_directory",
                observed="other",
                relative_path=(PROCESSED_ROOT / relative_path).as_posix(),
            )
    return directories, files


def _verify_file(root: Path, expected: _ExpectedFile) -> None:
    relative_path = (PROCESSED_ROOT / expected.relative_path).as_posix()
    path = root / expected.relative_path
    _require_equal(
        "publication.file.symlink",
        False,
        path.is_symlink(),
        relative_path=relative_path,
        work_id=expected.work_id,
    )
    _require_equal(
        "publication.file.regular",
        True,
        path.is_file(),
        relative_path=relative_path,
        work_id=expected.work_id,
    )
    try:
        observed_bytes = path.read_bytes()
    except OSError as error:
        raise PublicationError(
            "publication.file.read",
            expected={"readable": True},
            observed={"readable": False, "error_type": type(error).__name__},
            relative_path=relative_path,
            work_id=expected.work_id,
        ) from None
    observed_hash = _sha256(observed_bytes)
    _require_equal(
        "publication.file.byte_identity",
        {"sha256": expected.sha256, "byte_count": expected.byte_count},
        {"sha256": observed_hash, "byte_count": len(observed_bytes)},
        relative_path=relative_path,
        work_id=expected.work_id,
    )
    _require_equal(
        "publication.file.exact_bytes",
        True,
        observed_bytes == expected.content,
        relative_path=relative_path,
        work_id=expected.work_id,
    )
    try:
        decoded = observed_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PublicationError(
            "publication.file.strict_utf8",
            expected={"decodes": True},
            observed={"decodes": False, "error_type": type(error).__name__},
            relative_path=relative_path,
            work_id=expected.work_id,
        ) from None
    _require_equal(
        "publication.file.carriage_returns",
        0,
        decoded.count("\r"),
        relative_path=relative_path,
        work_id=expected.work_id,
    )
    _require_equal(
        "publication.file.single_terminal_lf",
        True,
        decoded.endswith("\n") and not decoded.endswith("\n\n"),
        relative_path=relative_path,
        work_id=expected.work_id,
    )
    if expected.work_id is None:
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as error:
            raise PublicationError(
                "publication.processing_manifest.json",
                expected={"valid": True},
                observed={"valid": False, "error_type": type(error).__name__},
                relative_path=relative_path,
            ) from None
        _require_equal(
            "publication.processing_manifest.root_type",
            "dict",
            type(parsed).__name__,
            relative_path=relative_path,
        )


def _verify_tree(root: Path, expected: _ExpectedPublication) -> None:
    _require_equal(
        "publication.root.symlink",
        False,
        root.is_symlink(),
        relative_path=PROCESSED_ROOT.as_posix(),
    )
    _require_equal(
        "publication.root.directory",
        True,
        root.is_dir(),
        relative_path=PROCESSED_ROOT.as_posix(),
    )
    actual_directories, actual_files = _entry_inventory(root)
    expected_directories = set(expected.directories)
    expected_files = {item.relative_path for item in expected.files}
    missing_directories = sorted(expected_directories - actual_directories)
    unexpected_directories = sorted(actual_directories - expected_directories)
    missing_files = sorted(expected_files - actual_files)
    unexpected_files = sorted(actual_files - expected_files)
    _require_equal(
        "publication.tree.entries",
        {
            "directory_count": len(expected_directories),
            "file_count": len(expected_files),
            "missing_count": 0,
            "unexpected_count": 0,
        },
        {
            "directory_count": len(actual_directories),
            "file_count": len(actual_files),
            "missing_count": len(missing_directories) + len(missing_files),
            "unexpected_count": len(unexpected_directories) + len(unexpected_files),
        },
        relative_path=PROCESSED_ROOT.as_posix(),
    )
    for expected_file in expected.files:
        _verify_file(root, expected_file)


def _report(expected: _ExpectedPublication, status: str) -> PublicationReport:
    return PublicationReport(
        status=status,
        dataset_id=expected.dataset_id,
        work_count=len(expected.report_works),
        total_processed_byte_count=sum(
            work.byte_count for work in expected.report_works
        ),
        processing_manifest_path=PROCESSING_MANIFEST_PATH.as_posix(),
        processing_manifest_sha256=expected.processing_manifest_sha256,
        works=expected.report_works,
    )


def _capture_staging_ownership(
    data_fd: int,
    staging_fd: int,
) -> _StagingOwnership:
    data_observed = os.fstat(data_fd)
    staging_observed = os.fstat(staging_fd)
    _require_equal(
        "publication.data_root.owned_directory",
        True,
        stat.S_ISDIR(data_observed.st_mode),
        relative_path="data",
    )
    _require_equal(
        "publication.staging.owned_directory",
        True,
        stat.S_ISDIR(staging_observed.st_mode),
        relative_path=STAGING_ROOT.as_posix(),
    )
    return _StagingOwnership(
        data_device=data_observed.st_dev,
        data_inode=data_observed.st_ino,
        staging_device=staging_observed.st_dev,
        staging_inode=staging_observed.st_ino,
    )


def _open_verified_data_directory(data_root: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    data_fd = os.open(data_root, flags)
    observed_path = os.stat(data_root, follow_symlinks=False)
    observed_fd = os.fstat(data_fd)
    if not stat.S_ISDIR(observed_fd.st_mode) or not os.path.samestat(
        observed_path, observed_fd
    ):
        os.close(data_fd)
        raise PublicationError(
            "publication.data_root.identity",
            expected={"verified_directory": True},
            observed={"verified_directory": False},
            relative_path="data",
        )
    return data_fd


def _acquire_publication_lock(data_root: Path) -> _PublicationLock:
    """Atomically acquire the empty cooperating-publisher presence lock."""

    data_fd = _open_verified_data_directory(data_root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        lock_fd = os.open(
            PUBLICATION_LOCK_PATH.name,
            flags,
            0o600,
            dir_fd=data_fd,
        )
    except OSError as error:
        os.close(data_fd)
        raise PublicationError(
            "publication.lock.acquire",
            expected={"acquired": True, "preexisting": False},
            observed={
                "acquired": False,
                "errno": error.errno,
                "error_name": errno.errorcode.get(error.errno, "UNKNOWN"),
            },
            relative_path=PUBLICATION_LOCK_PATH.as_posix(),
        ) from None
    data_observed = os.fstat(data_fd)
    lock_observed = os.fstat(lock_fd)
    try:
        lock_path_observed = os.stat(
            PUBLICATION_LOCK_PATH.name,
            dir_fd=data_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        os.close(lock_fd)
        os.close(data_fd)
        raise PublicationError(
            "publication.lock.identity",
            expected={"created_and_reachable": True},
            observed={
                "created_and_reachable": False,
                "error_type": type(error).__name__,
            },
            relative_path=PUBLICATION_LOCK_PATH.as_posix(),
        ) from None
    if (
        not stat.S_ISREG(lock_observed.st_mode)
        or lock_observed.st_size != 0
        or not os.path.samestat(lock_path_observed, lock_observed)
    ):
        os.close(lock_fd)
        os.close(data_fd)
        raise PublicationError(
            "publication.lock.identity",
            expected={"regular": True, "empty": True, "reachable": True},
            observed={
                "regular": stat.S_ISREG(lock_observed.st_mode),
                "empty": lock_observed.st_size == 0,
                "reachable": os.path.samestat(lock_path_observed, lock_observed),
            },
            relative_path=PUBLICATION_LOCK_PATH.as_posix(),
        )
    return _PublicationLock(
        data_fd=data_fd,
        lock_fd=lock_fd,
        data_device=data_observed.st_dev,
        data_inode=data_observed.st_ino,
        lock_device=lock_observed.st_dev,
        lock_inode=lock_observed.st_ino,
    )


def _close_preserving_publication_lock(lock: _PublicationLock) -> None:
    os.close(lock.lock_fd)
    os.close(lock.data_fd)


def _release_publication_lock(lock: _PublicationLock) -> None:
    """Release a verified lock under the accepted cooperative-workspace model."""

    data_observed = os.fstat(lock.data_fd)
    lock_observed = os.fstat(lock.lock_fd)
    try:
        path_observed = os.stat(
            PUBLICATION_LOCK_PATH.name,
            dir_fd=lock.data_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        _close_preserving_publication_lock(lock)
        raise PublicationError(
            "publication.lock.release_identity",
            expected={"owned_lock_reachable": True},
            observed={
                "owned_lock_reachable": False,
                "error_type": type(error).__name__,
            },
            relative_path=PUBLICATION_LOCK_PATH.as_posix(),
        ) from None
    identity_matches = (
        (data_observed.st_dev, data_observed.st_ino)
        == (lock.data_device, lock.data_inode)
        and (lock_observed.st_dev, lock_observed.st_ino)
        == (lock.lock_device, lock.lock_inode)
        and os.path.samestat(path_observed, lock_observed)
    )
    if not identity_matches:
        _close_preserving_publication_lock(lock)
        raise PublicationError(
            "publication.lock.release_identity",
            expected={"owned_lock_reachable": True},
            observed={"owned_lock_reachable": False},
            relative_path=PUBLICATION_LOCK_PATH.as_posix(),
        )
    try:
        os.unlink(PUBLICATION_LOCK_PATH.name, dir_fd=lock.data_fd)
    except OSError as error:
        _close_preserving_publication_lock(lock)
        raise PublicationError(
            "publication.lock.release",
            expected={"released": True},
            observed={
                "released": False,
                "error_type": type(error).__name__,
            },
            relative_path=PUBLICATION_LOCK_PATH.as_posix(),
        ) from None
    _close_preserving_publication_lock(lock)


def _create_owned_directory(
    parent_fd: int,
    name: str,
    relative_path: str,
) -> int:
    """Exclusively create and retain a no-follow directory descriptor."""

    os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(name, flags, dir_fd=parent_fd)
    observed_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    observed_fd = os.fstat(directory_fd)
    if not stat.S_ISDIR(observed_fd.st_mode) or not os.path.samestat(
        observed_path, observed_fd
    ):
        os.close(directory_fd)
        raise PublicationError(
            "publication.staging.directory_identity",
            expected={"created_and_bound": True},
            observed={"created_and_bound": False},
            relative_path=(PROCESSED_ROOT / relative_path).as_posix(),
        )
    return directory_fd


def _verify_directory_binding(binding: _DirectoryBinding) -> None:
    observed_path = os.stat(
        binding.name,
        dir_fd=binding.parent_fd,
        follow_symlinks=False,
    )
    observed_fd = os.fstat(binding.directory_fd)
    _require_equal(
        "publication.staging.directory_reachable",
        True,
        stat.S_ISDIR(observed_path.st_mode)
        and os.path.samestat(observed_path, observed_fd),
        relative_path=(PROCESSED_ROOT / binding.relative_path).as_posix(),
    )


def _write_all(file_fd: int, content: bytes) -> None:
    view = memoryview(content)
    written = 0
    while written < len(view):
        count = os.write(file_fd, view[written:])
        if count <= 0:
            raise OSError(errno.EIO, "write made no progress")
        written += count


def _read_all(file_fd: int) -> bytes:
    os.lseek(file_fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(file_fd, 64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _create_staged_file(
    parent_fd: int,
    name: str,
    expected: _ExpectedFile,
) -> None:
    """Exclusively create, write, and verify one file through its descriptor."""

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    file_fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    try:
        created = os.fstat(file_fd)
        _require_equal(
            "publication.staging.file_regular",
            True,
            stat.S_ISREG(created.st_mode),
            relative_path=(
                PROCESSED_ROOT / PurePosixPath(expected.relative_path)
            ).as_posix(),
            work_id=expected.work_id,
        )
        _write_all(file_fd, expected.content)
        os.fsync(file_fd)
        observed_bytes = _read_all(file_fd)
        _require_equal(
            "publication.staging.file_complete_write",
            {
                "sha256": expected.sha256,
                "byte_count": expected.byte_count,
            },
            {
                "sha256": _sha256(observed_bytes),
                "byte_count": len(observed_bytes),
            },
            relative_path=(
                PROCESSED_ROOT / PurePosixPath(expected.relative_path)
            ).as_posix(),
            work_id=expected.work_id,
        )
        observed_path = os.stat(
            name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        _require_equal(
            "publication.staging.file_reachable",
            True,
            stat.S_ISREG(observed_path.st_mode)
            and os.path.samestat(observed_path, created),
            relative_path=(
                PROCESSED_ROOT / PurePosixPath(expected.relative_path)
            ).as_posix(),
            work_id=expected.work_id,
        )
    finally:
        os.close(file_fd)


def _construct_staging_tree(
    data_root: Path,
    expected: _ExpectedPublication,
    ownership_holder: list[_StagingOwnership],
) -> _StagingOwnership:
    """Create the complete staging tree through retained directory descriptors."""

    data_fd = _open_verified_data_directory(data_root)
    opened_directory_fds: list[int] = [data_fd]
    bindings: list[_DirectoryBinding] = []
    try:
        staging_fd = _create_owned_directory(
            data_fd,
            STAGING_ROOT.name,
            STAGING_ROOT.name,
        )
        opened_directory_fds.append(staging_fd)
        bindings.append(
            _DirectoryBinding(
                parent_fd=data_fd,
                name=STAGING_ROOT.name,
                directory_fd=staging_fd,
                relative_path=STAGING_ROOT.name,
            )
        )
        ownership = _capture_staging_ownership(data_fd, staging_fd)
        ownership_holder.append(ownership)
        directory_fds = {"": staging_fd}
        for relative_directory in sorted(
            expected.directories,
            key=lambda value: (len(PurePosixPath(value).parts), value),
        ):
            relative = PurePosixPath(relative_directory)
            parent_relative = relative.parent.as_posix()
            if parent_relative == ".":
                parent_relative = ""
            parent_fd = directory_fds[parent_relative]
            directory_fd = _create_owned_directory(
                parent_fd,
                relative.name,
                relative_directory,
            )
            opened_directory_fds.append(directory_fd)
            directory_fds[relative_directory] = directory_fd
            bindings.append(
                _DirectoryBinding(
                    parent_fd=parent_fd,
                    name=relative.name,
                    directory_fd=directory_fd,
                    relative_path=relative_directory,
                )
            )
        for expected_file in expected.files:
            relative = PurePosixPath(expected_file.relative_path)
            parent_relative = relative.parent.as_posix()
            if parent_relative == ".":
                parent_relative = ""
            _create_staged_file(
                directory_fds[parent_relative],
                relative.name,
                expected_file,
            )
        for binding in bindings:
            _verify_directory_binding(binding)
        return ownership
    finally:
        for directory_fd in reversed(opened_directory_fds):
            os.close(directory_fd)


def _open_owned_data_directory(
    data_root: Path,
    ownership: _StagingOwnership,
) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        data_fd = os.open(data_root, flags)
    except OSError as error:
        raise PublicationError(
            "publication.data_root.open",
            expected={"owned_directory": True},
            observed={
                "owned_directory": False,
                "error_type": type(error).__name__,
            },
            relative_path="data",
        ) from None
    observed = os.fstat(data_fd)
    if (observed.st_dev, observed.st_ino) != (
        ownership.data_device,
        ownership.data_inode,
    ):
        os.close(data_fd)
        raise PublicationError(
            "publication.data_root.identity",
            expected={"matches_owned": True},
            observed={"matches_owned": False},
            relative_path="data",
        )
    return data_fd


def _staging_identity_matches(
    data_fd: int,
    ownership: _StagingOwnership,
) -> bool:
    try:
        observed = os.stat(
            STAGING_ROOT.name,
            dir_fd=data_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return (
        stat.S_ISDIR(observed.st_mode)
        and (observed.st_dev, observed.st_ino)
        == (ownership.staging_device, ownership.staging_inode)
    )


def _cleanup_owned_staging(
    data_root: Path,
    ownership: _StagingOwnership | None,
) -> str:
    """Inspect ownership but preserve failed staging when deletion cannot be bound."""

    if ownership is None:
        return "not_created"
    try:
        data_fd = _open_owned_data_directory(data_root, ownership)
    except PublicationError:
        return "preserved_data_root_identity_unverified"
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            staging_fd = os.open(STAGING_ROOT.name, flags, dir_fd=data_fd)
        except OSError:
            return "preserved_staging_identity_mismatch"
        try:
            observed = os.fstat(staging_fd)
            if (
                not stat.S_ISDIR(observed.st_mode)
                or (observed.st_dev, observed.st_ino)
                != (ownership.staging_device, ownership.staging_inode)
            ):
                return "preserved_staging_identity_mismatch"
        finally:
            os.close(staging_fd)
        if not _staging_identity_matches(data_fd, ownership):
            return "preserved_staging_path_replaced_during_cleanup"
        return "preserved_identity_bound_deletion_unavailable"
    finally:
        os.close(data_fd)


def _promote_staging_exclusive(
    data_root: Path,
    ownership: _StagingOwnership,
) -> None:
    """Atomically rename staging only if the final target does not exist."""

    _require_equal(
        "publication.promotion.platform",
        "darwin",
        sys.platform,
        relative_path=PROCESSED_ROOT.as_posix(),
    )
    data_fd = _open_owned_data_directory(data_root, ownership)
    try:
        _require_equal(
            "publication.staging.identity",
            True,
            _staging_identity_matches(data_fd, ownership),
            relative_path=STAGING_ROOT.as_posix(),
        )
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameatx_np = libc.renameatx_np
        except AttributeError:
            raise PublicationError(
                "publication.promotion.api",
                expected={"renameatx_np": True},
                observed={"renameatx_np": False},
                relative_path=PROCESSED_ROOT.as_posix(),
            ) from None
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameatx_np(
            data_fd,
            os.fsencode(STAGING_ROOT.name),
            data_fd,
            os.fsencode(PROCESSED_ROOT.name),
            _RENAME_EXCL | _RENAME_NOFOLLOW_ANY,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise PublicationError(
                "publication.promotion.exclusive",
                expected={"promoted": True, "target_absent": True},
                observed={
                    "promoted": False,
                    "errno": error_number,
                    "error_name": errno.errorcode.get(error_number, "UNKNOWN"),
                },
                relative_path=PROCESSED_ROOT.as_posix(),
            )
    finally:
        os.close(data_fd)


def publish_extracted_works(
    works: tuple[ExtractedWork, ...],
    repository_root: Path,
) -> PublicationReport:
    """Publish one complete verified tree, or accept an exact existing tree."""

    expected = _prepare_publication(works, repository_root)
    data_root = repository_root / "data"
    target_root = repository_root / PROCESSED_ROOT
    staging_root = repository_root / STAGING_ROOT

    _require_equal(
        "publication.data_root.symlink",
        False,
        data_root.is_symlink(),
        relative_path="data",
    )
    _require_equal(
        "publication.data_root.directory",
        True,
        data_root.is_dir(),
        relative_path="data",
    )
    publication_lock = _acquire_publication_lock(data_root)
    staging_ownership: _StagingOwnership | None = None
    ownership_holder: list[_StagingOwnership] = []
    uncertain_namespace_state = False
    try:
        staging_absent = not staging_root.exists() and not staging_root.is_symlink()
        if not staging_absent:
            uncertain_namespace_state = True
        _require_equal(
            "publication.staging.absent",
            True,
            staging_absent,
            relative_path=STAGING_ROOT.as_posix(),
        )
        if target_root.exists() or target_root.is_symlink():
            _verify_tree(target_root, expected)
            report = _report(expected, "already_current")
        else:
            uncertain_namespace_state = True
            staging_ownership = _construct_staging_tree(
                data_root,
                expected,
                ownership_holder,
            )
            _verify_tree(staging_root, expected)
            _promote_staging_exclusive(data_root, staging_ownership)
            staging_ownership = None
            _verify_tree(target_root, expected)
            report = _report(expected, "created")
    except PublicationError:
        if staging_ownership is None and ownership_holder:
            staging_ownership = ownership_holder[0]
        _cleanup_owned_staging(data_root, staging_ownership)
        if uncertain_namespace_state:
            _close_preserving_publication_lock(publication_lock)
        else:
            _release_publication_lock(publication_lock)
        raise
    except OSError as error:
        if staging_ownership is None and ownership_holder:
            staging_ownership = ownership_holder[0]
        cleanup_status = _cleanup_owned_staging(data_root, staging_ownership)
        if uncertain_namespace_state:
            _close_preserving_publication_lock(publication_lock)
            lock_status = "preserved_uncertain_transaction"
        else:
            _release_publication_lock(publication_lock)
            lock_status = "released_prepublication_failure"
        raise PublicationError(
            "publication.transaction",
            expected={"completed": True},
            observed={
                "completed": False,
                "error_type": type(error).__name__,
                "publication_lock": lock_status,
                "staging_cleanup": cleanup_status,
            },
            relative_path=PROCESSED_ROOT.as_posix(),
        ) from None
    _release_publication_lock(publication_lock)
    return report


def main() -> int:
    """Run extraction then publication, emitting only a content-safe summary."""

    from sebgpt.data.shakespeare_extract import extract_shakespeare_in_memory

    try:
        works = extract_shakespeare_in_memory(REPOSITORY_ROOT)
        report = publish_extracted_works(works, REPOSITORY_ROOT)
    except (PublicationError, ExtractionError, PreflightError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except (KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "invariant": "publication.schema",
                    "expected": {"valid": True},
                    "observed": {
                        "valid": False,
                        "error_type": type(error).__name__,
                    },
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(asdict(report.safe_summary()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
