"""Independent tests for the safe Phase 4 permitted-corpus factory."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.model.phase4_corpus as corpus_module  # noqa: E402
import sebgpt.model.phase4_experiment as experiment_module  # noqa: E402
from sebgpt.data.shifted_examples import (  # noqa: E402
    Phase4GovernanceError,
    Phase4TypeError,
)
from sebgpt.model.phase4_corpus import load_phase4_permitted_corpus  # noqa: E402
from sebgpt.model.phase4_experiment import (  # noqa: E402
    Phase4PermittedCorpus,
    load_phase4_experiment_config,
    validate_phase4_experiment_preflight,
)
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    load_accepted_vocabulary_binding,
)


EXPECTED_IDENTITIES = (
    (1, "hamlet", "train"),
    (2, "romeo-and-juliet", "train"),
    (3, "macbeth", "train"),
    (4, "a-midsummer-nights-dream", "train"),
    (5, "much-ado-about-nothing", "train"),
    (6, "henry-v", "train"),
    (7, "the-tempest", "validation"),
)
SEALED_RELATIVE_PATH = Path(
    "data/processed/shakespeare-eight-play/test/twelfth-night.txt"
)
RAW_RELATIVE_PATH = Path(
    "data/raw/gutenberg-ebook-100/complete-works.txt"
)
EXPECTED_PHASE_1_MANIFEST_SHA256 = (
    "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"
)
EXPECTED_PROCESSING_MANIFEST_SHA256 = (
    "bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc"
)


def _json(relative_path: str):
    return json.loads((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))


class SafePhase4CorpusFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase_1_manifest = _json(
            "docs/data/shakespeare-eight-play-manifest.json"
        )
        cls.processing_manifest = _json(
            "data/processed/shakespeare-eight-play/processing-manifest.json"
        )

    def test_manifest_artifact_hashes_match_test_local_and_production_authority(self) -> None:
        phase_1_bytes = (
            REPOSITORY_ROOT / "docs/data/shakespeare-eight-play-manifest.json"
        ).read_bytes()
        processing_bytes = (
            REPOSITORY_ROOT
            / "data/processed/shakespeare-eight-play/processing-manifest.json"
        ).read_bytes()

        self.assertEqual(
            hashlib.sha256(phase_1_bytes).hexdigest(),
            EXPECTED_PHASE_1_MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(processing_bytes).hexdigest(),
            EXPECTED_PROCESSING_MANIFEST_SHA256,
        )
        self.assertEqual(
            corpus_module.PHASE_1_MANIFEST_SHA256,
            EXPECTED_PHASE_1_MANIFEST_SHA256,
        )
        self.assertEqual(
            corpus_module.PROCESSING_MANIFEST_SHA256,
            EXPECTED_PROCESSING_MANIFEST_SHA256,
        )

    def test_public_factory_returns_exact_seven_work_handoff_in_order(self) -> None:
        corpus = load_phase4_permitted_corpus(REPOSITORY_ROOT)
        works = (*corpus.training_works, *corpus.validation_works)

        self.assertIsInstance(corpus, Phase4PermittedCorpus)
        self.assertEqual(len(corpus.training_works), 6)
        self.assertEqual(len(corpus.validation_works), 1)
        self.assertEqual(
            tuple((work.manifest_order, work.work_id, work.split) for work in works),
            EXPECTED_IDENTITIES,
        )
        self.assertIsNone(corpus.sealed_test_supplier)
        self.assertEqual(
            (
                corpus.training_source_tokens,
                corpus.training_targets,
                corpus.training_examples,
                corpus.validation_source_tokens,
                corpus.validation_targets,
                corpus.validation_examples,
            ),
            (792_705, 792_699, 12_389, 98_296, 98_295, 1_536),
        )

    def test_factory_reads_only_two_manifests_and_seven_permitted_files(self) -> None:
        real_read = corpus_module._read_bytes
        observed: list[Path] = []

        def read(repository_root, relative_path, *, invariant):
            observed.append(relative_path)
            if relative_path in (SEALED_RELATIVE_PATH, RAW_RELATIVE_PATH):
                raise AssertionError("sealed or raw all-eight source access")
            return real_read(
                repository_root,
                relative_path,
                invariant=invariant,
            )

        with patch.object(corpus_module, "_read_bytes", side_effect=read):
            corpus = load_phase4_permitted_corpus(REPOSITORY_ROOT)

        self.assertEqual(len((*corpus.training_works, *corpus.validation_works)), 7)
        self.assertEqual(
            observed,
            [
                Path("docs/data/shakespeare-eight-play-manifest.json"),
                Path(
                    "data/processed/shakespeare-eight-play/processing-manifest.json"
                ),
                *(
                    Path(f"data/processed/shakespeare-eight-play/{split}/{work_id}.txt")
                    for _, work_id, split in EXPECTED_IDENTITIES
                ),
            ],
        )
        self.assertNotIn(SEALED_RELATIVE_PATH, observed)
        self.assertNotIn(RAW_RELATIVE_PATH, observed)

    def test_path_read_sentinel_fails_if_sealed_or_raw_source_is_opened(self) -> None:
        real_read_bytes = Path.read_bytes
        forbidden_attempts: list[Path] = []

        def guarded_read(path: Path):
            try:
                relative = path.relative_to(REPOSITORY_ROOT)
            except ValueError:
                relative = path
            if relative in (SEALED_RELATIVE_PATH, RAW_RELATIVE_PATH):
                forbidden_attempts.append(relative)
                raise AssertionError("forbidden source path opened")
            return real_read_bytes(path)

        with patch.object(Path, "read_bytes", new=guarded_read):
            load_phase4_permitted_corpus(REPOSITORY_ROOT)

        self.assertEqual(forbidden_attempts, [])

    def test_ancestor_symlink_is_rejected_before_redirected_read_or_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            repository_root = temporary_root / "repository"
            outside_root = temporary_root / "outside"
            outside_manifest = outside_root / "docs/data"
            repository_root.mkdir()
            outside_manifest.mkdir(parents=True)
            (outside_manifest / "shakespeare-eight-play-manifest.json").write_bytes(
                b"redirected content must not be read"
            )
            (repository_root / "docs").symlink_to(
                outside_root / "docs",
                target_is_directory=True,
            )
            read_calls: list[Path] = []
            result = None

            def forbidden_read(path: Path):
                read_calls.append(path)
                raise AssertionError("content read before ancestor-symlink rejection")

            with patch.object(Path, "read_bytes", new=forbidden_read):
                with patch.object(corpus_module, "_load_permitted_work") as load_work:
                    with self.assertRaises(Phase4GovernanceError) as raised:
                        result = load_phase4_permitted_corpus(repository_root)

        self.assertEqual(
            raised.exception.details["invariant"],
            "phase4_corpus.phase_1_manifest.symlink_component",
        )
        self.assertEqual(read_calls, [])
        load_work.assert_not_called()
        self.assertIsNone(result)

    def test_exactly_seven_permitted_works_are_constructed(self) -> None:
        constructed: list[str] = []
        real_load = corpus_module._load_permitted_work

        def load(repository_root, record):
            constructed.append(record.work_id)
            if record.work_id == "twelfth-night":
                raise AssertionError("sealed work construction")
            return real_load(repository_root, record)

        with patch.object(
            corpus_module,
            "_load_permitted_work",
            side_effect=load,
        ):
            load_phase4_permitted_corpus(REPOSITORY_ROOT)

        self.assertEqual(
            constructed,
            [work_id for _, work_id, _ in EXPECTED_IDENTITIES],
        )

    def test_published_text_and_metadata_match_phase_1_authority(self) -> None:
        corpus = load_phase4_permitted_corpus(REPOSITORY_ROOT)
        works = (*corpus.training_works, *corpus.validation_works)
        result_by_id = {
            result["work_id"]: result
            for result in self.phase_1_manifest["processing"]["per_work_results"]
        }
        work_by_id = {
            record["work_id"]: record for record in self.phase_1_manifest["works"]
        }
        excluded_by_id = {
            record["work_id"]: record
            for record in self.phase_1_manifest["processing"]["excluded_ranges"]
        }

        for work in works:
            with self.subTest(work_id=work.work_id):
                authority = result_by_id[work.work_id]
                identity = work_by_id[work.work_id]
                content = (
                    REPOSITORY_ROOT / identity["processed_path"]
                ).read_bytes()
                self.assertEqual(work.processed_text.encode("utf-8"), content)
                self.assertEqual(work.title, authority["title"])
                self.assertEqual(work.work_id, authority["work_id"])
                self.assertEqual(work.split, authority["split"])
                self.assertEqual(work.manifest_order, authority["manifest_order"])
                self.assertEqual(
                    work.processed.sha256,
                    hashlib.sha256(content).hexdigest(),
                )
                self.assertEqual(work.processed.byte_count, len(content))
                self.assertEqual(work.outer_raw.sha256, authority["outer_raw"]["sha256"])
                self.assertEqual(
                    work.outer_raw.byte_count,
                    authority["outer_raw"]["byte_count"],
                )
                self.assertEqual(
                    work.retained_raw.sha256,
                    authority["retained_raw"]["sha256"],
                )
                self.assertEqual(
                    work.retained_raw.byte_count,
                    authority["retained_raw"]["byte_count"],
                )
                self.assertEqual(
                    work.processed_code_point_count,
                    authority["processed"]["code_point_count"],
                )
                self.assertEqual(work.normalization, "crlf_to_lf_only")
                self.assertEqual(
                    work.processed_line_count,
                    authority["processed"]["line_count"],
                )
                self.assertEqual(
                    work.processed_word_count,
                    authority["processed"]["word_count"],
                )
                self.assertEqual(work.body_marker, identity["body_title_marker"])
                self.assertEqual(
                    work.successor_marker,
                    identity["successor_title_marker"],
                )
                self.assertEqual(
                    (
                        work.outer_range.start_position.line_number,
                        work.outer_range.start_position.byte_offset,
                        work.outer_range.start_position.code_point_offset,
                    ),
                    (
                        identity["observed_start_line"],
                        identity["observed_start_byte"],
                        identity["observed_start_code_point"],
                    ),
                )
                excluded = excluded_by_id[work.work_id]
                self.assertEqual(
                    work.removed_local_contents_byte_count,
                    excluded["byte_count"],
                )
                self.assertEqual(
                    (
                        work.local_contents_position.line_number,
                        work.local_contents_position.byte_offset,
                        work.local_contents_position.code_point_offset,
                    ),
                    (
                        excluded["start"]["line_number"],
                        excluded["start"]["byte_offset"],
                        excluded["start"]["code_point_offset"],
                    ),
                )
                self.assertEqual(
                    (
                        work.dramatis_position.line_number,
                        work.dramatis_position.byte_offset,
                        work.dramatis_position.code_point_offset,
                    ),
                    (
                        excluded["end_exclusive"]["line_number"],
                        excluded["end_exclusive"]["byte_offset"],
                        excluded["end_exclusive"]["code_point_offset"],
                    ),
                )
                expected_outer_end_byte = (
                    identity["observed_start_byte"]
                    + authority["outer_raw"]["byte_count"]
                )
                excluded_code_points = (
                    excluded["end_exclusive"]["code_point_offset"]
                    - excluded["start"]["code_point_offset"]
                )
                expected_outer_end_code_point = (
                    identity["observed_start_code_point"]
                    + authority["processed"]["code_point_count"]
                    + authority["processed"]["line_count"]
                    + excluded_code_points
                )
                self.assertEqual(
                    (
                        work.outer_range.end_byte_offset,
                        work.outer_range.end_code_point_offset,
                    ),
                    (expected_outer_end_byte, expected_outer_end_code_point),
                )
                separator_width = (
                    identity["expected_empty_crlf_lines_before_successor"] * 2
                )
                self.assertEqual(
                    (
                        work.successor_position.line_number,
                        work.successor_position.byte_offset,
                        work.successor_position.code_point_offset,
                    ),
                    (
                        identity["observed_successor_line"],
                        expected_outer_end_byte + separator_width,
                        expected_outer_end_code_point + separator_width,
                    ),
                )

    def test_factory_result_passes_runner_preflight_without_manual_assembly(self) -> None:
        corpus = load_phase4_permitted_corpus(REPOSITORY_ROOT)
        vocabulary = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
        state = experiment_module._RepositoryState("a" * 40, True)
        with patch.object(
            experiment_module,
            "_read_repository_state",
            return_value=state,
        ):
            config = load_phase4_experiment_config()
            validate_phase4_experiment_preflight(config, vocabulary, corpus)

        self.assertEqual(
            tuple(inspect.signature(load_phase4_permitted_corpus).parameters),
            ("repository_root",),
        )

    def test_wrong_repository_root_type_is_rejected(self) -> None:
        with self.assertRaises(Phase4TypeError) as raised:
            load_phase4_permitted_corpus(".")  # type: ignore[arg-type]

        self.assertEqual(
            raised.exception.details["invariant"],
            "phase4_corpus.repository_root.path",
        )

    def test_phase_1_manifest_hash_failure_precedes_other_reads(self) -> None:
        with patch.object(
            corpus_module,
            "_read_bytes",
            return_value=b"not the accepted manifest",
        ) as read:
            with self.assertRaises(Phase4GovernanceError) as raised:
                load_phase4_permitted_corpus(REPOSITORY_ROOT)

        self.assertEqual(
            raised.exception.details["invariant"],
            "phase4_corpus.phase_1_manifest.sha256",
        )
        self.assertEqual(read.call_count, 1)

    def test_processing_manifest_hash_failure_precedes_work_reads(self) -> None:
        phase_1_bytes = (
            REPOSITORY_ROOT / "docs/data/shakespeare-eight-play-manifest.json"
        ).read_bytes()
        with patch.object(
            corpus_module,
            "_read_bytes",
            side_effect=(phase_1_bytes, b"wrong processing manifest"),
        ) as read:
            with self.assertRaises(Phase4GovernanceError) as raised:
                load_phase4_permitted_corpus(REPOSITORY_ROOT)

        self.assertEqual(
            raised.exception.details["invariant"],
            "phase4_corpus.processing_manifest.sha256",
        )
        self.assertEqual(read.call_count, 2)

    def test_identity_failures_precede_any_permitted_content_load(self) -> None:
        mutations = {}

        missing = copy.deepcopy(self.phase_1_manifest)
        missing["works"].pop(1)
        mutations["missing"] = missing

        duplicate = copy.deepcopy(self.phase_1_manifest)
        duplicate["works"][1] = copy.deepcopy(duplicate["works"][0])
        mutations["duplicate"] = duplicate

        reordered = copy.deepcopy(self.phase_1_manifest)
        reordered["works"][0], reordered["works"][1] = (
            reordered["works"][1],
            reordered["works"][0],
        )
        mutations["reordered"] = reordered

        wrong_split = copy.deepcopy(self.phase_1_manifest)
        wrong_split["works"][0]["split"] = "validation"
        mutations["wrong_split"] = wrong_split

        sealed_substitution = copy.deepcopy(self.phase_1_manifest)
        sealed_substitution["works"][6] = copy.deepcopy(
            sealed_substitution["works"][7]
        )
        mutations["sealed_substitution"] = sealed_substitution

        for name, manifest in mutations.items():
            with self.subTest(name=name):
                with patch.object(
                    corpus_module,
                    "_read_verified_json",
                    side_effect=(manifest, self.processing_manifest),
                ):
                    with patch.object(
                        corpus_module,
                        "_load_permitted_work",
                    ) as load_work:
                        with self.assertRaises(Phase4GovernanceError):
                            load_phase4_permitted_corpus(REPOSITORY_ROOT)
                load_work.assert_not_called()

    def test_processing_identity_failure_precedes_content_load(self) -> None:
        processing = copy.deepcopy(self.processing_manifest)
        processing["works"][0]["work_id"] = "unexpected"
        with patch.object(
            corpus_module,
            "_read_verified_json",
            side_effect=(self.phase_1_manifest, processing),
        ):
            with patch.object(corpus_module, "_load_permitted_work") as load_work:
                with self.assertRaises(Phase4GovernanceError):
                    load_phase4_permitted_corpus(REPOSITORY_ROOT)

        load_work.assert_not_called()

    def test_source_contains_no_all_eight_extractor_or_split_option(self) -> None:
        source = inspect.getsource(corpus_module)
        signature = inspect.signature(load_phase4_permitted_corpus)

        self.assertNotIn("extract_shakespeare_in_memory", source)
        self.assertNotIn("load_validated_source", source)
        self.assertEqual(tuple(signature.parameters), ("repository_root",))
        self.assertNotIn("split", signature.parameters)


if __name__ == "__main__":
    unittest.main()
