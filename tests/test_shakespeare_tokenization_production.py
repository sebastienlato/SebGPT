"""Production vocabulary and permitted-statistics checkpoint tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sebgpt.tokenization.code_point import CodePointTokenizer  # noqa: E402
from sebgpt.tokenization.shakespeare import (  # noqa: E402
    EXPECTED_NORMALIZATION as PRODUCER_PHASE_1_NORMALIZATION,
    build_shakespeare_training_vocabulary,
)
from sebgpt.tokenization.shakespeare_production import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION as PRODUCER_ARTIFACT_SCHEMA_VERSION,
    CONTRACT_DECISION as PRODUCER_CONTRACT_DECISION,
    EXPECTED_IMPLEMENTATION_COMMIT as PRODUCER_IMPLEMENTATION_COMMIT,
    ID_ORDER as PRODUCER_ID_ORDER,
    NORMALIZATION as PRODUCER_NORMALIZATION,
    SPECIAL_TOKEN_POLICY as PRODUCER_SPECIAL_TOKEN_POLICY,
    TOKENIZER_ID as PRODUCER_TOKENIZER_ID,
    TOKEN_UNIT as PRODUCER_TOKEN_UNIT,
    UNKNOWN_POLICY as PRODUCER_UNKNOWN_POLICY,
    VOCABULARY_RELATIVE_PATH as PRODUCER_VOCABULARY_PATH,
    ProductionTokenizationError,
    build_vocabulary_artifact,
    extract_permitted_phase_2_works,
    inspect_training_statistics,
    inspect_validation_statistics,
    verify_vocabulary_artifact,
    write_vocabulary_artifact,
)


EXPECTED_ARTIFACT_SHA256 = (
    "9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e"
)
EXPECTED_PHASE_1_MANIFEST_SHA256 = (
    "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"
)
TEST_EXPECTED_IMPLEMENTATION_COMMIT = (
    "de7a7f096fbbd8607c944412eaef30be9b686b56"
)
TEST_EXPECTED_PHASE_1_NORMALIZATION = "crlf_to_lf_only"
TEST_EXPECTED_ARTIFACT_PATH = Path(
    "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json"
)
TEST_EXPECTED_TOKENIZER_ID = "shakespeare-code-point-v1"
TEST_EXPECTED_TOKEN_UNIT = "python_str_code_point"
TEST_EXPECTED_NORMALIZATION = "none"
TEST_EXPECTED_UNKNOWN_POLICY = "error"
TEST_EXPECTED_SPECIAL_TOKEN_POLICY = "none"
TEST_EXPECTED_ID_ORDER = "ascending_unicode_code_point"
TEST_EXPECTED_SCHEMA_VERSION = 1
TEST_EXPECTED_CONTRACT_DECISION = "DEC-0014"
EXPECTED_TRAINING = (
    (
        1,
        "hamlet",
        "281070a10b0fe06b877a6e9342cedecad1e5f65657d58fc51b14ce378c70a243",
        177157,
    ),
    (
        2,
        "romeo-and-juliet",
        "59bfe35a3f7ccb63353aa4b8823f1a33fca4bd27c47943515ecc5fb8d6775775",
        141528,
    ),
    (
        3,
        "macbeth",
        "fb24ddc7c0c35f7e7d6e0989858e3d9cdb30d00208f2122938d7cd900a3699c6",
        103253,
    ),
    (
        4,
        "a-midsummer-nights-dream",
        "a9314798205c72c7cb7988813eb9df3d36e9731e45d882f761c907f31e178bd1",
        96353,
    ),
    (
        5,
        "much-ado-about-nothing",
        "fbb46df4f5297329599a4e58b160d7182b30262b1c497c0de4002e27f536ce7b",
        122103,
    ),
    (
        6,
        "henry-v",
        "eb8c7e711ac4182800d5b6280d1cf5c0c1df98f43c727d48cd04ca8288321560",
        152311,
    ),
)
EXPECTED_ROOT_KEYS = {
    "schema_version",
    "tokenizer_id",
    "token_unit",
    "normalization",
    "unknown_policy",
    "special_token_policy",
    "id_order",
    "contract_decision",
    "phase_1_manifest_sha256",
    "implementation_git_commit",
    "training_works",
    "vocabulary_size",
    "vocabulary",
}


class SealedTestGuard:
    split = "test"

    def __init__(self) -> None:
        self.text_access_count = 0

    @property
    def processed_text(self) -> str:
        self.text_access_count += 1
        raise AssertionError("sealed test text must not be accessed")


class ProductionVocabularyCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        all_works = extract_permitted_phase_2_works()
        assert all(work.split != "test" for work in all_works)
        cls.training_works = tuple(
            work for work in all_works if work.split == "train"
        )
        validation_works = tuple(
            work for work in all_works if work.split == "validation"
        )
        cls.validation_work = validation_works[0]
        cls.manifest_bytes = (
            REPOSITORY_ROOT / "docs/data/shakespeare-eight-play-manifest.json"
        ).read_bytes()
        cls.manifest = json.loads(cls.manifest_bytes.decode("utf-8"))
        membership: set[int] = set()
        for work in cls.training_works:
            for character in work.processed_text:
                membership.add(ord(character))
        cls.independent_vocabulary = tuple(sorted(membership))

        cls.production_vocabulary = build_shakespeare_training_vocabulary(
            cls.training_works,
            cls.manifest,
        )
        cls.tokenizer = CodePointTokenizer(cls.production_vocabulary)
        cls.artifact_build = build_vocabulary_artifact(
            cls.training_works,
            cls.manifest_bytes,
        )
        cls.training_statistics = inspect_training_statistics(
            cls.training_works,
            cls.manifest,
            cls.tokenizer,
        )
        cls.validation_statistics = inspect_validation_statistics(
            cls.validation_work,
            cls.manifest,
            cls.tokenizer,
        )

    def test_production_vocabulary_and_artifact_are_deterministic(self) -> None:
        repeated = build_vocabulary_artifact(
            self.training_works,
            self.manifest_bytes,
        )

        self.assertEqual(len(self.independent_vocabulary), 81)
        self.assertEqual(
            self.independent_vocabulary,
            tuple(sorted(set(self.independent_vocabulary))),
        )
        self.assertEqual(
            self.production_vocabulary,
            self.independent_vocabulary,
        )
        self.assertEqual(self.tokenizer.vocabulary_size, 81)
        self.assertEqual(self.artifact_build.vocabulary_size, 81)
        self.assertEqual(self.artifact_build.sha256, EXPECTED_ARTIFACT_SHA256)
        self.assertEqual(repeated.content, self.artifact_build.content)
        self.assertEqual(repeated.sha256, self.artifact_build.sha256)

    def test_canonical_artifact_schema_and_provenance_are_exact(self) -> None:
        self.assertEqual(
            PRODUCER_IMPLEMENTATION_COMMIT,
            TEST_EXPECTED_IMPLEMENTATION_COMMIT,
        )
        self.assertEqual(
            PRODUCER_PHASE_1_NORMALIZATION,
            TEST_EXPECTED_PHASE_1_NORMALIZATION,
        )
        self.assertEqual(PRODUCER_VOCABULARY_PATH, TEST_EXPECTED_ARTIFACT_PATH)
        self.assertEqual(PRODUCER_TOKENIZER_ID, TEST_EXPECTED_TOKENIZER_ID)
        self.assertEqual(PRODUCER_TOKEN_UNIT, TEST_EXPECTED_TOKEN_UNIT)
        self.assertEqual(PRODUCER_NORMALIZATION, TEST_EXPECTED_NORMALIZATION)
        self.assertEqual(PRODUCER_UNKNOWN_POLICY, TEST_EXPECTED_UNKNOWN_POLICY)
        self.assertEqual(
            PRODUCER_SPECIAL_TOKEN_POLICY,
            TEST_EXPECTED_SPECIAL_TOKEN_POLICY,
        )
        self.assertEqual(PRODUCER_ID_ORDER, TEST_EXPECTED_ID_ORDER)
        self.assertEqual(
            PRODUCER_ARTIFACT_SCHEMA_VERSION,
            TEST_EXPECTED_SCHEMA_VERSION,
        )
        self.assertEqual(
            PRODUCER_CONTRACT_DECISION,
            TEST_EXPECTED_CONTRACT_DECISION,
        )

        path = REPOSITORY_ROOT / TEST_EXPECTED_ARTIFACT_PATH
        content = path.read_bytes()
        artifact = json.loads(content.decode("utf-8"))

        verify_vocabulary_artifact(
            content,
            self.training_works,
            self.manifest_bytes,
        )
        self.assertEqual(content, self.artifact_build.content)
        self.assertEqual(set(artifact), EXPECTED_ROOT_KEYS)
        self.assertEqual(artifact["schema_version"], TEST_EXPECTED_SCHEMA_VERSION)
        self.assertEqual(artifact["tokenizer_id"], TEST_EXPECTED_TOKENIZER_ID)
        self.assertEqual(artifact["token_unit"], TEST_EXPECTED_TOKEN_UNIT)
        self.assertEqual(artifact["normalization"], TEST_EXPECTED_NORMALIZATION)
        self.assertEqual(artifact["unknown_policy"], TEST_EXPECTED_UNKNOWN_POLICY)
        self.assertEqual(
            artifact["special_token_policy"],
            TEST_EXPECTED_SPECIAL_TOKEN_POLICY,
        )
        self.assertEqual(artifact["id_order"], TEST_EXPECTED_ID_ORDER)
        self.assertEqual(
            artifact["contract_decision"],
            TEST_EXPECTED_CONTRACT_DECISION,
        )
        self.assertEqual(
            artifact["implementation_git_commit"],
            TEST_EXPECTED_IMPLEMENTATION_COMMIT,
        )
        self.assertEqual(
            artifact["phase_1_manifest_sha256"],
            EXPECTED_PHASE_1_MANIFEST_SHA256,
        )
        self.assertEqual(self.artifact_build.sha256, EXPECTED_ARTIFACT_SHA256)

    def test_six_training_records_match_independent_expectations(self) -> None:
        artifact = json.loads(self.artifact_build.content.decode("utf-8"))
        records = artifact["training_works"]

        observed = tuple(
            (
                record["manifest_order"],
                record["work_id"],
                record["processed_sha256"],
            )
            for record in records
        )
        expected = tuple(
            (order, work_id, sha256)
            for order, work_id, sha256, _ in EXPECTED_TRAINING
        )
        self.assertEqual(observed, expected)
        self.assertEqual(len({record["work_id"] for record in records}), 6)
        self.assertEqual(len({record["manifest_order"] for record in records}), 6)

    def test_vocabulary_records_are_complete_ordered_and_literal_free(self) -> None:
        artifact = json.loads(self.artifact_build.content.decode("utf-8"))
        records = artifact["vocabulary"]

        self.assertEqual(artifact["vocabulary_size"], 81)
        self.assertEqual(len(records), 81)
        self.assertEqual(
            tuple(record["token_id"] for record in records),
            tuple(range(81)),
        )
        code_points = tuple(record["code_point"] for record in records)
        self.assertEqual(code_points, self.independent_vocabulary)
        self.assertEqual(self.tokenizer.code_points, self.independent_vocabulary)
        self.assertEqual(
            tuple(
                self.independent_vocabulary[record["token_id"]]
                for record in records
            ),
            self.independent_vocabulary,
        )
        self.assertTrue(all(0 <= code_point <= 0x10FFFF for code_point in code_points))
        self.assertTrue(
            all(
                record["code_point_uplus"] == f"U+{record['code_point']:04X}"
                for record in records
            )
        )
        all_keys = set().union(*(record.keys() for record in records))
        self.assertNotIn("character", all_keys)
        self.assertNotIn("literal_character", all_keys)
        self.assertNotIn("name", all_keys)
        self.assertNotIn("unicode_name", all_keys)

    def test_training_statistics_match_phase_1_counts(self) -> None:
        statistics = self.training_statistics
        observed = tuple(
            (work.work_id, work.token_count)
            for work in statistics.works
        )
        expected = tuple(
            (work_id, code_point_count)
            for _, work_id, _, code_point_count in EXPECTED_TRAINING
        )

        self.assertEqual(observed, expected)
        self.assertEqual(statistics.total_token_count, 792705)
        self.assertEqual(statistics.coverage, 1.0)
        self.assertEqual(statistics.unknown_count, 0)
        self.assertTrue(all(work.coverage == 1.0 for work in statistics.works))
        self.assertTrue(all(work.unknown_count == 0 for work in statistics.works))
        self.assertTrue(
            all(work.round_trip_success for work in statistics.works)
        )

    def test_training_frequency_table_is_complete_and_sums_exactly(self) -> None:
        frequencies = self.training_statistics.frequencies
        independent_ids = {
            code_point: token_id
            for token_id, code_point in enumerate(self.independent_vocabulary)
        }
        independent_counts: Counter[int] = Counter()
        for work in self.training_works:
            for character in work.processed_text:
                independent_counts[independent_ids[ord(character)]] += 1
        expected = tuple(
            (
                token_id,
                code_point,
                independent_counts[token_id],
            )
            for token_id, code_point in enumerate(self.independent_vocabulary)
        )
        observed = tuple(
            (entry.token_id, entry.code_point, entry.count)
            for entry in frequencies
        )

        self.assertEqual(len(frequencies), 81)
        self.assertEqual(observed, expected)
        self.assertEqual(tuple(entry.token_id for entry in frequencies), tuple(range(81)))
        self.assertEqual(
            tuple(entry.code_point for entry in frequencies),
            self.independent_vocabulary,
        )
        self.assertTrue(all(entry.count > 0 for entry in frequencies))
        self.assertEqual(
            sum(entry.count for entry in frequencies),
            self.training_statistics.total_token_count,
        )

    def test_validation_statistics_are_aggregate_and_do_not_change_vocabulary(self) -> None:
        before = self.tokenizer.code_points
        statistics = self.validation_statistics

        self.assertEqual(statistics.work_count, 1)
        self.assertEqual(statistics.token_count, 98296)
        self.assertEqual(statistics.coverage, 1.0)
        self.assertEqual(statistics.unknown_count, 0)
        self.assertTrue(statistics.round_trip_success)
        self.assertEqual(self.tokenizer.code_points, before)

    def test_statistics_reject_test_before_text_access(self) -> None:
        guarded_test = SealedTestGuard()

        with self.assertRaises(ProductionTokenizationError) as raised:
            inspect_validation_statistics(
                guarded_test,  # type: ignore[arg-type]
                self.manifest,
                self.tokenizer,
            )

        self.assertEqual(raised.exception.details["invariant"], "validation.work.split")
        self.assertEqual(guarded_test.text_access_count, 0)

    def test_only_canonical_artifact_is_persisted(self) -> None:
        artifact_root = REPOSITORY_ROOT / TEST_EXPECTED_ARTIFACT_PATH.parents[1]
        artifact_files = tuple(
            sorted(
                path.relative_to(REPOSITORY_ROOT).as_posix()
                for path in artifact_root.rglob("*")
                if path.is_file()
            )
        )
        self.assertEqual(
            artifact_files,
            (TEST_EXPECTED_ARTIFACT_PATH.as_posix(),),
        )

        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            first = write_vocabulary_artifact(
                self.artifact_build,
                temporary_root,
            )
            second = write_vocabulary_artifact(
                self.artifact_build,
                temporary_root,
            )
            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), self.artifact_build.content)
