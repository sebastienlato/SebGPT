"""Tests for the corpus-free accepted-vocabulary runtime boundary."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import sebgpt.tokenization.vocabulary_artifact as artifact_module  # noqa: E402
from sebgpt.model.embeddings import (  # noqa: E402
    EmbeddingContractError,
    TokenPositionEmbedding,
)
from sebgpt.tokenization.vocabulary_artifact import (  # noqa: E402
    VocabularyArtifactError,
    VocabularyBinding,
    load_accepted_vocabulary_binding,
)


TEST_ARTIFACT_PATH = Path(
    "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json"
)
TEST_ARTIFACT_SHA256 = (
    "9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e"
)
TEST_CODE_POINTS = (
    10,
    32,
    33,
    38,
    40,
    41,
    44,
    45,
    46,
    58,
    59,
    63,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    93,
    95,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    201,
    224,
    226,
    230,
    231,
    232,
    233,
    234,
    238,
    8212,
    8216,
    8217,
    8220,
    8221,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class VocabularyBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.accepted_content = (REPOSITORY_ROOT / TEST_ARTIFACT_PATH).read_bytes()
        self.accepted_object = json.loads(self.accepted_content.decode("utf-8"))

    @contextmanager
    def _temporary_root(self, content: bytes | None, *, directory=False):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / TEST_ARTIFACT_PATH
            path.parent.mkdir(parents=True)
            if directory:
                path.mkdir()
            elif content is not None:
                path.write_bytes(content)
            yield root

    def _load_with_test_hash(self, root: Path, content: bytes):
        derived_hash = hashlib.sha256(content).hexdigest()
        with patch.object(
            artifact_module,
            "EXPECTED_ARTIFACT_SHA256",
            derived_hash,
        ):
            return load_accepted_vocabulary_binding(root)

    def _assert_artifact_failure(self, artifact: object, invariant: str) -> None:
        content = _canonical_bytes(artifact)
        with self._temporary_root(content) as root:
            with self.assertRaises(VocabularyArtifactError) as raised:
                self._load_with_test_hash(root, content)
        self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_01_canonical_artifact_loads_exact_identity_and_code_points(self) -> None:
        binding = load_accepted_vocabulary_binding(REPOSITORY_ROOT)

        self.assertEqual(binding.tokenizer_id, "shakespeare-code-point-v1")
        self.assertEqual(binding.schema_version, 1)
        self.assertEqual(binding.vocabulary_size, 81)
        self.assertEqual(binding.artifact_sha256, TEST_ARTIFACT_SHA256)
        self.assertEqual(binding.code_points, TEST_CODE_POINTS)
        self.assertEqual(hashlib.sha256(self.accepted_content).hexdigest(), TEST_ARTIFACT_SHA256)

    def test_02_missing_and_non_regular_canonical_paths_fail_safely(self) -> None:
        with self._temporary_root(None) as missing_root:
            with self.assertRaises(VocabularyArtifactError) as missing:
                load_accepted_vocabulary_binding(missing_root)
        with self._temporary_root(None, directory=True) as directory_root:
            with self.assertRaises(VocabularyArtifactError) as directory:
                load_accepted_vocabulary_binding(directory_root)

        self.assertEqual(missing.exception.details["invariant"], "vocabulary.path.regular_file")
        self.assertEqual(directory.exception.details["invariant"], "vocabulary.path.regular_file")

    def test_03_invalid_utf8_and_non_object_json_fail_safely(self) -> None:
        cases = ((b"\xff", "vocabulary.utf8"), (b"[]\n", "vocabulary.json_object"))
        for content, invariant in cases:
            with self.subTest(invariant=invariant):
                with self._temporary_root(content) as root:
                    with self.assertRaises(VocabularyArtifactError) as raised:
                        self._load_with_test_hash(root, content)
                self.assertEqual(raised.exception.details["invariant"], invariant)

    def test_04_wrong_identity_schema_and_size_fail_before_binding(self) -> None:
        cases = (
            ("tokenizer_id", "other", "vocabulary.tokenizer_id"),
            ("schema_version", True, "vocabulary.schema_version"),
            ("vocabulary_size", 80, "vocabulary.size"),
        )
        for field, value, invariant in cases:
            with self.subTest(field=field):
                artifact = dict(self.accepted_object)
                artifact[field] = value
                self._assert_artifact_failure(artifact, invariant)

        mutations = []
        extra_root = json.loads(self.accepted_content.decode("utf-8"))
        extra_root["extra"] = None
        mutations.append((extra_root, "vocabulary.root_keys"))
        semantic = json.loads(self.accepted_content.decode("utf-8"))
        semantic["token_unit"] = "other"
        mutations.append((semantic, "vocabulary.semantic_fields"))
        provenance = json.loads(self.accepted_content.decode("utf-8"))
        provenance["implementation_git_commit"] = "0" * 40
        mutations.append((provenance, "vocabulary.provenance"))
        training = json.loads(self.accepted_content.decode("utf-8"))
        training["training_works"][0]["manifest_order"] = True
        mutations.append((training, "vocabulary.training_works"))
        token_id = json.loads(self.accepted_content.decode("utf-8"))
        token_id["vocabulary"][0]["token_id"] = True
        mutations.append((token_id, "vocabulary.records"))
        code_point = json.loads(self.accepted_content.decode("utf-8"))
        code_point["vocabulary"][0]["code_point"] = True
        mutations.append((code_point, "vocabulary.records"))
        notation = json.loads(self.accepted_content.decode("utf-8"))
        notation["vocabulary"][0]["code_point_uplus"] = "U+000B"
        mutations.append((notation, "vocabulary.code_point_uplus"))

        for artifact, invariant in mutations:
            with self.subTest(invariant=invariant):
                self._assert_artifact_failure(artifact, invariant)

    def test_05_same_size_different_mapping_is_rejected_by_pinned_hash(self) -> None:
        artifact = json.loads(self.accepted_content.decode("utf-8"))
        artifact["vocabulary"][1]["code_point"] = 31
        artifact["vocabulary"][1]["code_point_uplus"] = "U+001F"
        content = _canonical_bytes(artifact)

        with self._temporary_root(content) as root:
            with self.assertRaises(VocabularyArtifactError) as raised:
                load_accepted_vocabulary_binding(root)

        self.assertEqual(raised.exception.details["invariant"], "vocabulary.sha256")

    def test_06_otherwise_valid_different_bytes_are_rejected_by_hash(self) -> None:
        content = self.accepted_content + b"\n"

        with self._temporary_root(content) as root:
            with self.assertRaises(VocabularyArtifactError) as raised:
                load_accepted_vocabulary_binding(root)

        self.assertEqual(raised.exception.details["invariant"], "vocabulary.sha256")
        with self._temporary_root(content) as root:
            with self.assertRaises(VocabularyArtifactError) as canonical:
                self._load_with_test_hash(root, content)
        self.assertEqual(
            canonical.exception.details["invariant"],
            "vocabulary.canonical_serialization",
        )

    def test_07_errors_are_safe_and_binding_crosses_constructor_once(self) -> None:
        secret_root_fragment = "SECRET-ROOT"
        with tempfile.TemporaryDirectory(prefix=secret_root_fragment) as directory:
            root = Path(directory)
            with self.assertRaises(VocabularyArtifactError) as missing:
                load_accepted_vocabulary_binding(root)
        rendered = str(missing.exception) + repr(missing.exception)
        self.assertNotIn(secret_root_fragment, rendered)
        self.assertEqual(
            missing.exception.details["path"],
            TEST_ARTIFACT_PATH.as_posix(),
        )
        with self.assertRaises(TypeError):
            missing.exception.details["new"] = "value"  # type: ignore[index]
        with self.assertRaises(VocabularyArtifactError) as direct:
            VocabularyBinding()
        self.assertEqual(direct.exception.details["invariant"], "vocabulary.binding.factory")
        forged = object.__new__(VocabularyBinding)
        with self.assertRaises(EmbeddingContractError) as forged_error:
            TokenPositionEmbedding(
                forged,
                embedding_dim=32,
                max_positions=256,
                seed=1337,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        self.assertEqual(forged_error.exception.details["invariant"], "vocabulary.binding")

        binding = load_accepted_vocabulary_binding(REPOSITORY_ROOT)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no artifact I/O")):
            module = TokenPositionEmbedding(
                binding,
                embedding_dim=32,
                max_positions=256,
                seed=1337,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            output = module(torch.tensor([0], dtype=torch.long))
        self.assertEqual(tuple(output.shape), (1, 32))
