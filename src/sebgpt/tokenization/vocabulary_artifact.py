"""Corpus-free loading of the accepted Shakespeare vocabulary artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from sebgpt.tokenization.code_point import CodePointTokenizer, code_point_uplus


__all__ = [
    "VocabularyArtifactError",
    "VocabularyBinding",
    "load_accepted_vocabulary_binding",
]


VOCABULARY_RELATIVE_PATH = Path(
    "artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json"
)
EXPECTED_ARTIFACT_SHA256 = (
    "9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e"
)
EXPECTED_TOKENIZER_ID = "shakespeare-code-point-v1"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_VOCABULARY_SIZE = 81
EXPECTED_PHASE_1_MANIFEST_SHA256 = (
    "157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb"
)
EXPECTED_IMPLEMENTATION_COMMIT = "de7a7f096fbbd8607c944412eaef30be9b686b56"

_ROOT_KEYS = frozenset(
    {
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
)
_TRAINING_WORK_KEYS = frozenset(
    {"work_id", "manifest_order", "processed_sha256"}
)
_VOCABULARY_KEYS = frozenset(
    {"token_id", "code_point", "code_point_uplus"}
)
_EXPECTED_TRAINING_WORKS = (
    (
        1,
        "hamlet",
        "281070a10b0fe06b877a6e9342cedecad1e5f65657d58fc51b14ce378c70a243",
    ),
    (
        2,
        "romeo-and-juliet",
        "59bfe35a3f7ccb63353aa4b8823f1a33fca4bd27c47943515ecc5fb8d6775775",
    ),
    (
        3,
        "macbeth",
        "fb24ddc7c0c35f7e7d6e0989858e3d9cdb30d00208f2122938d7cd900a3699c6",
    ),
    (
        4,
        "a-midsummer-nights-dream",
        "a9314798205c72c7cb7988813eb9df3d36e9731e45d882f761c907f31e178bd1",
    ),
    (
        5,
        "much-ado-about-nothing",
        "fbb46df4f5297329599a4e58b160d7182b30262b1c497c0de4002e27f536ce7b",
    ),
    (
        6,
        "henry-v",
        "eb8c7e711ac4182800d5b6280d1cf5c0c1df98f43c727d48cd04ca8288321560",
    ),
)
_BINDING_FACTORY_MARKER = object()


class VocabularyArtifactError(ValueError):
    """A deterministic vocabulary failure containing only safe facts."""

    __slots__ = ("_details",)

    def __init__(self, invariant: str, **safe_facts: Any) -> None:
        details = {"invariant": invariant, **safe_facts}
        self._details = MappingProxyType(details)
        super().__init__(json.dumps(details, ensure_ascii=True, sort_keys=True))

    @property
    def details(self) -> Mapping[str, Any]:
        return self._details


@dataclass(frozen=True, init=False)
class VocabularyBinding:
    """Immutable vocabulary identity produced only by the accepted loader."""

    tokenizer_id: str
    schema_version: int
    vocabulary_size: int
    artifact_sha256: str
    code_points: tuple[int, ...] = field(repr=False)
    _factory_marker: object = field(repr=False, compare=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> VocabularyBinding:
        raise VocabularyArtifactError("vocabulary.binding.factory")


def _require(condition: bool, invariant: str, **safe_facts: Any) -> None:
    if not condition:
        raise VocabularyArtifactError(invariant, **safe_facts)


def _is_exact_int(value: Any) -> bool:
    return type(value) is int


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise VocabularyArtifactError("vocabulary.canonical_serialization") from None
    return (serialized + "\n").encode("utf-8")


def _load_json_object(content: bytes) -> dict[str, Any]:
    _require(not content.startswith(b"\xef\xbb\xbf"), "vocabulary.utf8")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise VocabularyArtifactError("vocabulary.utf8") from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise VocabularyArtifactError("vocabulary.json_object") from None
    _require(type(value) is dict, "vocabulary.json_object")
    return value


def _verify_fixed_root_fields(artifact: Mapping[str, Any]) -> None:
    _require(frozenset(artifact) == _ROOT_KEYS, "vocabulary.root_keys")
    _require(
        _is_exact_int(artifact.get("schema_version"))
        and artifact.get("schema_version") == EXPECTED_SCHEMA_VERSION,
        "vocabulary.schema_version",
        expected=EXPECTED_SCHEMA_VERSION,
    )
    fixed_strings = {
        "tokenizer_id": EXPECTED_TOKENIZER_ID,
        "token_unit": "python_str_code_point",
        "normalization": "none",
        "unknown_policy": "error",
        "special_token_policy": "none",
        "id_order": "ascending_unicode_code_point",
        "contract_decision": "DEC-0014",
    }
    for name, expected in fixed_strings.items():
        _require(
            type(artifact.get(name)) is str and artifact.get(name) == expected,
            "vocabulary.tokenizer_id"
            if name == "tokenizer_id"
            else "vocabulary.semantic_fields",
            field=name,
            expected=expected,
        )
    phase_1_sha256 = artifact.get("phase_1_manifest_sha256")
    _require(
        _is_sha256(phase_1_sha256)
        and phase_1_sha256 == EXPECTED_PHASE_1_MANIFEST_SHA256,
        "vocabulary.provenance",
        field="phase_1_manifest_sha256",
        expected=EXPECTED_PHASE_1_MANIFEST_SHA256,
    )
    implementation_commit = artifact.get("implementation_git_commit")
    _require(
        _is_git_commit(implementation_commit)
        and implementation_commit == EXPECTED_IMPLEMENTATION_COMMIT,
        "vocabulary.provenance",
        field="implementation_git_commit",
        expected=EXPECTED_IMPLEMENTATION_COMMIT,
    )


def _verify_training_works(artifact: Mapping[str, Any]) -> None:
    records = artifact.get("training_works")
    _require(type(records) is list, "vocabulary.training_works")
    _require(
        len(records) == len(_EXPECTED_TRAINING_WORKS),
        "vocabulary.training_works",
        expected_count=len(_EXPECTED_TRAINING_WORKS),
    )
    for position, (record, expected) in enumerate(
        zip(records, _EXPECTED_TRAINING_WORKS, strict=True)
    ):
        _require(type(record) is dict, "vocabulary.training_works", position=position)
        _require(
            frozenset(record) == _TRAINING_WORK_KEYS,
            "vocabulary.training_works",
            position=position,
        )
        order, work_id, processed_sha256 = expected
        _require(
            _is_exact_int(record.get("manifest_order"))
            and record.get("manifest_order") == order,
            "vocabulary.training_works",
            position=position,
            field="manifest_order",
        )
        _require(
            type(record.get("work_id")) is str
            and record.get("work_id") == work_id,
            "vocabulary.training_works",
            position=position,
            field="work_id",
        )
        observed_sha256 = record.get("processed_sha256")
        _require(
            _is_sha256(observed_sha256) and observed_sha256 == processed_sha256,
            "vocabulary.training_works",
            position=position,
            field="processed_sha256",
        )


def _verify_vocabulary_records(artifact: Mapping[str, Any]) -> tuple[int, ...]:
    vocabulary_size = artifact.get("vocabulary_size")
    records = artifact.get("vocabulary")
    _require(
        _is_exact_int(vocabulary_size)
        and vocabulary_size == EXPECTED_VOCABULARY_SIZE,
        "vocabulary.size",
        expected=EXPECTED_VOCABULARY_SIZE,
    )
    _require(type(records) is list, "vocabulary.records")
    _require(
        len(records) == EXPECTED_VOCABULARY_SIZE,
        "vocabulary.records",
        expected_count=EXPECTED_VOCABULARY_SIZE,
    )

    code_points: list[int] = []
    for position, record in enumerate(records):
        _require(type(record) is dict, "vocabulary.records", position=position)
        _require(
            frozenset(record) == _VOCABULARY_KEYS,
            "vocabulary.records",
            position=position,
        )
        token_id = record.get("token_id")
        code_point = record.get("code_point")
        notation = record.get("code_point_uplus")
        _require(
            _is_exact_int(token_id) and token_id == position,
            "vocabulary.records",
            position=position,
            field="token_id",
        )
        _require(
            _is_exact_int(code_point) and 0 <= code_point <= 0x10FFFF,
            "vocabulary.records",
            position=position,
            field="code_point",
        )
        _require(
            not code_points or code_point > code_points[-1],
            "vocabulary.records",
            position=position,
            field="code_point",
        )
        _require(
            type(notation) is str and notation == code_point_uplus(code_point),
            "vocabulary.code_point_uplus",
            position=position,
        )
        code_points.append(code_point)

    canonical_state = tuple(code_points)
    CodePointTokenizer(canonical_state)
    return canonical_state


def _create_binding(code_points: tuple[int, ...]) -> VocabularyBinding:
    binding = object.__new__(VocabularyBinding)
    object.__setattr__(binding, "tokenizer_id", EXPECTED_TOKENIZER_ID)
    object.__setattr__(binding, "schema_version", EXPECTED_SCHEMA_VERSION)
    object.__setattr__(binding, "vocabulary_size", EXPECTED_VOCABULARY_SIZE)
    object.__setattr__(binding, "artifact_sha256", EXPECTED_ARTIFACT_SHA256)
    object.__setattr__(binding, "code_points", code_points)
    object.__setattr__(binding, "_factory_marker", _BINDING_FACTORY_MARKER)
    return binding


def _is_verified_vocabulary_binding(value: Any) -> bool:
    """Return whether value was produced by this module's accepted factory."""

    return (
        type(value) is VocabularyBinding
        and getattr(value, "_factory_marker", None) is _BINDING_FACTORY_MARKER
    )


def load_accepted_vocabulary_binding(repository_root: Path) -> VocabularyBinding:
    """Load and completely verify the accepted artifact without corpus access."""

    if not isinstance(repository_root, Path):
        raise VocabularyArtifactError("vocabulary.path.regular_file")
    path = repository_root / VOCABULARY_RELATIVE_PATH
    try:
        is_regular_file = path.is_file() and not path.is_symlink()
    except OSError:
        is_regular_file = False
    _require(
        is_regular_file,
        "vocabulary.path.regular_file",
        path=VOCABULARY_RELATIVE_PATH.as_posix(),
    )
    try:
        content = path.read_bytes()
    except OSError:
        raise VocabularyArtifactError(
            "vocabulary.path.regular_file",
            path=VOCABULARY_RELATIVE_PATH.as_posix(),
        ) from None

    observed_sha256 = hashlib.sha256(content).hexdigest()
    _require(
        observed_sha256 == EXPECTED_ARTIFACT_SHA256,
        "vocabulary.sha256",
        expected=EXPECTED_ARTIFACT_SHA256,
        observed=observed_sha256,
    )
    artifact = _load_json_object(content)
    _require(
        content == _canonical_json_bytes(artifact),
        "vocabulary.canonical_serialization",
    )
    _verify_fixed_root_fields(artifact)
    _verify_training_works(artifact)
    code_points = _verify_vocabulary_records(artifact)
    return _create_binding(code_points)
