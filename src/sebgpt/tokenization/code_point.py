"""A small, exact Unicode code-point tokenizer."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field


MAX_CODE_POINT = 0x10FFFF
_EXCLUDED_SEQUENCES = (str, bytes, bytearray)


def code_point_uplus(code_point: int) -> str:
    """Return the canonical uppercase notation for a validated code point."""

    return f"U+{code_point:04X}"


class TokenizationTypeError(TypeError):
    """A deterministic type failure that never includes an input value."""

    __slots__ = ("_invariant", "_position")

    def __init__(self, invariant: str, *, position: int | None = None) -> None:
        self._invariant = invariant
        self._position = position
        message = f"tokenization type invariant failed: {invariant}"
        if position is not None:
            message += f" at position {position}"
        super().__init__(message)

    @property
    def invariant(self) -> str:
        return self._invariant

    @property
    def position(self) -> int | None:
        return self._position


class TokenizerStateError(ValueError):
    """A deterministic canonical-state failure containing no text."""

    __slots__ = ("_invariant", "_position")

    def __init__(self, invariant: str, *, position: int | None = None) -> None:
        self._invariant = invariant
        self._position = position
        message = f"tokenizer state invariant failed: {invariant}"
        if position is not None:
            message += f" at position {position}"
        super().__init__(message)

    @property
    def invariant(self) -> str:
        return self._invariant

    @property
    def position(self) -> int | None:
        return self._position


class UnknownCodePointError(ValueError):
    """The first code point that is absent from the frozen vocabulary."""

    __slots__ = ("_code_point", "_u_plus", "_position")

    def __init__(self, code_point: int, position: int) -> None:
        self._code_point = code_point
        self._u_plus = code_point_uplus(code_point)
        self._position = position
        super().__init__(
            f"unsupported code point {code_point} ({self._u_plus}) "
            f"at position {position}"
        )

    @property
    def code_point(self) -> int:
        return self._code_point

    @property
    def u_plus(self) -> str:
        return self._u_plus

    @property
    def position(self) -> int:
        return self._position


class InvalidTokenIdError(ValueError):
    """The first exact integer outside the tokenizer's valid ID range."""

    __slots__ = (
        "_token_id",
        "_position",
        "_lower_bound",
        "_upper_bound_exclusive",
    )

    def __init__(
        self,
        token_id: int,
        position: int,
        upper_bound_exclusive: int,
    ) -> None:
        self._token_id = token_id
        self._position = position
        self._lower_bound = 0
        self._upper_bound_exclusive = upper_bound_exclusive
        super().__init__(
            f"token ID {token_id} at position {position} must satisfy "
            f"0 <= value < {upper_bound_exclusive}"
        )

    @property
    def token_id(self) -> int:
        return self._token_id

    @property
    def position(self) -> int:
        return self._position

    @property
    def lower_bound(self) -> int:
        return self._lower_bound

    @property
    def upper_bound_exclusive(self) -> int:
        return self._upper_bound_exclusive


@dataclass(frozen=True)
class CodePointTokenizer:
    """An immutable mapping whose token IDs are tuple positions."""

    code_points: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.code_points) is not tuple:
            raise TokenizerStateError("code_points.exact_tuple")
        if not self.code_points:
            raise TokenizerStateError("code_points.nonempty")

        previous: int | None = None
        for position, code_point in enumerate(self.code_points):
            if type(code_point) is not int:
                raise TokenizerStateError(
                    "code_points.element.exact_int",
                    position=position,
                )
            if not 0 <= code_point <= MAX_CODE_POINT:
                raise TokenizerStateError(
                    "code_points.element.range",
                    position=position,
                )
            if previous is not None:
                if code_point == previous:
                    raise TokenizerStateError(
                        "code_points.element.duplicate",
                        position=position,
                    )
                if code_point < previous:
                    raise TokenizerStateError(
                        "code_points.strictly_increasing",
                        position=position,
                    )
            previous = code_point

    @property
    def vocabulary_size(self) -> int:
        return len(self.code_points)

    def __repr__(self) -> str:
        return f"CodePointTokenizer(vocabulary_size={self.vocabulary_size})"

    def encode(self, text: str) -> tuple[int, ...]:
        """Map one string to immutable IDs, failing at the first unknown."""

        if not isinstance(text, str):
            raise TokenizationTypeError("encode.text.isinstance_str")

        token_ids: list[int] = []
        for position, character in enumerate(text):
            code_point = ord(character)
            token_id = bisect_left(self.code_points, code_point)
            if (
                token_id == self.vocabulary_size
                or self.code_points[token_id] != code_point
            ):
                raise UnknownCodePointError(code_point, position)
            token_ids.append(token_id)
        return tuple(token_ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        """Map a permitted integer sequence back to exact Python text."""

        if not isinstance(token_ids, Sequence) or isinstance(
            token_ids,
            _EXCLUDED_SEQUENCES,
        ):
            raise TokenizationTypeError("decode.token_ids.sequence")

        characters: list[str] = []
        for position in range(len(token_ids)):
            token_id = token_ids[position]
            if type(token_id) is not int:
                raise TokenizationTypeError(
                    "decode.token_id.exact_int_and_range",
                    position=position,
                )
            if not 0 <= token_id < self.vocabulary_size:
                raise InvalidTokenIdError(
                    token_id,
                    position,
                    self.vocabulary_size,
                )
            characters.append(chr(self.code_points[token_id]))
        return "".join(characters)


def build_code_point_vocabulary(documents: Sequence[str]) -> tuple[int, ...]:
    """Return sorted membership from independent strings without joining them."""

    if not isinstance(documents, Sequence) or isinstance(
        documents,
        _EXCLUDED_SEQUENCES,
    ):
        raise TokenizationTypeError("membership.documents.sequence")

    membership: set[int] = set()
    for position, document in enumerate(documents):
        if not isinstance(document, str):
            raise TokenizationTypeError(
                "membership.document.isinstance_str",
                position=position,
            )
        membership.update(ord(character) for character in document)
    return tuple(sorted(membership))
