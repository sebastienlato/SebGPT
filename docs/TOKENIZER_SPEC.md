# Phase 2 Code-Point Tokenizer Contract

## Status and authority

This document is the accepted detailed contract for SebGPT's first tokenizer.
The first independent review returned FAIL, Sebastien accepted its six findings,
and the corrected contract then passed focused independent re-review with no
must-fix issues. DEC-0013's scope correction is the accepted conceptual boundary;
DEC-0014 and this document are the accepted detailed contract.

This checkpoint authorizes documentation and contract design only. It does not
authorize tokenizer implementation, production vocabulary construction,
production encoding, token statistics, context windows, embeddings, or later
phases.

The Phase 1 dataset specification and authoritative manifest retain status
statements showing that tokenization was unauthorized when Phase 1 closed.
DEC-0013 and the current `PROJECT_STATE.md` supersede only those historical
authorization-status statements. This Phase 2 checkpoint does not modify or
reinterpret any Phase 1 dataset identity, artifact, split, hash, or policy.

## Accepted conceptual contract

### Token unit and text preservation

- One element produced by iterating a Python `str` is one token unit.
- The corresponding unit identity is its `ord()` Unicode code-point integer.
- This is code-point tokenization, not grapheme-cluster, byte, word, or subword
  tokenization.
- Tokenization performs no Unicode normalization or other text transformation.
- Exact round trips for supported text preserve capitalization, spaces, tabs,
  newlines, blank lines, punctuation, every Unicode code point, and the final
  newline.

### Vocabulary membership and ID principle

- Vocabulary membership is derived from training only.
- Numeric Unicode code-point ordering determines IDs.
- There are no special tokens initially.

### Unknown input and round trips

- An unsupported code point is rejected strictly rather than represented by an
  unknown token.
- For every supported input, `decode(encode(text)) == text` exactly.

### Document and phase boundary

- Documents remain independent.
- Phase 2 performs no context-window construction.

Exact freeze mechanics, contiguous zero-based IDs, error fields, API/container
rules, builder and enforcement architecture, artifact schema and provenance,
statistics, later-phase allocation, possible future tokenizers, and gate
mechanics below are governed by accepted DEC-0014, not DEC-0013.

## Accepted detailed vocabulary and split contract

- The production vocabulary is constructed once from exactly the approved six
  Phase 1 training works and is then frozen.
- Validation cannot add entries. The sealed test work cannot participate.
- Vocabulary code points are strictly increasing; tuple positions are
  contiguous token IDs from zero through `vocabulary_size - 1`.
- Frequency, first occurrence, locale, validation, and test information have no
  effect on membership or ID assignment.
- There are no special, reserved, unused, or gap IDs.
- Training may be encoded and inspected statistically only after authorization.
- Validation may use the frozen training vocabulary and aggregate statistics.
- The sealed test work remains untokenized and statistically uninspected during
  Phase 2.
- The historical zero-unseen-code-point audit does not authorize Phase 2 access
  to test identities, text, token sequences, or token statistics.

## Accepted code-point integer domain

A tokenizer code point is an exact built-in Python `int` satisfying:

```text
0 <= value <= 0x10FFFF
```

This domain includes surrogate code-point integers `U+D800` through `U+DFFF`.
Python `str` can contain them, and filtering them would add a conceptual
restriction beyond Python-string iteration plus `ord()`. The accepted
Shakespeare corpus cannot contain surrogates because it was produced through
strict UTF-8 decoding, but the generic tokenizer introduces no scalar-value
filter and no normalization.

Canonical uppercase notation is `U+` followed by the value's uppercase
hexadecimal representation padded to at least four digits: for example,
`U+000A`, `U+D800`, and `U+10FFFF`.

## Accepted public API

The implementation should expose one small immutable tokenizer object:

```python
CodePointTokenizer(code_points: tuple[int, ...])
CodePointTokenizer.encode(text: str) -> tuple[int, ...]
CodePointTokenizer.decode(token_ids: Sequence[int]) -> str
```

`Sequence` refers to the appropriate future standard-library
`collections.abc` or typing surface. Construction accepts canonical state only;
it does not sort, deduplicate, or coerce invalid caller-supplied values.

### Encode

- Input acceptance uses `isinstance(text, str)`, so ordinary `str` subclasses
  are accepted. Non-string values raise `TypeError` and are never converted with
  `str()`.
- Iteration position is a zero-based index into the Python `str`, so it counts
  code points, not UTF-8 bytes or displayed grapheme clusters.
- `encode("")` returns `()`.
- The result is an immutable tuple of exact built-in integers.
- Repeated code points produce repeated IDs.
- Equal tokenizer state and equal input always produce equal output.
- The method accepts exactly one string. Documents, iterables of documents,
  batches, files, and paths are outside this API.
- Encoding scans positions strictly from left to right. At the first unsupported
  code point it stops immediately, returns no partial tuple, and raises
  `UnknownCodePointError`.

Encoding may locate an ID by binary search in the ordered code-point tuple. At
the expected small vocabulary size, this keeps the canonical object state to
one sequence instead of maintaining a second exposed mapping.

### Unknown-code-point error

`UnknownCodePointError`, a `ValueError` subclass, may expose only:

- `code_point`: the `ord()` integer
- `u_plus`: uppercase `U+` notation with at least four hexadecimal digits
- `position`: the zero-based Python-`str` code-point index

Its deterministic message may contain those three values and no others. It must
not contain the literal character, `repr(text)`, surrounding text, excerpts,
filenames, document identities, or Unicode names. Error attributes should be
immutable or read-only. This content-safe shape does not authorize sealed-test
encoding: a test error would still reveal a test-specific code-point identity
and position.

### Decode

- Accept `token_ids` exactly when
  `isinstance(token_ids, collections.abc.Sequence)` is true, except that `str`,
  `bytes`, and `bytearray` are always rejected.
- This mechanical predicate accepts tuples and lists and rejects generators and
  iterators. A tensor is accepted only if it independently satisfies the same
  `collections.abc.Sequence` predicate and exclusions; the tokenizer contains
  no PyTorch-specific logic.
- Inputs are neither implicitly converted nor mutated.
- Every element must have exact type `int`; `bool` and other integer-like values
  are rejected rather than coerced.
- Any accepted empty sequence decodes to `""`.
- IDs from zero through `vocabulary_size - 1` are valid.
- Negative and out-of-range IDs raise the same
  `InvalidTokenIdError`, a `ValueError` subclass.
- Repeated valid IDs are allowed and reproduce the corresponding code point
  repeatedly.
- Decoding performs no normalization, cleanup, special-token filtering, or
  replacement.
- Validation proceeds strictly left to right and raises on the first invalid
  element. No partial decoded string is returned.

When the offending value has exact type `int`, the invalid-ID error may expose
only that integer, its zero-based sequence position, and the valid half-open
numeric range. It does not include adjacent IDs or partially decoded text. A
wrong element type error may expose only the element position, the exact-`int`
invariant, and applicable bounds. It must not include the observed type name,
the element's value, or its representation.

## Accepted tokenizer data model

The sole canonical vocabulary state is:

```text
code_points: tuple[int, ...]
```

The state itself must be a tuple. Every element has exact type `int`, falls in
the inclusive range `0` through `0x10FFFF` (including surrogates), and is
strictly greater than its predecessor. Duplicates are therefore impossible. A
production vocabulary must be nonempty.

The tuple index is the token ID and the tuple value is the code point.
`vocabulary_size` is the read-only derived value `len(code_points)`. Decoding
requires no second ID-to-code-point structure, and encoding may use binary
search over the tuple.

The implementation exposes no mutable dictionaries and creates no private
lookup cache. A cache is not justified for the first implementation.

## Accepted two-layer vocabulary construction

### Layer A — corpus-neutral membership algorithm

A small transparent function accepts independent Python strings and computes
vocabulary membership only. It knows nothing about Shakespeare, split names,
the manifest schema, `ExtractedWork`, validation, or sealed-test governance.

For every input string independently, it computes the set of `ord(character)`
values. It unions those per-document sets, sorts the result numerically, and
returns an immutable tuple. It never joins strings or defines adjacency. Set
union retains membership only, so it cannot create a cross-document sequence.

The neutral function validates its string inputs without coercion. It may return
an empty tuple for empty neutral input; the production orchestrator is
responsible for the nonempty production-vocabulary invariant.

### Layer B — Shakespeare production orchestrator

The production layer consumes the accepted Phase 1 `ExtractedWork` handoff and
the authoritative tracked Phase 1 manifest. It does not invent another generic
document class, reread raw or processed prose, rerun preflight or extraction, or
accept a concatenated corpus string.

Before accessing any `processed_text`, it validates safe metadata against the
authoritative manifest:

- manifest schema and dataset identity are the accepted Phase 1 values
- the input contains exactly the approved six training work IDs, once each
- work order is exactly the manifest order `1` through `6`
- every split is exactly `train`
- normalization identity is exactly the accepted Phase 1 identity
- processed SHA-256, byte count, and code-point count equal both the
  `ExtractedWork` provenance and manifest values

The approved IDs, in deterministic manifest order, are `hamlet`,
`romeo-and-juliet`, `macbeth`, `a-midsummer-nights-dream`,
`much-ado-about-nothing`, and `henry-v`.

Any unexpected, duplicate, missing, reordered, validation, or test work is
rejected before its text field or content is accessed. Only after every metadata
check passes may the orchestrator access each training `processed_text`
independently. It then strictly UTF-8 encodes that string and independently
recomputes its encoded byte count, SHA-256, and Python code-point count. Each
recomputed value must equal both the `ExtractedWork` provenance and the
authoritative manifest. A mismatch stops construction before the neutral
membership algorithm runs.

After all six documents pass, their six independent strings are passed as
separate inputs to Layer A. The returned tuple must be nonempty and satisfy the
canonical-state invariants. Production construction remains separately gated.

## Accepted sealed-test enforcement layer

`CodePointTokenizer` is generic and contains no Shakespeare, manifest, or split
knowledge. Mechanical sealed-test enforcement belongs to production vocabulary
and statistics orchestration.

Vocabulary-construction orchestration inspects safe metadata first and rejects
both validation and test objects before their `processed_text` property or
content is accessed.

Statistics orchestration may access training text. It may access validation
text only under the accepted aggregate-only policy. It rejects every test object
before its `processed_text` property or content is accessed.

Future deterministic tests must use sentinel or access-guard fixtures proving
each access boundary. Governance documentation remains an additional protection
rather than the only enforcement mechanism.

## Accepted canonical vocabulary artifact contract

### Format, path, and tracking

Use deterministic UTF-8 JSON at this path:

```text
artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json
```

The artifact should be Git-tracked as a narrow, explicit exception for this
small model-critical reproducibility record. JSON is preferred over CSV or a
Python module because it can hold both the mapping and its provenance without
making data executable.

Do not store literal characters. Control characters, spaces, and visually
confusable Unicode can make literal review ambiguous. Store the integer and
uppercase `U+` notation. Do not store Unicode names: they are redundant,
dependent on a Unicode database version, and unnecessary to reconstruct text.

### Exact accepted schema

The root is a JSON object with exactly these keys and JSON types; additional
keys are forbidden:

| Key | JSON type | Exact value or rule |
|---|---|---|
| `schema_version` | number (integer) | `1` |
| `tokenizer_id` | string | `"shakespeare-code-point-v1"` |
| `token_unit` | string | `"python_str_code_point"` |
| `normalization` | string | `"none"` |
| `unknown_policy` | string | `"error"` |
| `special_token_policy` | string | `"none"` |
| `id_order` | string | `"ascending_unicode_code_point"` |
| `contract_decision` | string | `"DEC-0014"` |
| `phase_1_manifest_sha256` | string | SHA-256 of the accepted tracked Phase 1 manifest |
| `implementation_git_commit` | string | Reviewed clean tokenizer implementation commit |
| `training_works` | array | Exactly six work objects |
| `vocabulary_size` | number (integer) | Positive number of vocabulary records |
| `vocabulary` | array | Ordered vocabulary-record objects |

JSON booleans are not integers for this schema. The
`implementation_git_commit` is the lowercase 40-hex-character object ID
returned by the current repository's `git rev-parse HEAD`; construction
requires that reviewed commit to be checked out with a clean worktree.

Each `training_works` element is an object with exactly:

| Key | JSON type | Rule |
|---|---|---|
| `work_id` | string | Exact approved nonempty Phase 1 work ID |
| `manifest_order` | number (integer) | Exact accepted manifest order |
| `processed_sha256` | string | Exact accepted processed-text SHA-256 |

The array contains exactly the six approved training works in ascending
`manifest_order`: `hamlet`, `romeo-and-juliet`, `macbeth`,
`a-midsummer-nights-dream`, `much-ado-about-nothing`, and `henry-v`. Work IDs
and manifest orders are unique. Every record must match the accepted Phase 1
manifest; validation and test records are forbidden.

Each `vocabulary` element is an object with exactly:

| Key | JSON type | Rule |
|---|---|---|
| `token_id` | number (integer) | Zero-based contiguous ID equal to array position |
| `code_point` | number (integer) | Value from `0` through `0x10FFFF` inclusive |
| `code_point_uplus` | string | Canonical uppercase representation of `code_point` |

`vocabulary_size` equals the vocabulary-array length. IDs begin at zero with no
gaps and equal their array positions. Code points are strictly increasing and
therefore unique. `code_point_uplus` equals `U+` followed by uppercase
hexadecimal padded to at least four digits. Literal-character and Unicode-name
fields are forbidden at every level.

Every SHA-256 field is exactly 64 lowercase hexadecimal characters matching
`^[0-9a-f]{64}$`. The artifact contains no source-file digest, Python version,
timestamp, hostname, absolute path, prose sample, validation fact, test fact,
unknown key, literal-character field, or Unicode-name field.

### Deterministic serialization and hashing

Serialize with sorted object keys, two-space indentation, ASCII-safe JSON,
disallowed non-finite numbers, UTF-8 without a BOM, LF line endings, and exactly
one terminal LF. Arrays retain their contract-defined order. Hash the exact
serialized bytes with SHA-256.

The artifact does not contain its own hash. Record its path and SHA-256 in the
Phase 2 production-checkpoint acceptance update and current project state.
Python version belongs in that construction/verification record rather than the
artifact because the mapping does not depend on Unicode names or platform
locale.

### Authority relationship and immutability

1. The Phase 1 authoritative manifest authorizes the exact training inputs.
2. Accepted DEC-0014 and its contract define mapping semantics.
3. The reviewed, clean, committed tokenizer implementation performs
   construction.
4. The accepted Git-tracked vocabulary artifact and its separately recorded
   SHA-256 become authoritative for the concrete token-ID mapping.

Runtime code must load and validate the frozen artifact; it must not silently
regenerate, extend, reorder, or overwrite it. Rebuilding from the same accepted
inputs with the reviewed implementation must reproduce identical bytes and
hash. A mismatch stops processing for review. After any model training begins,
a mapping change requires a new tokenizer ID and artifact rather than mutation
of the existing mapping.

Future model and training metadata must bind at least `tokenizer_id`, vocabulary
artifact SHA-256, schema version, and vocabulary size while recording dataset
manifest identity separately. Later embedding and output dimensions must equal
the frozen vocabulary size. Checkpoint format is deliberately not designed in
Phase 2.

## Accepted statistics plan

Statistics are computed only after the implementation and production
vocabulary gates authorize them. They do not alter the vocabulary.

### Training

- token count per independent work and total token count
- complete frequency-by-token-ID table, with code-point integer and optional
  `U+` notation but no required literal-character display
- 100% vocabulary coverage and zero unknowns as construction invariants
- exact per-work round-trip success

Frequency extrema are derived presentation from the complete frequency table,
not separate canonical data.

### Validation

- aggregate token count
- aggregate coverage proportion and unknown count
- exact whole-document round-trip success when the unknown count is zero

Validation statistics must use the already frozen training vocabulary and
cannot change it. Reports should contain structural and numeric facts rather
than source excerpts. Validation unknown identities are not exposed unless a
later review explicitly authorizes them.

All first-tokenizer Phase 2 statistics remain ephemeral in memory or console
output. No canonical statistics artifact is required.

### Sealed test

No Phase 2 encoding, token counts, coverage calculation, unknown check,
round-trip attempt, frequency analysis, token sequence, or tokenizer statistic
is permitted for the sealed test work.

## Context-window and later-phase boundary

Phase 2 produces independent token-ID streams only. It does not construct
context windows, input/target pairs, batches, embeddings, or model inputs.
Introducing next-token window and input/target semantics with Phase 4's
prediction objective, and production sampling and batching with Phase 8, is a
proposed future allocation consistent with the current roadmap. It is not an
accepted DEC-0013 commitment and may be changed by a later accepted decision.

Byte-level and BPE tokenizers are possible future educational comparisons only.
They are not Phase 2 exit requirements, are not part of accepted DEC-0013, and
are not authorized for implementation.

## Accepted Phase 2 gates

1. **Conceptual strategy approved.** Complete through corrected DEC-0013.
2. **Detailed contract documented.** Complete in corrected DEC-0014 and this
   document.
3. **Independent review, correction, and acceptance.** Complete: the first
   review failed, all accepted corrections were documented, and focused
   re-review passed with no must-fix issues.
4. **Contract commit.** Authorized for this accepted checkpoint.
5. **Contract push and remote verification.** Authorized after the contract
   commit and required immediately afterward.
6. **Explicit implementation authorization.** Not authorized.
7. **Implementation and deterministic tests.** Not started.
8. **Independent implementation review, correction, and acceptance.** Not
   started.
9. **Implementation commit.** Not started.
10. **Implementation push and remote verification.** Not started.
11. **Explicit production vocabulary and statistics authorization.** Not
    authorized.
12. **Construct vocabulary artifact and hash; inspect permitted statistics.**
    Not started.
13. **Independent production-checkpoint review, correction, and acceptance.**
    Not started.
14. **Vocabulary-checkpoint commit.** Not started.
15. **Vocabulary-checkpoint push and remote verification.** Not started.
16. **Phase 2 exit-criteria review.** Not started.
17. **Final Phase 2 closure commit, if required.** Not started.
18. **Final closure push and remote verification.** Not started.
19. **Only then authorize Phase 3.** Not authorized.

## Focused re-review result

The corrected contract passed focused independent re-review with no must-fix
issues. Sebastien explicitly accepted DEC-0014 and authorized the contract
checkpoint commit. This acceptance does not authorize implementation,
production vocabulary construction, statistics, context windows, or later-phase
work.
