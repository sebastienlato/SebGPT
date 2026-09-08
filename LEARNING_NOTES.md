# Learning Notes

This document grows only as concepts are actually covered with Sebastien. It is not a prewritten textbook or a list of topics merely planned for later.

## Entry format

For each concept covered, record:

- **Date**
- **Concept**
- **Explanation in our own words**
- **Small example or observation**
- **What remains unclear**
- **Related code, test, decision, or experiment**

## Concepts covered

### 2026-09-06 — Dataset suitability and leakage-resistant splits

- **Explanation in our own words:** A language model learns next-token patterns from the distribution it is shown, so a dataset defines the language, structure, subject matter, and limitations the model can learn. A useful first dataset is legally and technically traceable, coherent, small enough to inspect, large enough to contain repeated patterns, and reproducible from immutable raw material. Training data teaches parameters; validation data supports development choices; test data is reserved for a final unbiased check.
- **Small example or observation:** If lines from one play are randomly placed into every split, validation can contain nearby or repeated material from a play the model trained on. Assigning complete plays to one split each instead asks whether patterns learned from six plays transfer to an unseen play.
- **What remains unclear:** Actual processed counts remain unknown until extraction. Validation/test may contain Unicode code points absent from training; Phase 1 will report them unchanged under a sealed-test diagnostic contract, and Phase 2 will choose a generic unseen-input policy.
- **Related code, test, decision, or experiment:** `docs/DATASET_SPEC.md`; DEC-0007 and DEC-0008. The raw source is acquired, but no extraction implementation or processed data exists.

### 2026-09-06 — Extraction boundaries versus normalization

- **Explanation in our own words:** Extraction answers which source ranges belong to each document; minimal normalization answers how retained text is represented consistently. Split assignment places complete documents into train, validation, or test. A character inventory measures the resulting Unicode code points without changing them. Tokenization later decides how text maps to token IDs. Keeping these stages separate prevents a cleanup or file-layout choice from silently becoming a tokenizer decision.
- **Small example or observation:** The approved extractor will identify *Hamlet* by exact title and successor markers, remove its redundant local contents range, retain `Dramatis Personæ` and the dramatic text, and only then convert retained CRLF pairs to LF. It will write *Hamlet* as its own training document rather than concatenate it with another play.
- **What remains unclear:** Actual processed counts, hashes, and validation/test-only Unicode code points remain unknown until extraction is implemented and verified. Their discovery cannot authorize tokenization behavior.
- **Related code, test, decision, or experiment:** `docs/DATASET_SPEC.md`; `docs/data/shakespeare-eight-play-structure.md`; DEC-0009. No extractor or processed data exists yet.

### 2026-09-08 — Token units, vocabularies, and exact round trips

- **Explanation in our own words:** Raw text, token units, vocabulary entries, and token IDs are different layers. Choosing one Python `str` iteration element per token makes the unit a Unicode code point, not a displayed grapheme, UTF-8 byte, word, or subword. The Phase 1 count of 81 distinct training code points becomes the eventual vocabulary size only because we separately chose training-only code-point units and no special tokens. Sorting the integers returned by `ord()` gives a deterministic mapping without making frequency or source order part of token identity.
- **Small example or observation:** In a conceptual input such as `A\nA`, iteration sees three code points. The repeated `A` uses the same ID twice and the newline uses its own ID. Encoding the empty string produces an empty immutable sequence, and decoding it returns the empty string. Exact round trips require preserving every code point and forbid normalization, whitespace cleanup, or unknown-token substitution.
- **What remains unclear:** The conceptual strategy is accepted, but the proposed API strictness, safe error fields, direct `ExtractedWork` builder boundary, canonical JSON artifact, and statistics representation still require independent review. No production vocabulary or real token IDs exist yet.
- **Related code, test, decision, or experiment:** DEC-0013; proposed DEC-0014 and `docs/TOKENIZER_SPEC.md`. No tokenizer implementation or experiment exists.

### 2026-09-08 — Separating generic tokenization from corpus governance

- **Explanation in our own words:** A generic tokenizer only needs to understand code points and IDs; it should not know Shakespeare split policy. Corpus-neutral vocabulary membership can union per-document code-point sets without creating adjacency. A separate production orchestrator establishes that the six inputs are the accepted training documents before reading their text, while a statistics orchestrator mechanically prevents sealed-test access. This separates reusable mechanism from project-specific governance without weakening either.
- **Small example or observation:** Python `str` can contain surrogate code-point integers even though strict UTF-8 source decoding cannot produce them. Therefore the generic tokenizer's proposed domain includes every exact integer from `0` through `0x10FFFF`, while the production Shakespeare vocabulary naturally contains no surrogate because of its verified provenance. Decoding now accepts finite positional integer sequences such as tuples and lists, validates left-to-right, and returns no partial text after the first failure.
- **What remains unclear:** No known ambiguity remains from the six accepted review findings, but the corrected detailed contract still requires focused independent re-review and acceptance. None of its implementation mechanics are accepted merely because they are now documented.
- **Related code, test, decision, or experiment:** DEC-0013 scope correction; corrected proposed DEC-0014; `docs/TOKENIZER_SPEC.md`. No tokenizer implementation, vocabulary, token IDs, artifact, or experiment exists.

### 2026-09-08 — Contract review turns intent into mechanical boundaries

- **Explanation in our own words:** Independent review is part of the learning mechanism: it distinguishes a sound conceptual direction from details that are precise enough to implement and test. The accepted decode boundary is now a direct standard-library predicate with explicit exclusions, not an informal category. Split safety is likewise expressed as observable access behavior: validation is forbidden during vocabulary construction but permitted under aggregate-only statistics, while test text is inaccessible in both production paths.
- **Small example or observation:** A list of integer IDs satisfies the accepted decode container predicate, while a generator does not. Neither is silently converted. In future access-guard tests, a validation object must fail if vocabulary construction touches its text, but validation statistics may read it; a test object must fail on text access in either orchestration path.
- **What remains unclear:** The accepted contract has no known ambiguity after focused re-review. The implementation has not been designed in code or authorized, so concrete module structure and tests must follow the later implementation gate without changing the accepted behavior.
- **Related code, test, decision, or experiment:** DEC-0014 acceptance update; `docs/TOKENIZER_SPEC.md`; Phase 2 Gate 3 complete. No tokenizer code, vocabulary, production IDs, or statistics exist.
