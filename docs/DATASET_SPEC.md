# Phase 1 Dataset Specification and Acquisition Plan

## Status

- **Specification version:** 1.1
- **Corpus design:** Approved 2026-09-06
- **Dataset specification and acquisition contract:** Approved 2026-09-06
- **Acquisition authorized:** Yes, for the raw-source acquisition and structural-inspection milestone only
- **Data acquired:** No
- **Extraction implemented:** No
- **Tokenization implemented:** No

## Learning objective

The first SebGPT corpus should make the complete path from a documented source text to reproducible language-model inputs understandable and inspectable. It is intentionally a narrow Shakespearean dramatic corpus, not a representative sample of modern English and not a benchmark for general language ability.

The corpus should be large enough to contain repeated linguistic and dramatic patterns while remaining small enough to inspect, process, and train on repeatedly on the verified local Mac environment.

## Exact source

- **Provider:** Project Gutenberg
- **Catalog title:** *The Complete Works of William Shakespeare*
- **Author:** William Shakespeare
- **Project Gutenberg eBook number:** 100
- **Authoritative catalog URL:** <https://www.gutenberg.org/ebooks/100>
- **Required acquisition format:** The UTF-8 plain-text artifact linked as **Plain Text (accessible)** from the authoritative catalog page
- **Catalog state observed during design:** Last updated 2025-08-24; catalog labels the work public domain in the USA
- **Project Gutenberg terms:** <https://www.gutenberg.org/policy/license>

The catalog record and eBook number define the approved source. The resolved artifact URL, server metadata, actual catalog update date, access timestamp, byte count, and SHA-256 hash must be captured at acquisition time because Project Gutenberg may update the artifact.

One complete-works source is preferred to separate play downloads because it gives all eight works a consistent editorial and formatting convention. No mirror, third-party dataset wrapper, existing Tiny Shakespeare file, or alternate edition may be substituted without a new decision.

Project Gutenberg describes this item as public domain in the United States. At acquisition, preserve the source's included license text in the immutable raw artifact, record the catalog and license URLs, and note that use outside the United States remains subject to applicable local law. This project documentation is not a legal opinion.

## Approved corpus manifest

Approximate word counts are design estimates, not verified measurements. Acquisition and extraction must report actual byte, Unicode code-point, line, and whitespace-delimited word counts using the definitions below without changing the manifest merely to force an exact ratio.

| Order | Split | Work | Estimated words | Purpose |
|---:|---|---|---:|---|
| 1 | train | *Hamlet* | 30,000 | Long tragedy with dialogue, monologues, and varied speakers |
| 2 | train | *Romeo and Juliet* | 25,000 | Tragedy combining romance, comedy, and lyrical dialogue |
| 3 | train | *Macbeth* | 17,000 | Short tragedy with strong recurring language |
| 4 | train | *A Midsummer Night's Dream* | 17,000 | Comedy and fantasy with a contrasting tone |
| 5 | train | *Much Ado About Nothing* | 21,000 | Dialogue-heavy comedy and verbal exchanges |
| 6 | train | *Henry V* | 26,000 | History play with speeches, conflict, and a different register |
| 7 | validation | *The Tempest* | 17,000 | Unseen romance/fantasy work for development decisions |
| 8 | test | *Twelfth Night* | 20,000 | Unseen comedy reserved for final evaluation |

The design target is approximately 170,000 words and 0.9–1.1 million processed Unicode code points. Estimated proportions are approximately 79% training, 10% validation, and 11% test.

Sonnets, poems, and all other plays are excluded. They may not be added merely to reach the approximate size target.

## Split policy

The unit of splitting is a complete work, not a line, scene, paragraph, token, or context window.

- The six training works are the only text permitted to influence learned parameters or learned preprocessing.
- *The Tempest* is used for validation during development and model-selection decisions.
- *Twelfth Night* is the sealed test work. Except for the narrowly defined structural and machine-diagnostic access contract below, it must not be used for training, preprocessing decisions, hyperparameter tuning, early stopping, or repeated informal evaluation.
- A sequence or context window must never cross a work boundary, even between two works in the same split.
- Split membership and manifest order are fixed before acquisition and must not be changed in response to measured loss or generated samples.

Whole-work splitting makes evaluation harder and noisier than random slicing, but it measures transfer to an unseen play rather than continuation of a familiar play. The approximate ratio is subordinate to this leakage-resistant semantic boundary.

## Raw and processed data separation

The planned local layout is:

```text
data/
├── raw/
│   └── gutenberg-ebook-100/
│       └── complete-works.txt
└── processed/
    └── shakespeare-eight-play/
        ├── train/
        ├── validation/
        ├── test/
        └── processing-manifest.json
```

These paths are a specification only and must not be created before acquisition is approved.

- `data/raw/gutenberg-ebook-100/complete-works.txt` contains the exact retrieved bytes and is immutable after hashing. Never edit the raw artifact in place.
- `data/processed/` contains only deterministic derivatives of the raw artifact.
- The raw master above is the sole narrow exception to the general dataset-ignore policy. It must be Git-tracked together with its completed provenance record after acquisition verification. `.gitattributes` disables Git text conversion and ordinary textual diffs for this exact path so a checkout preserves the hashed bytes without reports exposing the source body.
- All processed data and every other current or future dataset path remain excluded from Git unless a later decision explicitly authorizes another exact exception.
- `docs/data/shakespeare-eight-play-manifest.json` is the authoritative tracked machine-readable record for concrete source identity, acquisition facts, hashes, the work manifest, transformations, counts, split membership, and audit results.
- The Markdown specification defines policy; the tracked JSON manifest records the concrete dataset instance. If they conflict with each other or with `DECISIONS.md`, processing must stop for review.
- Any `source-metadata.json`, `processing-manifest.json`, logs, or reports generated under ignored `data/` paths are reproducible execution outputs. They may support debugging but never override the tracked manifest.
- If the upstream source changes, preserve the identity of the version already used. Treat a newer artifact as a new dataset version rather than silently replacing the raw file.

## Normalization and extraction principles

Processing must be deliberately minimal, mechanical, and auditable. Acquisition does not authorize any processing.

The following character-decoding and line-ending semantics are fixed:

1. Decode the approved source with strict UTF-8 semantics, equivalent to Python's `encoding="utf-8", errors="strict"`; decoding errors stop processing.
2. Do not use `utf-8-sig`. If the decoded source begins with a byte-order mark, retain it as U+FEFF during inspection and report it. Its eventual inclusion or exclusion requires an explicit extraction rule.
3. Convert every CRLF pair (`\r\n`) to one LF (`\n`), then convert every remaining CR (`\r`) to LF. Perform no other newline transformation.
4. Apply no Unicode normalization before or after line-ending conversion.

Within each approved extraction range, preserve exactly, except for the line-ending conversion above:

- original spelling and capitalization
- punctuation and Unicode characters
- act and scene headings
- speaker labels
- stage directions
- every blank line, tab, ordinary space, other whitespace code point, trailing space, and line in source order

Forbidden without an explicit later decision:

- Unicode normalization
- smart-quote or dash replacement
- lowercasing
- spelling modernization
- whitespace collapsing
- punctuation removal
- automatic typo correction
- removing or rewriting speaker labels, stage directions, or structural headings
- deduplicating naturally repeated Shakespearean phrases
- adding or removing a terminal newline

The exact source ranges for the Project Gutenberg wrapper, table of contents, other material outside the selected plays, and each selected play cannot be known safely before the raw artifact is inspected. No such ranges are authorized by this specification. The post-acquisition extraction-design gate must document exact start/end structural markers, line or byte ranges, filename mapping, excluded ranges with reasons, and terminal-newline behavior. Those rules require review before extraction implementation. Ambiguous, missing, or repeated candidate markers must stop the design process rather than trigger a guessed boundary.

Every approved transformation must later be deterministic, reviewable in code, and accompanied by before/after counts.

## Unicode character inventory and unseen-character audit

Phase 1 defines Unicode character inventories only. It does not select a token unit, tokenizer algorithm, token vocabulary, encoding map, unknown-token mechanism, or unseen-input policy. All such choices belong to Phase 2.

The training character inventory must be derived from processed **training data only**. Before tokenizer implementation, Phase 1 must audit the processed validation and test documents for Unicode code points absent from that training inventory.

For Phase 1 counting, a “character” is one Unicode code point obtained by iterating the strictly decoded Python string after CRLF/CR-to-LF conversion and approved extraction. It is not a byte, glyph, grapheme cluster, or future token.

- Combining marks are separate code points and are not composed with neighboring code points.
- U+0020 SPACE, U+0009 CHARACTER TABULATION, U+000A LINE FEED, every other whitespace code point, and every control code point are counted individually and remain distinct.
- U+000D CARRIAGE RETURN must not remain after the defined line-ending conversion.
- U+FEFF is counted if it lies within an approved extraction range; it is never silently stripped as a byte-order mark.
- No NFC, NFD, NFKC, NFKD, case, punctuation, whitespace, or compatibility normalization is permitted.

Each character inventory must be sorted by ascending integer code-point value and record the frequency of every code point. For every validation/test code point absent from the training inventory, the machine-readable audit must record only these fields:

- `code_point_decimal`: the integer code-point value
- `code_point_hex`: `U+` followed by uppercase hexadecimal padded to at least four digits, such as `U+000A` or `U+1F600`
- `unicode_name`: the Unicode name, or the exact string `UNNAMED` when no name exists
- `validation_count` and `test_count`: occurrence counts kept separate
- `occurrences`: records containing only `work_id`, one-based `line_number`, and zero-based `code_point_offset` from the start of that processed work

The audit is observational only. It must not display surrounding source prose or source slices and must not silently discard, replace, normalize, or add any code point. Phase 2 must choose a generic unseen-input policy that applies uniformly to arbitrary unseen input; its rationale may not cite or tailor behavior to the identities, counts, or locations observed in the sealed test work.

This structural/diagnostic audit is the sole permitted pre-evaluation exception to test sealing. Automated tools may inspect the test work only to locate and verify approved structural boundary markers and to emit the fields listed above. Before final evaluation, humans and automated reports must not expose test dialogue, prose, surrounding context, samples, token sequences, or model behavior on the test work. Repository review must use hashes, file sizes, status, and diff statistics rather than commands that print the raw source body.

## Leakage rules

1. Keep each complete work in exactly one split.
2. Never create overlapping windows across work or split boundaries.
3. At the extraction-design gate, define exact exclusions for source material outside the selected plays so shared source boilerplate cannot enter multiple splits.
4. Apply one approved extraction and line-ending procedure to every work; do not tune it using test content or results.
5. Derive any learned preprocessing from training data only.
6. Do not use validation or test text as examples when making Phase 2 tokenization decisions.
7. Detect and report accidental duplicated works or duplicated extracted regions using hashes and boundary/count checks.
8. Report meaningful cross-split repeated passages if discovered, but do not automatically delete natural repetition.
9. Use validation for development choices and reserve test for the explicitly scheduled final evaluation.

## Provenance and reproducibility requirements

Use these deterministic count definitions:

- **Raw bytes:** exact filesystem byte length of the unmodified acquired artifact.
- **Processed bytes:** byte length after encoding the processed string with strict UTF-8.
- **Unicode code points:** Python `len(text)` after approved extraction and CRLF/CR-to-LF conversion.
- **Lines:** zero for an empty string; otherwise the number of U+000A LINE FEED code points plus one when the string does not end in U+000A.
- **Words:** Python `len(text.split())`, using the Python version recorded in the authoritative manifest. This is a whitespace-delimited diagnostic, not a token count.

Acquisition must record:

- source provider, title, author, eBook number, catalog URL, and resolved artifact URL
- source catalog update date shown at acquisition
- UTC acquisition timestamp
- format and declared encoding
- HTTP `ETag` and `Last-Modified` values when supplied
- raw byte count and SHA-256 hash
- applicable source license/terms URLs and the license notice contained in the raw artifact
- acquisition method and tool versions

Processing must record:

- source raw SHA-256 hash
- processing script path and Git commit
- Python version and operating platform
- exact extraction markers for every selected work
- deterministic work order and split assignment
- all transformations performed
- per-work and per-split byte, character, line, and word counts
- SHA-256 hash of every processed document
- excluded ranges and the reasons for exclusion
- duplicate/boundary audit results
- train/validation/test Unicode character inventories and the unseen-character audit using the defined code-point semantics

The raw master and its SHA-256-bearing manifest must coexist in the same committed repository tree. Git history records the commit that introduced them; the manifest does not attempt to contain its own commit hash. A checkout of that tree must recover raw bytes matching the manifest's SHA-256. Re-running processing from that raw hash and the recorded processing-code revision must produce byte-identical processed files and hashes.

## Planned acquisition procedure

After explicit acquisition approval:

1. Re-read this specification and confirm the repository is clean enough to identify acquisition-related changes.
2. Open the authoritative catalog page and verify title, author, eBook number, update date, public-domain label, and current terms.
3. Resolve the catalog's current **Plain Text (accessible)** link without substituting another edition or format.
4. Retrieve that artifact once over HTTPS into the planned raw directory without transforming it.
5. Capture provenance metadata in `docs/data/shakespeare-eight-play-manifest.json` and calculate the raw byte count and SHA-256 hash immediately.
6. Verify that the saved bytes decode strictly as UTF-8, while leaving the raw file unchanged.
7. Inspect only source metadata and structural markers needed to propose exact excluded and work-boundary ranges. Apply the pre-evaluation test-access contract; do not emit test prose.
8. Verify `.gitattributes` preserves the raw hash through Git, then commit the raw master and completed tracked manifest together so the exact source remains recoverable.
9. Update `PROJECT_STATE.md`, then stop for review. Do not extract, normalize, inventory characters, tokenize, or create splits during the acquisition milestone.

## Acceptance gates

The corpus design and dataset specification/acquisition contract were approved on 2026-09-06. Acquisition is authorized only for the raw-source acquisition, factual provenance capture, and structural-inspection procedure above. Extraction requires another review after acquisition and structural inspection. Tokenization remains Phase 2 and is not authorized by approval of any Phase 1 step.
