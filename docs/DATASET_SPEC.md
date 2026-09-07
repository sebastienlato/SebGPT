# Phase 1 Dataset Specification and Acquisition Plan

## Status

- **Specification version:** 1.4
- **Corpus design:** Approved 2026-09-06
- **Dataset specification and acquisition contract:** Approved 2026-09-06
- **Acquisition authorized:** Yes, for the raw-source acquisition and structural-inspection milestone only
- **Data acquired:** Yes; verified and tracked in the approved acquisition checkpoint
- **Extraction contract:** Approved 2026-09-06
- **Preflight contract clarification:** Approved 2026-09-06
- **Read-only preflight implementation authorized:** Yes
- **Read-only preflight implemented and verified:** Yes; accepted and tracked in its implementation checkpoint
- **In-memory extraction contract clarification:** Approved 2026-09-06
- **In-memory extraction implementation authorized:** Yes
- **In-memory extraction implemented and verified:** Yes; accepted and tracked in its implementation checkpoint
- **Transactional publisher design:** Approved 2026-09-07
- **Publisher implementation authorized:** Yes, for implementation and temporary-directory tests only
- **Publisher threat model:** Ordinary/static workspace and cooperating publishers that obey `data/.publish-lock`; unrelated same-user adversarial namespace mutation is outside the guarantee
- **Publisher implemented and verified:** Yes, including the cooperating-publisher lock, no-clobber promotion, failure-state preservation, and descriptor-relative hardening; accepted and tracked in its implementation checkpoint
- **Processed-dataset filesystem publication authorized:** No
- **Processed-dataset filesystem publication implemented:** No
- **In-memory Unicode character inventory authorized:** Yes
- **In-memory Unicode character inventory implemented and verified:** Yes; accepted and tracked in its implementation checkpoint
- **Unseen-character comparison:** Complete; all three cross-split cardinalities are zero
- **Occurrence-location reporting:** Not applicable for the pinned corpus because the candidate set is empty
- **Final processing-result ledger complete:** No
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
3. After range selection, require zero lone CR characters and convert every CRLF pair (`\r\n`) to one LF (`\n`). Perform no other newline transformation.
4. Apply no Unicode normalization before or after line-ending conversion.

Within the retained ranges after the approved boundary and local-contents exclusions, preserve exactly, except for the line-ending conversion above:

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
- removing or rewriting retained speaker labels, stage directions, or structural headings outside the approved local-contents exclusion
- deduplicating naturally repeated Shakespearean phrases

The approved extraction contract below defines exact marker-based selection, exclusions, output paths, and terminal-newline behavior. Ambiguous, missing, repeated, or reordered required markers must stop extraction rather than trigger a guessed boundary or special case.

Every approved transformation must later be deterministic, reviewable in code, and accompanied by before/after counts.

## Exact preflight structural contract

This section defines structural verification values only. It does not authorize implementation or change the approved extraction outputs.

### Full-line matching semantics

1. Verify the pinned raw SHA-256 and the raw line-ending contract before marker matching.
2. Strictly decode the raw bytes as UTF-8 in memory without BOM stripping or Unicode normalization.
3. Divide the decoded source only at the verified CRLF pairs. A “logical line” is the sequence of Unicode code points before one terminating CRLF; the CRLF is not part of the logical-line value.
4. Compare logical lines by exact, case-sensitive Unicode code-point equality. Do not trim, collapse, case-fold, normalize, or otherwise transform either the source line or expected marker.
5. Leading spaces, trailing spaces, punctuation, and Unicode characters are significant. Every structural marker below requires zero trailing spaces unless explicitly stated otherwise.

Marker selection is therefore an exact full-logical-line comparison after strict decoding, not a raw byte-substring search. Raw bytes remain authoritative for the source hash and CRLF assertions.

### Gutenberg wrapper markers

The following exact logical-line values must each occur exactly once:

- START: `*** START OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF WILLIAM SHAKESPEARE ***`
- END: `*** END OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF WILLIAM SHAKESPEARE ***`

Both begin at column zero and have no leading or trailing spaces. The START marker must precede the global contents marker; the END marker must follow every selected work and successor marker. Their observed absolute positions remain provenance assertions only and must not select ranges.

### Global contents heading and region

The global contents heading is the exact logical line formed by 20 consecutive U+0020 SPACE code points followed by `Contents`, with no trailing spaces. It is not the same marker as a play-local contents heading, which is exactly `Contents` at column zero with no leading or trailing spaces.

The global region must match this complete sequence:

1. Exactly one global heading matching `20 × U+0020 + "Contents"`.
2. Exactly one empty logical line immediately after the heading; its raw representation is CRLF only.
3. Exactly 44 consecutive nonempty entry lines. Each entry is exactly four U+0020 SPACE code points followed by the corresponding title below, with no trailing spaces.
4. Exactly four empty logical lines immediately after entry 44; each raw line is CRLF only, with no spaces or tabs.
5. The exact full-line body marker `THE SONNETS` at column zero immediately after those four empty lines.

The authoritative 44 entry values, excluding their required four-space prefix, are:

1. `THE SONNETS`
2. `ALL’S WELL THAT ENDS WELL`
3. `THE TRAGEDY OF ANTONY AND CLEOPATRA`
4. `AS YOU LIKE IT`
5. `THE COMEDY OF ERRORS`
6. `THE TRAGEDY OF CORIOLANUS`
7. `CYMBELINE`
8. `THE TRAGEDY OF HAMLET, PRINCE OF DENMARK`
9. `THE FIRST PART OF KING HENRY THE FOURTH`
10. `THE SECOND PART OF KING HENRY THE FOURTH`
11. `THE LIFE OF KING HENRY THE FIFTH`
12. `THE FIRST PART OF HENRY THE SIXTH`
13. `THE SECOND PART OF KING HENRY THE SIXTH`
14. `THE THIRD PART OF KING HENRY THE SIXTH`
15. `KING HENRY THE EIGHTH`
16. `THE LIFE AND DEATH OF KING JOHN`
17. `THE TRAGEDY OF JULIUS CAESAR`
18. `THE TRAGEDY OF KING LEAR`
19. `LOVE’S LABOUR’S LOST`
20. `THE TRAGEDY OF MACBETH`
21. `MEASURE FOR MEASURE`
22. `THE MERCHANT OF VENICE`
23. `THE MERRY WIVES OF WINDSOR`
24. `A MIDSUMMER NIGHT’S DREAM`
25. `MUCH ADO ABOUT NOTHING`
26. `THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE`
27. `PERICLES, PRINCE OF TYRE`
28. `KING RICHARD THE SECOND`
29. `KING RICHARD THE THIRD`
30. `THE TRAGEDY OF ROMEO AND JULIET`
31. `THE TAMING OF THE SHREW`
32. `THE TEMPEST`
33. `THE LIFE OF TIMON OF ATHENS`
34. `THE TRAGEDY OF TITUS ANDRONICUS`
35. `TROILUS AND CRESSIDA`
36. `TWELFTH NIGHT; OR, WHAT YOU WILL`
37. `THE TWO GENTLEMEN OF VERONA`
38. `THE TWO NOBLE KINSMEN`
39. `THE WINTER’S TALE`
40. `A LOVER’S COMPLAINT`
41. `THE PASSIONATE PILGRIM`
42. `THE PHOENIX AND THE TURTLE`
43. `THE RAPE OF LUCRECE`
44. `VENUS AND ADONIS`

Preflight must compare this ordered list literally. It must not infer titles, authorship, genre, capitalization, or semantic Shakespeare structure.

### Successor-separator assertions

The generic end-boundary rule remains: scan backward from the exact successor marker across consecutive CRLF-only empty lines and exclude them. In addition, preflight must assert the complete pinned-source observations recorded per work in the authoritative manifest:

- *Hamlet*: 4 empty CRLF-only lines; 0 whitespace-only lines.
- *Romeo and Juliet*: 4 empty CRLF-only lines; 0 whitespace-only lines.
- *Macbeth*: 5 empty CRLF-only lines; 0 whitespace-only lines.
- *A Midsummer Night’s Dream*: 4 empty CRLF-only lines; 0 whitespace-only lines.
- *Much Ado About Nothing*: 4 empty CRLF-only lines; 0 whitespace-only lines.
- *Henry V*: 4 empty CRLF-only lines; 0 whitespace-only lines.
- *The Tempest*: 4 empty CRLF-only lines; 0 whitespace-only lines.
- *Twelfth Night*: 4 empty CRLF-only lines; 0 whitespace-only lines.

These successor-separator counts are required verification assertions, not boundary-selection inputs. Other observed blank-line counts around selected title markers are informational provenance only. A line containing one or more spaces, tabs, or other whitespace code points is whitespace-only, not empty, and must cause the separator assertion to fail rather than being removed as part of the empty-line run.

## Deterministic eight-play extraction contract

This contract is approved as policy. It does not authorize implementation or processed-output creation.

1. The only permitted input is `data/raw/gutenberg-ebook-100/complete-works.txt` with SHA-256 `3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37`.

2. Read the raw file as bytes and verify its hash before decoding. Decode a copy in memory using strict UTF-8 without BOM stripping or Unicode normalization. Never rewrite the raw file.

3. Verify the exact Gutenberg START and END full-line markers, global contents heading, complete ordered 44-entry contents region, and structural expectations defined in the preflight contract above. Recorded absolute lines and offsets are assertions only, never boundary-selection inputs.

4. Identify each selected work by its exact full-line body-title marker after the global contents. Require one global-contents occurrence and exactly one post-contents body occurrence for every selected marker.

5. Identify each work's successor using the exact successor-title mapping in `docs/data/shakespeare-eight-play-structure.md`. Use the start of that successor marker as the outer upper bound.

6. Set the work start at the first byte/code point of its body-title marker. Exclude every preceding separator line.

7. Starting immediately before the successor marker, remove consecutive lines containing no bytes other than CRLF. End the raw work segment immediately after the CRLF terminating the preceding nonempty line. Require the per-work empty and whitespace-only separator counts in the preflight contract. Whitespace-only lines are not considered empty; encountering one in the boundary separator fails preflight.

8. Include the top-level work title.

9. Remove the local contents range beginning at the first exact full line `Contents` after the work title and ending immediately before the exact per-work `Dramatis Personæ` line constructed under the shared-marker rule below. Require both markers to occur exactly once, be correctly ordered, and lie inside the work's outer range.

10. Include `Dramatis Personæ`, acts, scenes, speaker labels, stage directions, indentation, blank lines, spaces, tabs, trailing spaces, punctuation, capitalization, spelling, and every Unicode code point within the retained ranges.

11. After range selection, replace every CRLF pair with LF. Require zero lone CR characters before conversion and zero CR characters afterward. Perform no other normalization.

12. Preserve the final source line terminator through extraction so CRLF conversion produces exactly one terminal LF. Do not use general-purpose stripping or whitespace-normalization operations.

13. Write one processed UTF-8 document per approved work at its assigned split path. Do not concatenate works or create sequences spanning document boundaries:

    - `data/processed/shakespeare-eight-play/train/hamlet.txt`
    - `data/processed/shakespeare-eight-play/train/romeo-and-juliet.txt`
    - `data/processed/shakespeare-eight-play/train/macbeth.txt`
    - `data/processed/shakespeare-eight-play/train/a-midsummer-nights-dream.txt`
    - `data/processed/shakespeare-eight-play/train/much-ado-about-nothing.txt`
    - `data/processed/shakespeare-eight-play/train/henry-v.txt`
    - `data/processed/shakespeare-eight-play/validation/the-tempest.txt`
    - `data/processed/shakespeare-eight-play/test/twelfth-night.txt`

14. For each work, record:

    - stable work ID, title, manifest order, and split
    - exact start and successor markers
    - observed marker occurrence counts
    - raw byte, Unicode code-point, and line offsets
    - outer raw-range SHA-256 and byte count, measured before local-contents removal
    - retained raw-range SHA-256 and byte count, measured after local-contents removal and before CRLF-to-LF conversion
    - every excluded range and reason
    - normalization operations
    - processed repository-relative path
    - processed byte, Unicode code-point, line, and whitespace-word counts
    - processed SHA-256
    - Python version, script path, and processing-code commit

15. Verify range ordering, non-overlap, expected first marker, expected successor exclusion, Gutenberg-wrapper exclusion, local-contents removal, `Dramatis Personæ` retention, LF-only output, terminal LF, distinct output hashes, and byte-identical repeated runs.

16. Apply the same algorithm to training, validation, and test without split-specific extraction behavior.

17. For *Twelfth Night*, expose only structural markers and machine-readable numeric diagnostics. Never display source prose, surrounding context, text diffs, token sequences, or samples. A failed test invariant stops processing and does not authorize a test-specific rule.

18. Extraction produces text documents and provenance metadata only. It does not create a character inventory, tokens, token IDs, vocabulary, unknown-input policy, context windows, batches, or model inputs.

19. Character inventories are a later Phase 1 diagnostic over the approved processed files. Tokenization and all unseen-input representation choices remain Phase 2 decisions.

### Boundary-selection authority

Exact title and successor strings come from `docs/data/shakespeare-eight-play-structure.md` and the authoritative manifest. Observed absolute lines, byte offsets, and Unicode code-point offsets are required verification assertions tied to the pinned raw hash but never selection inputs. Per-work successor-separator counts explicitly listed above and in the manifest are also required assertions. Other observed blank-line counts are informational provenance only. The implementation must select by exact full-line structural markers and fail if independently calculated required assertions differ; it must not select by hard-coded absolute offsets.

### Shared `Dramatis Personæ` marker and per-work indentation

The canonical marker text is exactly `Dramatis Personæ`. That shared value contains no leading or trailing whitespace and is stored once in the authoritative manifest.

For each work, construct its expected exact logical line as:

`U+0020 SPACE × dramatis_leading_u0020_count` + `Dramatis Personæ`

The approved per-work values are:

| Work | `dramatis_leading_u0020_count` |
|---|---:|
| *Hamlet* | 0 |
| *Romeo and Juliet* | 1 |
| *Macbeth* | 0 |
| *A Midsummer Night's Dream* | 0 |
| *Much Ado About Nothing* | 0 |
| *Henry V* | 0 |
| *The Tempest* | 0 |
| *Twelfth Night* | 1 |

Match using exact, case-sensitive full-logical-line Unicode equality under the approved CRLF line semantics. Do not strip, trim, normalize Unicode, collapse whitespace, or infer indentation. No trailing spaces are permitted. Exactly one constructed marker must occur within each selected outer work range, after the exact zero-indent local `Contents` marker and before the outer-range end. Required order is:

`work title < local Contents < exact per-work Dramatis Personæ line < outer-range end`

The shared marker avoids repeating identical Unicode text in every work record; only the source-varying leading-space count belongs to each work.

### Provenance hash-range definitions

The following are three distinct byte ranges at three deterministic pipeline stages. Each has its own SHA-256 and byte count; the hashes must never be substituted for or conflated with one another.

**Outer raw range**

- Contains original raw-source bytes.
- Begins at the first raw byte of the selected work's exact body-title marker.
- Ends immediately after the CRLF terminating the final nonempty source line selected by the approved successor-separator rule.
- Is measured before local-contents removal and therefore includes the removable local `Contents` range.
- Its SHA-256 is computed over those exact contiguous original bytes.

**Retained raw range**

- Is the exact byte concatenation of:
  1. outer raw bytes from the outer start through immediately before the local `Contents` marker; and
  2. outer raw bytes beginning at the exact constructed per-work `Dramatis Personæ` line through the outer end.
- Is measured after local-contents removal and before CRLF-to-LF normalization.
- Its SHA-256 is computed over those exact concatenated raw bytes.

**Processed range**

- Is the UTF-8 byte representation resulting solely from replacing every retained CRLF pair with LF.
- Permits no other normalization or text transformation.
- Its SHA-256 is computed over those final in-memory processed bytes.

### Split isolation

The six training plays remain six physical documents. A later dataset loader must treat each file as an independent sequence domain and may never form a context window across documents. *The Tempest* and *Twelfth Night* remain separate physical and logical validation/test documents. This constrains later loading but does not choose tokens, context length, sampling strategy, or model inputs.

### Publication concurrency and threat model

Processed-dataset publication is transaction-safe for an ordinary/static SebGPT workspace and for multiple cooperating SebGPT publisher invocations that all obey the exclusive empty presence lock at `data/.publish-lock`. The lock is created atomically without following symlinks, held across output-state inspection, staging construction, promotion, and final verification, and released after ordinary success or a clean pre-publication refusal. A stale or uncertain lock and any staging residue are preserved for explicit human resolution; later publishers refuse them.

Within that threat model, inputs and bytes are validated before writing, staged regular files are created exclusively without following symlinks, exact reruns are idempotent, mismatching or partial state is refused, failed transactions preserve uncertainty, final promotion atomically refuses an existing destination, and the published tree is verified afterward. Descriptor-relative and inode checks remain defense-in-depth. The lock coordinates SebGPT publishers only and is not a security boundary: the publisher does not guarantee safety against an unrelated process with the same filesystem authority that ignores the protocol and deliberately mutates the `data` namespace during the transaction. Darwin provides neither atomic directory creation with descriptor return nor inode-conditional source rename for this directory-tree architecture.

## Unicode character inventory and unseen-character audit

Phase 1 defines Unicode character inventories only. It does not select a token unit, tokenizer algorithm, token vocabulary, encoding map, unknown-token mechanism, or unseen-input policy. All such choices belong to Phase 2.

The training character inventory must be derived from processed **training data only**. Before tokenizer implementation, Phase 1 must audit the processed validation and test documents for Unicode code points absent from that training inventory.

For Phase 1 counting, a “character” is one Unicode code point obtained by iterating the strictly decoded Python string after approved extraction and CRLF-to-LF conversion, with lone CR rejected. It is not a byte, glyph, grapheme cluster, or future token.

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

For the pinned corpus, the accepted production comparison found validation-not-in-train cardinality 0, test-not-in-train cardinality 0, and test-not-in-train-or-validation cardinality 0. The occurrence-location candidate set and occurrence-entry collection are therefore both empty. Occurrence-location reporting is not applicable, and no test-only code-point identities or prose were emitted. This is the mechanical zero-candidate result of the existing contract, not a change to it.

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
- **Unicode code points:** Python `len(text)` after approved extraction and CRLF-to-LF conversion, with lone CR rejected.
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
8. Verify `.gitattributes` will preserve the raw hash through Git without staging or committing the acquisition milestone.
9. Update the tracked provenance and state records, then stop for review. Do not extract, normalize, inventory characters, tokenize, or create splits during the acquisition milestone.
10. Only after explicit milestone approval, commit the raw master and completed provenance records together so the exact source becomes recoverable from repository history before extraction design begins.

## Acceptance gates

The corpus design, dataset specification/acquisition contract, deterministic extraction contract, exact preflight clarification, and source-faithful `Dramatis Personæ`/provenance hash-range clarification were approved on 2026-09-06. Raw-source acquisition, structural inspection, read-only preflight, deterministic in-memory extraction, and the in-memory Unicode character inventory are complete, accepted, and tracked in dedicated checkpoints. The unseen-character audit is complete with zero candidates, making occurrence-location reporting not applicable for the pinned corpus. The transactional publisher's Models A/B cooperative-workspace threat model, exclusive lock, no-clobber promotion, failure-state preservation, and descriptor-relative hardening are accepted and tracked in its implementation checkpoint. Production filesystem publication and processed-output creation remain unauthorized, and the final processing-result ledger is incomplete. Phase 1 is not complete. Tokenization remains Phase 2 and is not authorized by approval of any Phase 1 step.
