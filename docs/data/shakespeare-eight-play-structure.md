# Shakespeare Raw-Source Structural Inspection

## Status and scope

- **Inspected:** 2026-09-06
- **Acquisition milestone review:** Approved 2026-09-06
- **Raw source:** `data/raw/gutenberg-ebook-100/complete-works.txt`
- **Raw SHA-256:** `3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37`
- **Inspection type:** Read-only structural diagnostics
- **Extraction contract:** Approved 2026-09-06
- **Exact preflight clarification:** Approved 2026-09-06
- **Read-only preflight implemented and verified:** Yes; accepted and tracked in its implementation checkpoint
- **In-memory extraction marker/hash clarification:** Approved 2026-09-06
- **In-memory extraction implemented and verified:** Yes; accepted and tracked in its implementation checkpoint
- **Processed-dataset filesystem publication authorized:** No
- **Processed-dataset filesystem publication performed:** No

The inspection located source-level markers and measured raw structure only. It did not decode and rewrite the stored file, normalize line endings, remove material, extract works, create splits, inventory characters, or tokenize text. For *Twelfth Night*, only its exact title marker, offsets, surrounding blank-line counts, and the next top-level work marker were exposed. No test prose or internal test structure was printed or analyzed.

## Raw file measurements

- Bytes: 5,638,480
- Strict UTF-8 decoding: passed
- UTF-8 BOM: absent
- Unicode code points before any normalization: 5,575,099
- LF bytes: 196,398
- CRLF pairs: 196,398
- Lone CR bytes: 0
- Lone LF bytes: 0
- File ending: CRLF
- NUL bytes: 0

Every observed line ending is CRLF. These measurements describe the immutable raw artifact; no line endings were converted.

## Project Gutenberg wrapper and global front structure

Offsets are zero-based. Line numbers are one-based raw physical lines.

- Source metadata/preamble occupies lines 1–24.
- The unique Gutenberg START marker is line 25, byte offset 815, Unicode code-point offset 815. Its exact logical-line value is `*** START OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF WILLIAM SHAKESPEARE ***`, with zero leading or trailing spaces.
- A complete-works title page appears after the START marker: the collection title is on line 30 and author line on line 32.
- The global contents marker is line 37. Its exact logical-line value is 20 U+0020 SPACE code points followed by `Contents`, with zero trailing spaces.
- Line 38 is one CRLF-only empty line.
- The global contents entries occupy lines 39–82 and contain 44 consecutive nonempty work-title entries. Each exact logical line has four leading U+0020 SPACE code points, one title from the authoritative ordered list in `docs/DATASET_SPEC.md` and the machine manifest, and zero trailing spaces.
- Lines 83–86 are four CRLF-only empty lines.
- The first top-level body marker is `THE SONNETS` on line 87.
- The unique Gutenberg END marker is line 196,048, byte offset 5,619,552, Unicode code-point offset 5,556,275. Its exact logical-line value is `*** END OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF WILLIAM SHAKESPEARE ***`, with zero leading or trailing spaces.
- The full Gutenberg license footer begins at byte offset 5,619,638, immediately after the END marker's terminating CRLF, and continues through EOF for 18,842 bytes.
- The license-footer SHA-256 is `ae0d20778842abddfa129878dabff2a1c67a33d854c294ecfe7ac0d51e0d8dcb`.

Full-line matching removes only the terminating CRLF and otherwise compares exact, case-sensitive Unicode code points without trimming or normalization. The approved extraction contract uses absolute observations as verification expectations and selects boundaries only by exact full-line markers.

## General work-delimiter pattern

Each selected work has one exact uppercase title line in the global contents and one unique exact uppercase body-title line after the global contents. Selected body titles have no leading or trailing spaces. Seven selected markers have four blank lines immediately after the title; *Henry V* has two. All selected markers have four blank lines immediately before the title. These title-surrounding counts are informational provenance, not required preflight assertions.

Selected plays contain their own internal `Contents` section, repeated act/scene listings, a source-indented form of the shared `Dramatis Personæ` marker, and then the play body. The approved contract includes the top-level title, removes the local contents range, and retains the exact per-work `Dramatis Personæ` line and the play body.

### `Dramatis Personæ` structural observations

The shared canonical marker text is exactly `Dramatis Personæ`, with code points ending in U+00E6 and with no whitespace stored in the shared value. The expected logical line is constructed by prefixing the per-work number of U+0020 SPACE code points. No selected work has another indentation variant or trailing spaces.

| Work | Leading U+0020 count | Exact-marker occurrences in outer range |
|---|---:|---:|
| *Hamlet* | 0 | 1 |
| *Romeo and Juliet* | 1 | 1 |
| *Macbeth* | 0 | 1 |
| *A Midsummer Night's Dream* | 0 | 1 |
| *Much Ado About Nothing* | 0 | 1 |
| *Henry V* | 0 | 1 |
| *The Tempest* | 0 | 1 |
| *Twelfth Night* | 1 | 1 |

A whole-file exact-title scan found 46 post-global-contents matches for the 44 global contents titles. Two unselected titles occur twice: `KING HENRY THE EIGHTH` at lines 71,019 and 71,059, and `VENUS AND ADONIS` at lines 194,612 and 194,646. Therefore, extraction must not assume every contents title has exactly one later occurrence. All eight selected body-title markers were unique in the post-contents scan.

## Selected work boundary markers

The approved contract uses each exact “next marker” below as the outer end-exclusive structural anchor, then mechanically excludes the consecutive empty CRLF separator lines immediately before it.

| Split | Work | Body-title marker | Start line | Start byte | Start code-point offset | Next top-level marker | Next line | Required empty CRLF lines before next | Required whitespace-only lines before next |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| train | *Hamlet* | `THE TRAGEDY OF HAMLET, PRINCE OF DENMARK` | 34,438 | 976,463 | 964,679 | `THE FIRST PART OF KING HENRY THE FOURTH` | 41,136 | 4 | 0 |
| train | *Henry V* | `THE LIFE OF KING HENRY THE FIFTH` | 51,139 | 1,471,705 | 1,454,574 | `THE FIRST PART OF HENRY THE SIXTH` | 56,085 | 4 | 0 |
| train | *Macbeth* | `THE TRAGEDY OF MACBETH` | 96,022 | 2,774,909 | 2,744,552 | `MEASURE FOR MEASURE` | 100,172 | 5 | 0 |
| train | *A Midsummer Night's Dream* | `A MIDSUMMER NIGHT’S DREAM` | 114,050 | 3,281,135 | 3,245,088 | `MUCH ADO ABOUT NOTHING` | 117,535 | 4 | 0 |
| train | *Much Ado About Nothing* | `MUCH ADO ABOUT NOTHING` | 117,535 | 3,382,288 | 3,345,304 | `THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE` | 122,135 | 4 | 0 |
| train | *Romeo and Juliet* | `THE TRAGEDY OF ROMEO AND JULIET` | 143,372 | 4,109,746 | 4,066,035 | `THE TAMING OF THE SHREW` | 148,639 | 4 | 0 |
| validation | *The Tempest* | `THE TEMPEST` | 153,507 | 4,389,497 | 4,342,367 | `THE LIFE OF TIMON OF ATHENS` | 157,343 | 4 | 0 |
| test | *Twelfth Night* | `TWELFTH NIGHT; OR, WHAT YOU WILL` | 172,093 | 4,902,784 | 4,849,279 | `THE TWO GENTLEMEN OF VERONA` | 176,589 | 4 | 0 |

Manifest order is intentionally independent of source order. Deterministic extraction must use stable work identifiers and approved explicit boundaries rather than assuming the six training works are adjacent in the source.

## Formatting facts relevant to extraction design

- The raw source uses CRLF exclusively; the approved processing contract later converts CRLF to LF.
- Typography includes Unicode punctuation, including the curly apostrophe in `A MIDSUMMER NIGHT’S DREAM`, and the `æ` character in `Dramatis Personæ`.
- The source includes a global contents list and play-local contents lists, so repeated structural headings are expected.
- Title-surrounding blank-line counts are informational and not completely uniform. Successor-separator counts in the boundary table are required assertions; *Macbeth* has five CRLF-only lines and the other selected works have four, with zero whitespace-only separator lines for all eight.
- Exact-title occurrence counts are not uniform across all 44 works, even though the eight selected body-title markers are unique.
- The catalog reported `Last Update: 2025-08-24`, while the downloaded artifact response reported `Last-Modified: Tue, 01 Sep 2026 07:55:49 GMT`. Both values are retained as separate provenance facts.

## Approved extraction-policy summary

- Include each selected work's exact top-level body-title line.
- Select the outer end using the exact successor marker, excluding its immediately preceding empty CRLF separator lines.
- Remove the local range from the first exact zero-indent `Contents` line through immediately before the exact constructed per-work `Dramatis Personæ` line.
- Retain the exact source-indented `Dramatis Personæ` line, acts, scenes, speaker labels, stage directions, typography, indentation, and internal whitespace.
- Convert CRLF to LF only after range selection; perform no other normalization.
- Produce one terminal LF by preserving the final retained source CRLF through extraction and converting it.
- Write eight separate processed documents at the paths defined in `docs/DATASET_SPEC.md`.
- Treat the observed absolute positions as assertions, never selection inputs.
- Apply the sealed-test structural/diagnostic contract to *Twelfth Night*.

This summary records policy only. The accepted in-memory extraction implementation exists, but no processed data has been published.
