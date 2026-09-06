# Shakespeare Raw-Source Structural Inspection

## Status and scope

- **Inspected:** 2026-09-06
- **Acquisition milestone review:** Approved 2026-09-06
- **Raw source:** `data/raw/gutenberg-ebook-100/complete-works.txt`
- **Raw SHA-256:** `3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37`
- **Inspection type:** Read-only structural diagnostics
- **Extraction rules approved:** No; design discussion only is authorized
- **Extraction performed:** No

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
- The unique Gutenberg START marker is line 25, byte offset 815, Unicode code-point offset 815.
- A complete-works title page appears after the START marker: the collection title is on line 30 and author line on line 32.
- The global `Contents` marker is line 37.
- The global contents entries occupy lines 39–82 and contain 44 uppercase work-title entries.
- The first top-level body marker is `THE SONNETS` on line 87.
- The unique Gutenberg END marker is line 196,048, byte offset 5,619,552, Unicode code-point offset 5,556,275.
- The full Gutenberg license footer begins at byte offset 5,619,638, immediately after the END marker's terminating CRLF, and continues through EOF for 18,842 bytes.
- The license-footer SHA-256 is `ae0d20778842abddfa129878dabff2a1c67a33d854c294ecfe7ac0d51e0d8dcb`.

These observations do not yet authorize which exact lines belong to or are excluded from processed data. That decision belongs to the extraction-design gate.

## General work-delimiter pattern

Each selected work has one exact uppercase title line in the global contents and one unique exact uppercase body-title line after the global contents. Selected body titles have no leading or trailing spaces. Seven selected markers have four blank lines immediately after the title; *Henry V* has two. All selected markers have four blank lines immediately before the title.

Inspected non-test plays contain their own internal `Contents` section, repeated act/scene listings, a `Dramatis Personæ` heading, and then the play body. This means “the work starts at its title” does not by itself answer whether a play-local contents list belongs in processed text; that inclusion decision must be explicit at the extraction-design gate.

A whole-file exact-title scan found 46 post-global-contents matches for the 44 global contents titles. Two unselected titles occur twice: `KING HENRY THE EIGHTH` at lines 71,019 and 71,059, and `VENUS AND ADONIS` at lines 194,612 and 194,646. Therefore, extraction must not assume every contents title has exactly one later occurrence. All eight selected body-title markers were unique in the post-contents scan.

## Selected work boundary markers

The “next marker” below is an observed candidate for an end-exclusive structural boundary, not an approved extraction range.

| Split | Work | Body-title marker | Start line | Start byte | Start code-point offset | Next top-level marker | Next line |
|---|---|---|---:|---:|---:|---|---:|
| train | *Hamlet* | `THE TRAGEDY OF HAMLET, PRINCE OF DENMARK` | 34,438 | 976,463 | 964,679 | `THE FIRST PART OF KING HENRY THE FOURTH` | 41,136 |
| train | *Henry V* | `THE LIFE OF KING HENRY THE FIFTH` | 51,139 | 1,471,705 | 1,454,574 | `THE FIRST PART OF HENRY THE SIXTH` | 56,085 |
| train | *Macbeth* | `THE TRAGEDY OF MACBETH` | 96,022 | 2,774,909 | 2,744,552 | `MEASURE FOR MEASURE` | 100,172 |
| train | *A Midsummer Night's Dream* | `A MIDSUMMER NIGHT’S DREAM` | 114,050 | 3,281,135 | 3,245,088 | `MUCH ADO ABOUT NOTHING` | 117,535 |
| train | *Much Ado About Nothing* | `MUCH ADO ABOUT NOTHING` | 117,535 | 3,382,288 | 3,345,304 | `THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE` | 122,135 |
| train | *Romeo and Juliet* | `THE TRAGEDY OF ROMEO AND JULIET` | 143,372 | 4,109,746 | 4,066,035 | `THE TAMING OF THE SHREW` | 148,639 |
| validation | *The Tempest* | `THE TEMPEST` | 153,507 | 4,389,497 | 4,342,367 | `THE LIFE OF TIMON OF ATHENS` | 157,343 |
| test | *Twelfth Night* | `TWELFTH NIGHT; OR, WHAT YOU WILL` | 172,093 | 4,902,784 | 4,849,279 | `THE TWO GENTLEMEN OF VERONA` | 176,589 |

Manifest order is intentionally independent of source order. Deterministic extraction must use stable work identifiers and approved explicit boundaries rather than assuming the six training works are adjacent in the source.

## Formatting facts relevant to extraction design

- The raw source uses CRLF exclusively; the approved processing contract later converts CRLF to LF.
- Typography includes Unicode punctuation, including the curly apostrophe in `A MIDSUMMER NIGHT’S DREAM`, and the `æ` character in `Dramatis Personæ`.
- The source includes a global contents list and play-local contents lists, so repeated structural headings are expected.
- Blank-line counts around title markers are mostly, but not completely, uniform.
- Exact-title occurrence counts are not uniform across all 44 works, even though the eight selected body-title markers are unique.
- The catalog reported `Last Update: 2025-08-24`, while the downloaded artifact response reported `Last-Modified: Tue, 01 Sep 2026 07:55:49 GMT`. Both values are retained as separate provenance facts.

## Questions reserved for extraction design

The next gate must decide, with exact line/byte rules:

- whether selected work title lines are included
- whether each play-local contents list is included or excluded
- whether each `Dramatis Personæ` section is included
- the exact end-exclusive boundary for every selected work
- which blank lines adjacent to approved boundaries are included
- terminal-newline behavior after CRLF-to-LF conversion
- exact processed filenames and the manifest schema for extracted results

No answer is selected by this inspection record.
