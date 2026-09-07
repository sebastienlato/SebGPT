# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 1 — Dataset

## Current milestone

In-memory Unicode character inventory accepted and tracked in its dedicated checkpoint; occurrence-location audit remains unimplemented.

## Completed work

- SebGPT project concept agreed.
- Learning-first approach agreed.
- Python and PyTorch direction agreed.
- Work will be used for primary project collaboration.
- Codex will be used as an engineering assistant.
- Repository files will be the authoritative source of truth.
- Initial continuity documents and project directories created.
- Phase 0.1 continuity infrastructure reviewed and approved.
- Mac environment audited: macOS 26.6.2, Apple M4 Max (`arm64`), Python.org CPython 3.14.4, pip 26.0.1, and Git 2.54.0.
- Project-local `.venv` created and excluded from Git.
- Stable PyTorch 2.14.0 installed from its native macOS ARM64 wheel.
- Direct and fully resolved dependency manifests created.
- CPU tensor operations, autograd, MPS availability, and an MPS tensor operation verified.
- Environment test suite added and verified with all three tests passing on the host Mac.
- Git repository initialized on `main` with a clean Phase 0 baseline commit.
- Phase 1 explicitly authorized.
- Dataset suitability, scope, provenance, and leakage tradeoffs discussed.
- Project Gutenberg eBook #100 selected as the sole source for an approximately 170,000-word eight-play corpus.
- Whole-work manifest approved: six training plays, *The Tempest* for validation, and *Twelfth Night* for test.
- Training-only Unicode character inventory and non-mutating validation/test unseen-character audit approved without selecting a tokenizer.
- Dataset specification and gated acquisition contract documented in `docs/DATASET_SPEC.md`.
- Exact raw-source Git preservation exception and authoritative tracked manifest contract documented.
- Corrected dataset specification and acquisition contract reviewed and approved.
- Dataset-design specification committed as `a3821da6331c8e885f6935c442e896105c9a9340`.
- The catalog-linked Project Gutenberg eBook #100 UTF-8 artifact acquired once to the approved raw path without transformation.
- Raw source verified at 5,638,480 bytes with SHA-256 `3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37`.
- Strict UTF-8 decoding passed; no BOM or NUL bytes were found; all 196,398 observed line endings are CRLF.
- Source provenance and response metadata recorded in the authoritative tracked JSON manifest.
- Read-only wrapper, global contents, selected-work marker, and formatting inspection recorded without exposing *Twelfth Night* prose.
- Raw-source acquisition and provenance milestone reviewed and approved for its dedicated checkpoint.
- Raw-source acquisition checkpoint committed as `8ffba6e005ac26e1b3f81c1d914df97e2acd7b25`.
- Deterministic eight-play extraction policy approved and recorded as DEC-0009.
- Exact marker-based boundaries, content inclusions/exclusions, minimal normalization, split paths, verification invariants, metadata, and sealed-test rules documented.
- Exact Gutenberg wrapper anchors, global contents contract, ordered 44-entry list, and eight successor-separator assertions approved under DEC-0010.
- Exact-preflight documentation checkpoint committed as `6efb00a5cd8e9831f83846fc30acf152bc27e1e9`.
- Final authoritative-status consistency verification completed with no remaining technical blockers.
- Read-only preflight implementation explicitly authorized.
- Standard-library preflight and focused tests implemented without extraction or output-writing behavior.
- Production preflight passed against the pinned source; all 16 focused tests passed, and the full 19-test suite passed with the expected restricted-context MPS skip.
- Independent review passed and the read-only preflight implementation was accepted for its dedicated checkpoint.
- Source-faithful per-work Dramatis indentation and three distinct provenance hash stages approved under DEC-0011.
- Deterministic in-memory extraction explicitly authorized and implemented through a separate module.
- Immutable validated-source handoff added so extraction consumes the exact byte object that passed preflight.
- Eighteen focused extraction tests passed; the 17-test preflight suite passed; the full 38-test suite passed with the expected restricted-context MPS skip.
- Production extraction returned eight separate works deterministically and computed three-stage provenance without creating processed output.
- Independent review passed and the deterministic in-memory extraction implementation was accepted for its dedicated checkpoint.
- In-memory Unicode character inventory explicitly authorized and implemented without rereading, re-extracting, or publishing text.
- Ten focused inventory tests passed; extraction and preflight regressions passed; the full 48-test suite passed with the expected restricted-context MPS skip.
- Production inventory counted eight independent works deterministically under Python 3.14.4 without creating `data/processed/`.
- Independent review passed with no must-fix issues, and Sebastien accepted the in-memory Unicode character-inventory checkpoint.

## Current work

None. The in-memory Unicode character inventory is accepted; occurrence-location audit and filesystem publication remain separate, unauthorized work.

## Current model status

None. No model code exists.

## Last verified working state

The accepted preflight and in-memory extraction remain intact. The inventory consumes the accepted `tuple[ExtractedWork, ...]`, counts exact Python Unicode code points, returns immutable per-work/split/global aggregates and cross-split relationships, and exposes only totals/cardinalities through its safe summary. Production reported 1,005,813 total and 81 distinct code points; validation-not-in-train, test-not-in-train, and test-not-in-train-or-validation cardinalities are all zero. No occurrence-location audit, filesystem publication, tokenizer, dataset loader, or model implementation exists.

## Next exact step

Sebastien explicitly authorizes the occurrence-location audit before it is implemented.

## Known issues

- PyTorch emits a warning that optional NumPy interoperability is unavailable. NumPy is intentionally not installed because no current milestone requires it.
- MPS is unavailable inside some restricted execution contexts, but direct host verification and the unrestricted test suite both pass.
- The catalog reports a 2025-08-24 update date, while the artifact response reports `Last-Modified: Tue, 01 Sep 2026 07:55:49 GMT`; both are recorded without assuming they describe the same revision mechanism.
- In-memory processed counts and hashes are reproducibly computed but are not published to dataset files or a processing manifest.
- The occurrence-location audit is not implemented. Current in-memory relationship cardinalities show no validation/test code points absent from training, but that does not authorize skipping its separate review gate.

## Important constraints

- Do not implement processed-dataset filesystem publication or create `data/processed/` until Sebastien explicitly authorizes that separate milestone.
- Do not add dependencies without an approved, recorded reason and lock-file update.
- Do not write extracted text, normalized text, split documents, or extraction metadata to disk.
- Do not implement or publish the occurrence-location audit without explicit authorization.
- Derive the Phase 1 Unicode character inventory from training data only; do not define a token vocabulary in Phase 1.
- Audit and report validation/test-only Unicode code points under the sealed-test diagnostic contract without discarding, replacing, normalizing, or adding them.
- Do not implement tokenization, neural networks, Transformers, training, or inference yet.
- Do not use a pretrained language model or external model API as the model implementation.
- Do not skip educational phases or silently change the model architecture.
- Preserve reproducibility and keep core machine-learning code understandable.

## Open questions

- Does Sebastien authorize the occurrence-location audit?

## Session handoff

Read-only preflight, deterministic in-memory extraction, and the in-memory Unicode character inventory are accepted and tracked in dedicated checkpoints. Occurrence-location audit and filesystem publication remain unauthorized and unimplemented. Tokenization has not started, and Phase 2 has not begun.
