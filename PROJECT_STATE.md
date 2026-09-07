# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 1 — Dataset

## Current milestone

Final expected-result ledger accepted and tracked in its dedicated checkpoint; production publication remains unauthorized.

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
- The unseen-character comparison found zero validation/test code points absent from training; its candidate and occurrence-entry collections are empty, so occurrence-location reporting is not applicable for the pinned corpus.
- The transactional publisher design was accepted and recorded as DEC-0012.
- A standard-library publisher and 26 focused tests were implemented using temporary repositories only; the full 74-test suite passed with the expected restricted-context MPS skip.
- Production `ExtractedWork` compatibility passed through a temporary repository without creating the real `data/processed/` tree.
- Publisher reviews reproduced target-promotion and cleanup-ownership races; exclusive no-clobber promotion and preservation of failed staging state now pass deterministic regression tests for target, top-level staging, nested-entry, and final-root replacement.
- Descriptor-relative exclusive/no-follow directory and file creation prevents staged-path symlinks or substituted intermediate directories from redirecting writes; both construction races now have deterministic regression coverage.
- Darwin feasibility review established that the directory-tree publisher cannot guarantee safety against an unrelated same-user namespace adversary; Sebastien accepted an append-only DEC-0012 amendment limiting guarantees to ordinary workspaces and cooperating locked publishers.
- An empty atomic `data/.publish-lock` presence-file protocol now serializes cooperating publishers, releases after success or clean pre-publication refusal, and is preserved with stale or uncertain state; four focused lock-lifecycle tests pass.
- Final independent review passed with no must-fix issues, and Sebastien accepted the complete publisher implementation and DEC-0012 amendment.
- The authoritative manifest now records publisher revision `cd4cd8159b420920aa66755630fe26f9633a5373`, the accepted environment, exact exclusions, per-work provenance and processed results, split/global aggregates, character-inventory results, duplicate/boundary conclusions, and expected generated-manifest identity.
- A production ledger-verification test and temporary-root publication/rerun verification pass; the publisher suite has 27 tests and the full 75-test suite passes with the expected restricted-context MPS skip.
- Independent review passed with no must-fix issues, and Sebastien accepted the final expected-result ledger checkpoint.

## Current work

None. The final expected-result ledger is accepted; production publication remains unauthorized and unperformed.

## Current model status

None. No model code exists.

## Last verified working state

The accepted preflight, extraction, inventory, zero-candidate audit, and publisher remain intact. The authoritative ledger now matches all production in-memory provenance hashes/counts, split/global aggregates, character-inventory results, and audit conclusions. Its expected deterministic generated manifest has SHA-256 `bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc`. Temporary publication using the exact ledger succeeds and an exact rerun returns `already_current`. The real `data/processed/` tree remains absent, production publication is unauthorized, and no tokenizer, dataset loader, or model implementation exists.

## Next exact step

Sebastien explicitly authorizes production processed-dataset publication before `data/processed/` is created.

## Known issues

- PyTorch emits a warning that optional NumPy interoperability is unavailable. NumPy is intentionally not installed because no current milestone requires it.
- MPS is unavailable inside some restricted execution contexts, but direct host verification and the unrestricted test suite both pass.
- The catalog reports a 2025-08-24 update date, while the artifact response reports `Last-Modified: Tue, 01 Sep 2026 07:55:49 GMT`; both are recorded without assuming they describe the same revision mechanism.
- In-memory processed counts and hashes are reproducibly computed but are not published to dataset files or a processing manifest.
- The final expected-result ledger is accepted and tracked; processed files have not been published.

## Important constraints

- Do not implement processed-dataset filesystem publication or create `data/processed/` until Sebastien explicitly authorizes that separate milestone.
- Do not add dependencies without an approved, recorded reason and lock-file update.
- Do not write extracted text, normalized text, split documents, or extraction metadata to disk.
- Do not implement occurrence-location lookup for the pinned corpus; the accepted audit has zero candidates and requires no location records.
- Derive the Phase 1 Unicode character inventory from training data only; do not define a token vocabulary in Phase 1.
- Audit and report validation/test-only Unicode code points under the sealed-test diagnostic contract without discarding, replacing, normalizing, or adding them.
- Do not implement tokenization, neural networks, Transformers, training, or inference yet.
- Do not use a pretrained language model or external model API as the model implementation.
- Do not skip educational phases or silently change the model architecture.
- Preserve reproducibility and keep core machine-learning code understandable.

## Open questions

- Does Sebastien authorize production processed-dataset publication?

## Session handoff

Read-only preflight, deterministic in-memory extraction, the in-memory Unicode character inventory, the zero-candidate audit, the cooperative-lock publisher, and the final expected-result ledger are accepted and tracked in dedicated checkpoints. Production publication remains unauthorized, the real `data/processed/` tree is absent, tokenization has not started, and Phase 2 has not begun.
