# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 1 — Dataset

## Current milestone

Preflight technical contract approved, committed, and consistency-verified; implementation pending explicit authorization.

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

## Current work

None. Technical blockers are resolved; read-only preflight implementation awaits explicit authorization.

## Current model status

None. No model code exists.

## Last verified working state

The raw master remains preserved at the approved SHA-256. DEC-0010 is accepted, the exact-preflight checkpoint is committed, and all authoritative status records agree that the technical contract is approved while implementation is unauthorized. No preflight code, extractor, processed data, character inventory, tokenizer, dataset loader, or model implementation exists.

## Next exact step

Sebastien explicitly authorizes the read-only preflight implementation layer before any preflight code, extractor, or processed-output work begins.

## Known issues

- PyTorch emits a warning that optional NumPy interoperability is unavailable. NumPy is intentionally not installed because no current milestone requires it.
- MPS is unavailable inside some restricted execution contexts, but direct host verification and the unrestricted test suite both pass.
- The catalog reports a 2025-08-24 update date, while the artifact response reports `Last-Modified: Tue, 01 Sep 2026 07:55:49 GMT`; both are recorded without assuming they describe the same revision mechanism.
- Actual processed counts and hashes remain unknown until the approved extractor is implemented and run.
- Validation/test-only Unicode code points are intentionally unresolved until the required audit is run and Phase 2 chooses a generic unseen-input policy.

## Important constraints

- Do not implement extraction or create processed data until Sebastien explicitly authorizes implementation of the approved contract.
- Do not add dependencies without an approved, recorded reason and lock-file update.
- Do not extract, normalize, split, or tokenize text during the acquisition milestone.
- Derive the Phase 1 Unicode character inventory from training data only; do not define a token vocabulary in Phase 1.
- Audit and report validation/test-only Unicode code points under the sealed-test diagnostic contract without discarding, replacing, normalizing, or adding them.
- Do not implement tokenization, neural networks, Transformers, training, or inference yet.
- Do not use a pretrained language model or external model API as the model implementation.
- Do not skip educational phases or silently change the model architecture.
- Preserve reproducibility and keep core machine-learning code understandable.

## Open questions

- Does Sebastien authorize implementation of the approved read-only preflight contract?

## Session handoff

The exact wrapper, global-contents, ordered-entry, and successor-separator assertions are approved, committed, and consistency-verified. Resume only with explicit authorization for the read-only preflight implementation layer; extraction outputs, character inventories, and tokenization remain unauthorized.
