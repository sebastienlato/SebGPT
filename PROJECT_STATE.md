# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 1 — Dataset

## Current milestone

Dataset specification approved; raw-source acquisition and structural inspection authorized.

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

## Current work

Creating the dedicated dataset-design checkpoint before acquiring the raw master.

## Current model status

None. No model code exists.

## Last verified working state

The Phase 0 environment remains verified. The approved Phase 1 corpus design and corrected acquisition contract document the source, eight-play manifest, whole-work splits, raw-source preservation, authoritative provenance record, raw/processed boundary, mechanical character semantics, leakage rules, training-only character inventory, and gated acquisition procedure. No dataset file has been accessed or downloaded, and no acquisition, extraction, preprocessing, tokenization, or model implementation exists.

## Next exact step

Commit the approved dataset-design checkpoint, then acquire and verify only the unmodified Project Gutenberg eBook #100 raw master under the approved contract.

## Known issues

- PyTorch emits a warning that optional NumPy interoperability is unavailable. NumPy is intentionally not installed because no current milestone requires it.
- MPS is unavailable inside some restricted execution contexts, but direct host verification and the unrestricted test suite both pass.
- The current artifact URL, source byte count, source hash, and source-side metadata cannot be recorded until acquisition is approved.
- Exact play boundary markers and measured corpus sizes cannot be verified until the approved raw source is acquired and inspected.
- Validation/test-only Unicode code points are intentionally unresolved until the required audit is run and Phase 2 chooses a generic unseen-input policy.

## Important constraints

- Acquire only the exact raw source authorized by `docs/DATASET_SPEC.md`.
- Do not add dependencies without an approved, recorded reason and lock-file update.
- Do not extract, normalize, split, or tokenize text during the acquisition milestone.
- Derive the Phase 1 Unicode character inventory from training data only; do not define a token vocabulary in Phase 1.
- Audit and report validation/test-only Unicode code points under the sealed-test diagnostic contract without discarding, replacing, normalizing, or adding them.
- Do not implement tokenization, neural networks, Transformers, training, or inference yet.
- Do not use a pretrained language model or external model API as the model implementation.
- Do not skip educational phases or silently change the model architecture.
- Preserve reproducibility and keep core machine-learning code understandable.

## Open questions

- None for the approved acquisition-only milestone.

## Session handoff

The Phase 1 corpus design and acquisition contract are approved. First commit the dedicated design checkpoint; then acquire and fingerprint only the unmodified Project Gutenberg source, populate the tracked manifest, preserve the raw bytes, inspect only permitted structural markers, and stop again before extraction design or implementation.
