# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 9 — Generation and Evaluation (accepted conceptual architecture;
exact detailed contract explicitly accepted after completed independent review;
documentation/contract checkpoint publication authorized; implementation
unauthorized)

## Current milestone

Phases 0–8 are formally remotely closed. Phase 8 closure commit is
`6d086625cb293f61700bd59b551ea310a5b680b9`, `Complete Phase 8 training and
checkpointing`. Its accepted result commit is
`334119e63c716e4922cd0b67da4f5fc221faa886`; accepted implementation authority
is `809834323d53407cb4a54ae539585bb3d78856eb`; corrected provenance contract
and implementation commits are `c50d77ac935bdf924b9b429a5776c419982a5d11`
and `3e0b9c69963702718446901b2fb85e4f5e38aea9`.

Sebastien accepted the Phase 9 learning/design continuity inspection and four
consolidated conceptual directions. DEC-0021 records a Phase 9-owned read-only
`best_validation` inference boundary; transparent rolling-context generation;
greedy and local-generator temperature/top-k categorical decoding; and
validation-led quantitative, baseline, qualitative, and separately gated
sealed-test evaluation.

## Completed work

- Phase 8 Gates 89–93 are reconciled with their completed outcome: the exact
  five-file closure commit was authorized, created, pushed, and independently
  remote-verified at `6d086625cb293f61700bd59b551ea310a5b680b9`.
- Phase 8 experiment `EXP-20260912-01` remains accepted PASS. Final training
  loss is `2.4611534265660575`; final and best validation loss is
  `2.4967258539791177` at epoch `10`; checkpoint/catalog validation and exact
  resume passed.
- The retained epoch-10 object SHA-256 is
  `6990c89166d48732b0601aeae1edd57a9f0e22eec1c7c1160d5be1b8281c9148`.
  It occupies distinct canonical `latest` and `best_validation` roles. Catalog
  SHA-256 is
  `6d590f0355ec950d770e77b4c542d65cda15012fbea56349b97b154a0d614241`.
- The Phase 9 continuity inspection established the exact roadmap goal and exit
  criteria, inherited authority, concepts, genuine design choices, and the
  inference-loader gap without loading a checkpoint, constructing a model,
  generating, evaluating, or accessing the sealed test.
- Sebastien accepted the four consolidated Phase 9 conceptual directions, now
  recorded in DEC-0021 without implementation-level mechanics.
- Sebastien authorized documentation-only detailed-contract drafting.
  Proposed `docs/GENERATION_EVALUATION_SPEC.md` defines exact authority,
  interfaces, loading, generation, sampling, evaluation, baselines, qualitative
  evidence, artifacts, sealed-test protection, errors, tests, exit mapping, and
  governance.
- Fresh independent contract review returned `REQUIRES CORRECTION` with no
  BLOCKER, five MAJOR findings, and four MINOR findings. It independently
  accepted the proposal's core generation, sampling, window, metric, uniform,
  Laplace/add-one, and production-count mathematics.
- The authorized focused correction resolves deterministic evaluation-ID and
  record selection, pure-mechanics/production-runner separation, durable
  development attempt lifecycle, exhaustive inherited executable-byte
  protection, exact evidence/failure schemas, cross-predictor ownership,
  baseline naming/rationale, workflow wording, and architecture-versus-contract
  policy status. Corrected proposal SHA-256 is
  `8c2eafd9ed1f4a86a80bafa11000fd1b62a68655382c6a824822789d9b5bdafc`.
- Focused re-review accepted three prior MAJOR corrections and left only nested
  result-reference/predecessor mechanics, import-closure completeness,
  publication-state consistency, and one wording issue. The final authorized
  documentation correction adds exact artifact/baseline/sample references,
  bounded predecessor traversal, the two missing package initializers, one
  unified four-state publication model, sealed-marker ambiguity consumption,
  terminal mutual exclusion, and corrected first-failure language. Final
  proposal SHA-256 is
  `6625e6e045c80dedf17ea344384c9b4a3a325b6761e65e4c796b134a547c1e83`.
- Final focused review returned `ACCEPT WITH MINOR CORRECTIONS`, with no
  BLOCKER and no remaining/new MAJOR finding. The sole minor correction removes
  unreachable `attempt_marker_publication` from development failure-artifact
  stages while preserving all marker lifecycle semantics. No independent
  re-review remains required. Final specification SHA-256 is
  `61489c3a146d6fa6d3ff6a59f6052ff978ae283d42520c599bd1a25b14a71780`.
- Sebastien explicitly accepted the exact 2,371-line Phase 9 detailed contract
  at that SHA-256 and authorized one six-file documentation checkpoint titled
  `Define Phase 9 generation and evaluation contract`. Contract acceptance and
  publication do not authorize implementation.

## Current work

The accepted continuity/conceptual documentation and proposed detailed contract
are applied locally in `DECISIONS.md`, `LEARNING_NOTES.md`, `PROJECT_STATE.md`,
`README.md`, `ROADMAP.md`, and new
`docs/GENERATION_EVALUATION_SPEC.md`. Tracked edits remain unstaged, the new
specification remains untracked, and everything is uncommitted and unpushed.
Independent review and explicit acceptance of the final detailed contract are
complete. The exact six-file documentation checkpoint is authorized for
publication. No implementation or execution is authorized.

## Current model status

The accepted trained MiniGPT remains unchanged: one CPU-long rank-one sequence
of length `1..256` produces CPU-float32 logits `(T, 81)`. The model has four
causal Transformer blocks, 90 unique trainable tensors, 63,825 parameters,
eight local dropout generators, final normalization, and an untied biased head.

Phase 9 will later select the canonical catalog's semantic `best_validation`
role through a separately specified read-only inference loader. The accepted
Phase 8 public loader remains a latest-only training-continuation interface and
is not modified by DEC-0021.

## Last verified working state

- Branch `main`, local `HEAD`, `origin/main`, and last queried live remote
  `main` equal Phase 8 closure `6d086625cb293f61700bd59b551ea310a5b680b9`
  at local ahead/behind `0/0`.
- The index is empty. The only worktree changes are the six authorized
  documentation paths; no source, test, checkpoint, experiment record, retained
  artifact, or generated evaluation artifact changed.
- Corrected specification SHA-256 is
  `61489c3a146d6fa6d3ff6a59f6052ff978ae283d42520c599bd1a25b14a71780`;
  its 96 Markdown fences are balanced and `git diff --check` passes.
- Canonical catalog, epoch-10 checkpoint, retained training evidence, and
  `EXPERIMENT_LOG.md` hashes remain exact.
- No checkpoint/model was loaded; no train/validation/test prose was opened; no
  prompt was tokenized; and no generation, baseline fitting, or metric
  calculation occurred.

This correction session does not rerun training or model tests and makes no new
runtime or model-quality claim.

## Next exact step

After successful commit, push, and live-remote verification of this exact
accepted documentation checkpoint, obtain separate explicit Phase 9
implementation authorization.

## Remaining workflow

Publish and independently remote-verify the accepted contract checkpoint; then
separately authorize implementation; independently review, correct, accept,
commit, push, and remote-verify implementation; then pre-register and separately
authorize any development evaluation and optional one-shot sealed-test event
before result review, exit acceptance, and formal Phase 9 closure.

## Known issues

- The Phase 8 public checkpoint loader is intentionally latest-only and restores
  training state; Phase 9 needs the accepted distinct read-only inference
  boundary before it can safely consume `best_validation`.
- Runtime remains a material cost: the accepted Phase 8 run took `29m 23.81s`.
- Optional NumPy interoperability is unavailable and intentionally unnecessary.
- Processed corpus derivatives, checkpoints, and experiment evidence are
  intentionally ignored by Git and must be preserved.
- No independent contract-review finding remains. The final exact API,
  validation, lifecycle, record, schema, publication, evidence, and test
  mechanics are explicitly accepted.

## Important constraints

- Do not modify the accepted Phase 7 model, DEC-0020, the accepted Phase 8
  specification/source/tests, `EXP-20260912-01`, or retained Phase 8 artifacts.
- Do not load a checkpoint, construct the trained model, generate, sample,
  calculate Phase 9 metrics/baselines, or create evaluation artifacts before
  the applicable later authorization.
- Do not access *Twelfth Night* or the sealed test without a separate explicit
  one-shot authorization after the complete evaluation authority is frozen.
- Do not correct or accept the detailed Phase 9 contract, implement source/tests,
  stage, commit, or push without the applicable later authorization.
- Keep exploratory generation distinct from accepted evaluation evidence and
  do not silently change the accepted conceptual architecture.

## Open questions

No model-architecture or detailed-contract choice remains open. The consolidated
contract-level policy package is accepted. Future implementation authorization
and the later optional sealed-test decision remain separate gates.

## Session handoff

Phase 8 is formally remotely closed at `6d086625`. The stale pre-closure
current-state wording has been reconciled without rewriting historical entries.
DEC-0021 records Sebastien's accepted Phase 9 conceptual architecture. Current
changes now include the complete proposed detailed contract and remain
documentation only. Tracked edits are unstaged, the specification is untracked,
and everything is uncommitted and unpushed. No Phase 9 implementation,
checkpoint loading, generation, evaluation, or sealed-test access is authorized.
Final focused review returned `ACCEPT WITH MINOR CORRECTIONS`; the sole narrow
minor edit is applied, no further independent re-review is required, and
Sebastien explicitly accepted the exact contract. The six-file contract
checkpoint is authorized for publication; after successful remote verification,
the next exact step is separate Phase 9 implementation authorization.
