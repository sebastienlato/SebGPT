# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 9 — Generation and Evaluation (accepted contract remotely published at
`48357be6f717e7d4e0445b96ca9886f122673a83`; implementation and final
sample-stage correction independently accepted; accepted implementation
checkpoint committed locally and not pushed; real development/sealed execution
remains unauthorized)

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
- The accepted six-file Phase 9 contract checkpoint was committed, pushed, and
  remotely verified at `48357be6f717e7d4e0445b96ca9886f122673a83`, `Define
  Phase 9 generation and evaluation contract`.
- Sebastien separately authorized Phase 9 implementation and synthetic testing
  under that exact checkpoint. The new `sebgpt.inference` package implements
  the contract-owned types/errors, independent best-validation checkpoint
  validation, transparent generation/sampling, pure metrics/baselines, and
  governed development/sealed lifecycle boundaries. Synthetic focused tests
  and all inherited regressions pass; no real Phase 9 execution occurred.
- Fresh Work Phase 9.1 independent inspection returned `WORK CORRECTION
  REQUIRED`. It identified incomplete sealed prerequisites, unsafe path-based
  publication, partial record/evidence validation, unrelated-record parser
  overreach, incomplete checkpoint entry and inherited-file validation,
  incomplete post-load checks, direct-window continuity gaps, incorrect runner
  ordering, and missing adversarial tests.
- Sebastien authorized only focused implementation/synthetic-test correction.
  The correction adds complete selected/frozen authority checks, scoped record
  parsing, descriptor-relative no-follow publication and sealed reading,
  entry-identity revalidation, exact evidence validators, content-safe failure
  normalization, direct-window/baseline integrity, exact operation ordering,
  and adversarial synthetic coverage. Acceptance remains pending independent
  re-review.
- Independent Codex review of that correction returned `REQUIRES CORRECTION`
  with no BLOCKER, six MAJOR findings, and one MINOR finding: incomplete runtime/
  payload mirroring, metric/failure cross-field validation, completed-result
  sealed authority, durable unknown-marker consumption, ancestor-directory
  fsync, exact shared MiniGPT structure, and sealed failure-stage attribution.
- Sebastien authorized only those focused implementation/test corrections. The
  second correction mirrors the complete accepted runtime and Phase 8 payload
  envelope, closes evidence arithmetic/failure tables, requires exact completed
  development authority and historical commit introduction, durably appends an
  uncertain sealed result after unknown marker publication, fsyncs every new
  publication ancestor, shares one exact MiniGPT structural oracle, and splits
  sealed open from validation failures. Acceptance remains pending re-review.
- The latest independent focused re-review accepted checkpoint runtime/payload
  validation, the completed-development sealed prerequisite, and sealed
  failure-stage attribution. It returned `REQUIRES CORRECTION` with no BLOCKER
  for four remaining MAJOR findings: open-ended failure-invariant prefixes,
  non-crash-safe unknown-marker consumption, reused-ancestor durability, and
  missing exact module-class validation.
- Sebastien authorized only those four corrections. Phase 9 now uses finite
  exact failure-invariant sets, a cross-process locked deterministic durable
  marker journal plus atomic uncertain-result publication, parent fsync and
  entry revalidation for reused as well as created ancestors, and an exact
  class oracle for every named MiniGPT module. Acceptance remains pending
  focused re-review.
- The latest focused re-review accepted crash/concurrency-safe sealed
  consumption, reused-ancestor durability, and exact MiniGPT module classes. It
  left one MAJOR defect: sample failure stages still used open-ended `sample_`
  prefix acceptance despite exact invariant membership.
- Sebastien authorized only that final narrow correction. One exact six-entry
  sample-stage-to-sample-ID mapping now governs safe-fact selection, rule
  selection, and failure normalization; invented stages and mismatched valid
  pairs fail closed.
- Final independent Codex sample-stage re-review returned `PASS — READY FOR
  PHASE 9 SAMPLE-STAGE ACCEPTANCE`, with no BLOCKER, MAJOR, or MINOR findings.
  It verified all six exact mappings, rejection of invented/suffixed/truncated/
  case-mutated/empty stages and mismatched valid pairs, absence of prefix-based
  acceptance, and satisfaction of obligation 35.
- Sebastien explicitly accepted the reviewed Phase 9 implementation checkpoint
  and authorized its complete local commit. Fresh focused tests pass 94/94,
  inherited regressions pass 119/119, and the complete suite passes 631 tests
  with one expected restricted-context MPS skip. Nothing was pushed.

## Current work

The independently accepted Phase 9 implementation checkpoint is committed
locally on `main` and remains unpushed. No implementation correction or real
evaluation work is in progress. Real checkpoint loading, real train/validation
execution, generation, evidence publication, and sealed-test access remain
unauthorized.

## Current model status

The accepted trained MiniGPT remains unchanged: one CPU-long rank-one sequence
of length `1..256` produces CPU-float32 logits `(T, 81)`. The model has four
causal Transformer blocks, 90 unique trainable tensors, 63,825 parameters,
eight local dropout generators, final normalization, and an untied biased head.

The new Phase 9 loader is designed to select the canonical catalog's semantic
`best_validation` role after later execution authorization. The accepted Phase
8 public loader remains a latest-only training-continuation interface and is
unchanged.

## Last verified working state

- Branch `main`; the accepted implementation checkpoint is the current local
  `HEAD`, one commit ahead of `origin/main` and zero behind. The worktree and
  index are clean; nothing was pushed.
- Focused Phase 9 synthetic tests pass 94/94. Relevant tokenizer, vocabulary,
  MiniGPT, and Phase 8 checkpoint regressions pass 119/119. The complete suite
  runs 632 tests with 631 passes, one expected restricted-context MPS skip, and
  zero failures.
- Corrected specification SHA-256 is
  `61489c3a146d6fa6d3ff6a59f6052ff978ae283d42520c599bd1a25b14a71780`;
  the accepted specification remains unmodified and `git diff --check` passes.
- Canonical catalog, epoch-10 checkpoint, retained training evidence, and
  `EXPERIMENT_LOG.md` hashes remain exact.
- No real checkpoint/model was loaded; no real train/validation/test prose was
  opened; no trained-model generation, real baseline fitting, or real metric
  calculation occurred. Tests used only tracked vocabulary metadata and small
  deterministic synthetic objects.

This implementation session does not rerun training and makes no new runtime or
model-quality claim.

## Next exact step

Obtain separate authorization to push and remotely verify the accepted Phase 9
implementation checkpoint.

## Remaining workflow

Push and remote-verify the accepted implementation checkpoint only when
separately authorized; then pre-register and separately authorize one
development evaluation. Review and accept its result before any optional
one-shot sealed-test authorization, result review, exit acceptance, and formal
Phase 9 closure.

## Known issues

- The Phase 8 public checkpoint loader remains intentionally latest-only and
  byte-identical; the new Phase 9 loader is accepted but cannot be executed
  against real artifacts until the applicable later execution authorization.
- Runtime remains a material cost: the accepted Phase 8 run took `29m 23.81s`.
- Optional NumPy interoperability is unavailable and intentionally unnecessary.
- Processed corpus derivatives, checkpoints, and experiment evidence are
  intentionally ignored by Git and must be preserved.
- No independent contract-review finding remains. The final exact API,
  validation, lifecycle, record, schema, publication, evidence, and test
  mechanics are explicitly accepted.
- No independent Phase 9 implementation-review finding remains.

## Important constraints

- Do not modify the accepted Phase 7 model, DEC-0020, the accepted Phase 8
  specification/source/tests, `EXP-20260912-01`, or retained Phase 8 artifacts.
- Do not load a checkpoint, construct the trained model, generate, sample,
  calculate Phase 9 metrics/baselines, or create evaluation artifacts before
  the applicable later authorization.
- Do not access *Twelfth Night* or the sealed test without a separate explicit
  one-shot authorization after the complete evaluation authority is frozen.
- Do not push, run governed evaluation, or create Phase 9 experiment/evidence
  records without the applicable later authorization.
- Keep exploratory generation distinct from accepted evaluation evidence and
  do not silently change the accepted conceptual architecture.

## Open questions

No model-architecture or detailed-contract choice remains open. The consolidated
contract-level policy package and implementation are accepted. Remote
publication, later real development authorization, and the optional sealed-test
decision remain separate gates.

## Session handoff

Phase 8 is formally remotely closed at `6d086625`. The exact accepted Phase 9
contract is remotely published at `48357be6`. Final independent review returned
`PASS — READY FOR PHASE 9 SAMPLE-STAGE ACCEPTANCE`, and Sebastien accepted the
complete implementation checkpoint. The local implementation commit has 94
focused tests, 119 inherited regression tests, and a 632-test full suite with
zero failures and one expected MPS skip. It is one commit ahead of
`origin/main`, with a clean worktree/index, and remains unpushed. No real
checkpoint/corpus/evaluation execution or sealed access occurred. The next
exact gate is separate push and remote-verification authorization.
