# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 9 — Generation and Evaluation (accepted contract remotely published at
`48357be6f717e7d4e0445b96ca9886f122673a83`; implementation and final
sample-stage correction independently accepted; accepted implementation
checkpoint remotely preserved at
`41204d9dabd811d630ee82f49b7bcd3fd6b12d5a`; accepted development
pre-registration remotely preserved at
`f6dd0a57c61ecfaa008dda4a40fc7645b14c0f21`; development authorization remotely
preserved at `680949233c95c266a8ff790452441ddec821b580`; one governed
development attempt completed with immutable evidence; completed result record
independently reviewed, explicitly accepted, and committed locally; the result
checkpoint remains unpushed pending independent post-commit verification;
sealed planning and execution remain unauthorized)

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
  with one expected restricted-context MPS skip. The exact 11-file checkpoint
  was committed, pushed unchanged, and live-remote verified at
  `41204d9dabd811d630ee82f49b7bcd3fd6b12d5a`.
- Read-only real-execution readiness inspection established the exact
  development-before-sealed lifecycle and found no implementation defect. It
  also confirmed that a later execution process must first establish the
  accepted deterministic runtime envelope without weakening it.
- Sebastien authorized documentation-only development pre-registration
  drafting. Unused ID `EXP-20260913-01` now has one exact development plan in
  `EXPERIMENT_LOG.md`; it freezes the accepted contract, implementation,
  checkpoint, runtime, dataset, tokenizer, model, evaluation, sample, marker,
  evidence, sealed-exclusion, and score-independent success authorities.
- Independent Codex review returned `PASS — READY FOR PHASE 9 DEVELOPMENT
  PRE-REGISTRATION ACCEPTANCE`, with no BLOCKER, MAJOR, or MINOR finding.
  Sebastien accepted the exact plan at SHA-256
  `582025519d994f74363831128af7ef4851c2cd5e41fdfaa2af0ba0a31516f0c9`
  and authorized its three-file acceptance commit. Commit
  `13bedfa917a386501ce90783c080bf179ccf263d`, `Pre-register Phase 9 development
  evaluation`, was created locally and not pushed.
- Independent post-commit review returned `FAIL — DO NOT PUSH` with one MAJOR
  governance-only finding: `PROJECT_STATE.md` and `ROADMAP.md` still described
  the already accepted/committed pre-registration as an uncommitted draft
  awaiting initial review. The accepted `EXPERIMENT_LOG.md` plan remains
  byte-for-byte correct and is not reopened.
- Sebastien authorized only a two-file documentation correction draft for that
  finding. Incorporating it into the accepted local checkpoint, post-amend
  verification, and push/live verification remain separate later gates.
- Focused independent re-review returned `PASS — GOVERNANCE CORRECTION READY
  FOR INCORPORATION`, with no BLOCKER, MAJOR, or MINOR finding. Sebastien then
  authorized its incorporation into the unpushed checkpoint by amend.
- The reviewed correction was incorporated into amended commit
  `fd0890e38e24907e740a71c5d933febe4c1ce38e`, retaining parent
  `41204d9dabd811d630ee82f49b7bcd3fd6b12d5a`, the title `Pre-register Phase 9
  development evaluation`, and the exact accepted plan bytes.
- Final post-amend review returned `FAIL — DO NOT PUSH` with one remaining
  MAJOR governance finding: present-state wording still named superseded
  pre-amend commit `13bedfa917a386501ce90783c080bf179ccf263d` as the current
  checkpoint. The finding does not reopen the accepted plan. Sebastien
  authorized only this final two-file self-reference correction draft.
- Focused independent review passed that self-reference correction, and
  Sebastien authorized its incorporation. Amended checkpoint
  `9408211609e2f19089c519acd77f091d2866fe10` retained the same parent, title,
  three-file scope, and accepted plan bytes.
- The latest independent verification returned `FAIL — DO NOT PUSH` with one
  MAJOR governance finding: current-state sections still encoded ephemeral
  correction, worktree, review, and amendment mechanics. The accepted plan
  remained unchanged. Sebastien authorized a narrow durable-governance wording
  correction that preserves history while keeping transient mechanics out of
  current state.
- Independent review passed the durable-governance correction with no remaining
  finding. It was incorporated into the accepted pre-registration checkpoint,
  whose complete committed state then passed final independent verification.
  Commit `f6dd0a57c61ecfaa008dda4a40fc7645b14c0f21`, `Pre-register Phase 9
  development evaluation`, was pushed and live-remote verified unchanged.
- Sebastien authorized documentation-only drafting of the exact same-ID
  development-authorization record. The proposed record is not effective
  execution authority unless it later passes independent review, is explicitly
  accepted, committed, pushed, and live-verified.
- Independent Codex review returned `PASS — READY FOR PHASE 9 DEVELOPMENT-
  AUTHORIZATION ACCEPTANCE`, with no BLOCKER, MAJOR, or MINOR finding. Sebastien
  accepted the exact authorization record at SHA-256
  `3e3a180a3ec1a80c5e3e0d7ab4294965ab513fe31af02b8f558d797f0d4702a2`
  and authorized its local acceptance checkpoint. Acceptance does not make the
  authorization effective before push and live verification.
- Authorization commit `680949233c95c266a8ff790452441ddec821b580`, `Authorize
  Phase 9 development evaluation`, passed post-commit review, was pushed, and
  was live-remote verified. Its exact plan and authorization hashes remained
  unchanged, establishing effective authority for a separately authorized run.
- Sebastien authorized exactly one governed development attempt for
  `EXP-20260913-01`. A fresh dedicated process established the accepted runtime
  and invoked the accepted runner once. Durable marker SHA-256 is
  `5262543c930d06f9764ada8a19b7b441f0b1d2195b96d71f0b3df83a06139f6e`;
  completed evidence SHA-256 is
  `d1485cc1a84b26aaa072f267a278963b3c8e25085bee5c9e58665f5a267c9bf0`.
  The attempt completed, the ID is permanently consumed, failure evidence is
  absent, and sealed-test access remained `none`.
- Independent Codex evidence review returned `PASS — DEVELOPMENT EVIDENCE READY
  FOR RESULT RECORD`, with no BLOCKER, MAJOR, or MINOR finding. Sebastien
  authorized only documentation drafting of the exact completed development
  result; result acceptance and sealed-test planning remained unauthorized at
  that drafting gate.
- Sebastien supplied the subsequent independent Codex result-review authority:
  `PASS — READY FOR PHASE 9 DEVELOPMENT-RESULT ACCEPTANCE`, with no BLOCKER,
  MAJOR, or MINOR findings. On 2026-09-15, Sebastien explicitly accepted the
  exact completed `EXP-20260913-01` development result at SHA-256
  `1cceb5af08a9090b22db39b0064a0a104778fa280aab9f22b378d85f99e6381c`,
  classification `completed`, development predicate PASS. That acceptance gate
  authorized governance reconciliation only. The accepted record retains its
  frozen `Next gate: independent_development_result_review` value.
- Sebastien separately authorized the exact three-file local acceptance
  checkpoint: `EXPERIMENT_LOG.md`, `PROJECT_STATE.md`, and `ROADMAP.md`.
  The accepted result and reconciled governance are committed locally under
  `Record accepted Phase 9 development result`, with development-authorization
  commit `680949233c95c266a8ff790452441ddec821b580` as parent. The checkpoint
  remains unpushed; independent post-commit verification is required before
  any separately authorized push/live verification. No sealed work is authorized.

## Current work

The accepted `EXP-20260913-01` development pre-registration is remotely
preserved at `f6dd0a57c61ecfaa008dda4a40fc7645b14c0f21`, with its plan
unchanged. Its accepted development authorization is remotely preserved at
`680949233c95c266a8ff790452441ddec821b580`. The authorized development
attempt executed exactly once, completed successfully, and is permanently
consumed. Its unchanged immutable marker/evidence passed independent review.
The exact completed development-result record subsequently passed independent
review and is explicitly accepted. The accepted result and reconciled
governance are committed locally in the current unpushed acceptance checkpoint.
Independent post-commit verification is required before separately authorized
push/live verification. The attempt may never be rerun and its
artifacts may never be modified. No sealed plan or authorization exists;
sealed planning/access remain unauthorized and *Twelfth Night* remains sealed.
Phase 9 remains open.

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

- Branch `main`; the current local development-result acceptance checkpoint
  has parent `680949233c95c266a8ff790452441ddec821b580`. Live `origin/main`
  remains at that development-authorization commit; local `main` is ahead one
  commit and behind zero, with a clean worktree and index.
- Focused Phase 9 synthetic tests pass 94/94. Relevant tokenizer, vocabulary,
  MiniGPT, and Phase 8 checkpoint regressions pass 119/119. The complete suite
  runs 632 tests with 631 passes, one expected restricted-context MPS skip, and
  zero failures.
- Corrected specification SHA-256 is
  `61489c3a146d6fa6d3ff6a59f6052ff978ae283d42520c599bd1a25b14a71780`;
  the accepted specification remains unmodified and `git diff --check` passes.
- The accepted Phase 8 result record and all tracked authority identities remain
  exact; no retained checkpoint, catalog, corpus, or evidence bytes were opened
  or modified during pre-registration drafting.
- The accepted `EXP-20260913-01` plan remains byte-for-byte unchanged at
  SHA-256 `582025519d994f74363831128af7ef4851c2cd5e41fdfaa2af0ba0a31516f0c9`.
- Acceptance reconciliation preserves the authorization at SHA-256
  `3e3a180a3ec1a80c5e3e0d7ab4294965ab513fe31af02b8f558d797f0d4702a2`
  and the accepted result at SHA-256
  `1cceb5af08a9090b22db39b0064a0a104778fa280aab9f22b378d85f99e6381c`.
  Safe parser, canonical rendering, 25-label completed-result validation,
  record-chain validation, and result/evidence bindings pass. Failure evidence
  remains absent; no evaluation or generation was repeated.
- The immutable attempt marker and completed evidence remain exact at SHA-256
  `5262543c930d06f9764ada8a19b7b441f0b1d2195b96d71f0b3df83a06139f6e`
  and `d1485cc1a84b26aaa072f267a278963b3c8e25085bee5c9e58665f5a267c9bf0`.
- The governed development run accessed only the six training works and *The
  Tempest*, completed the contracted metrics, baselines, and six fixed samples,
  and performed no sealed-test access.

The acceptance checkpoint records the existing result and its governance; it
makes no new runtime or model-quality claim.

## Next exact step

Obtain independent post-commit verification of the complete current local
`EXP-20260913-01` development-result acceptance checkpoint.

## Remaining workflow

Obtain independent post-commit verification of the local acceptance checkpoint.
Only on PASS may separate authorization permit push and live-remote verification.
Only afterward may an optional sealed-test plan be considered under its own
authorization, review, acceptance, publication, and explicit one-shot access
gates. Result acceptance itself authorizes no sealed work. Phase 9 exit
acceptance and closure remain later gates.

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
- The ordinary interactive runtime is not the accepted execution envelope;
  later authorized execution must use a fresh dedicated process with the exact
  frozen CPU/determinism/thread/MKLDNN identity before runner preflight.
- The completed development result is independently reviewed and explicitly
  accepted and committed locally. Remote publication remains pending:
  independent post-commit PASS and separately authorized push/live verification
  are required before any separately authorized sealed planning.

## Important constraints

- Do not modify the accepted Phase 7 model, DEC-0020, the accepted Phase 8
  specification/source/tests, `EXP-20260912-01`, or retained Phase 8 artifacts.
- Do not load a checkpoint, construct the trained model, generate, sample,
  calculate Phase 9 metrics/baselines, or create evaluation artifacts before
  the applicable later authorization.
- Do not access *Twelfth Night* or the sealed test without a separate explicit
  one-shot authorization after the complete evaluation authority is frozen.
- Do not rerun `EXP-20260913-01` or alter its accepted result or marker/evidence.
  Do not push or use the accepted result for sealed planning without the
  applicable later authorization and verification gates.
- Keep exploratory generation distinct from accepted evaluation evidence and
  do not silently change the accepted conceptual architecture.

## Open questions

No model-architecture or detailed-contract choice remains open. The consolidated
contract-level policy package, implementation, and completed development result
are accepted. Result publication and the optional sealed-test decision remain
separate gates.

## Session handoff

Phase 8 is formally remotely closed at `6d086625`. The exact accepted Phase 9
contract is remotely published at `48357be6`; the accepted implementation is
remotely preserved at `41204d9`. Original accepted local pre-registration
commit `13bedfa` failed post-commit review for stale governance; its reviewed
correction was incorporated into `fd0890e`, whose final post-amend review found
one remaining current-state self-reference issue. Its correction was
incorporated into `940821`, whose independent verification found that current
state still encoded ephemeral workflow mechanics. The durable correction and
complete checkpoint then passed independent verification, and accepted
pre-registration `EXP-20260913-01` is remotely preserved at `f6dd0a5` with
unchanged plan SHA-256
`582025519d994f74363831128af7ef4851c2cd5e41fdfaa2af0ba0a31516f0c9`.
The exact same-ID development-authorization record is independently reviewed
and accepted; remote authority is `6809492`. The one separately authorized
development attempt completed without retry. Immutable marker/evidence hashes
are `5262543c...39f6e` and `d1485cc1...c9bf0`, and independent evidence review
passed. Subsequent independent result review returned
`PASS — READY FOR PHASE 9 DEVELOPMENT-RESULT ACCEPTANCE`, with no BLOCKER,
MAJOR, or MINOR findings. Sebastien explicitly accepted result SHA-256
`1cceb5af08a9090b22db39b0064a0a104778fa280aab9f22b378d85f99e6381c`.
The accepted result bytes and immutable artifacts are unchanged. The accepted
result and reconciled governance are committed locally in the current unpushed
acceptance checkpoint. Its parent and live `origin/main` remain `6809492`;
`main` is ahead one commit and behind zero, with a clean worktree and index.
Next: obtain independent post-commit verification of the complete checkpoint;
require PASS before separately authorized push/live verification.
No sealed plan or authorization exists; sealed planning/access
remain unauthorized, *Twelfth Night* remains sealed, and Phase 9 remains open.
