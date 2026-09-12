# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 8 — Training and Checkpointing (implementation remotely accepted; provenance contract corrected and accepted)

## Current milestone

Phase 7 — Complete Mini-GPT is formally remotely closed at
`33d4510421107848c4aa8a6014f4a7b1e391065a`, `Complete Phase 7 Mini-GPT`.
Its accepted detailed-contract checkpoint is
`60b2a9cce55da79ccc9fbd03fad014cb2a939290`, and its accepted implementation
checkpoint is `3139b1736f005fe903e2ea111d91934478b5a683`.

Sebastien explicitly authorized Phase 8 learning/design. The read-only
continuity inspection is complete, and Sebastien accepted the consolidated
Phase 8 conceptual training/checkpointing policy as DEC-0020. Documentation-only
detailed-contract drafting and stale Phase 7 closure reconciliation were then
authorized.

`docs/TRAINING_CHECKPOINTING_SPEC.md` makes DEC-0020 mechanical. Its original
accepted pre-provenance SHA-256 was
`0b3e2a79de7038e2233560c0836101d2f5757f9a5e3c3e89e6ccb62e7cc6fffd`.
Fresh independent review returned CORRECT BEFORE
ACCEPTANCE with three BLOCKER, eight IMPORTANT, and one MINOR finding. Master
Chat accepted all twelve without changing DEC-0020 and authorized only their
targeted documentation correction. Focused re-review passed those corrections
but returned CORRECT AGAIN BEFORE ACCEPTANCE with one remaining BLOCKER, two
IMPORTANT schema/API findings, and no MINOR finding. Master Chat accepted all
three without changing DEC-0020 and authorized this final targeted correction.
Final focused independent re-review returned PASS with all three resolved, no
new or remaining finding, and exact recommendation `ACCEPT CORRECTED CONTRACT`.
Master Chat formally accepted the corrected detailed contract without changing
DEC-0020. The contract checkpoint was committed, pushed, and remotely verified
at `09b2c2e0487a0b7a766655951422471085d943a8`. Master Chat then explicitly
authorized Phase 8 implementation and focused/full synthetic verification.
Gate 25 implementation is complete locally. Fresh independent implementation
review returned five BLOCKER, six IMPORTANT, and one MINOR finding. Master Chat
authorized their targeted correction, which is complete locally under unchanged
DEC-0020 and the accepted contract. Feasibility measurement and real training
remain unauthorized. Focused re-review then returned one remaining BLOCKER and
two IMPORTANT findings. Their final targeted production/evidence correction is
complete locally. Final focused re-review accepted those production corrections
and returned one remaining IMPORTANT evidence-quality gap. Its evidence-only
correction is complete and changed no Phase 8 source.
Final evidence-focused re-review left only obligations 4 and 9 partially
evidenced. Their final test-only correction is complete, found no production
defect, and changed no Phase 8 source.
Final two-obligation verification accepted obligation 9 and left only
obligation 4's static evidence bypassable. Its final test-only correction is
complete, found no production defect, and changed no Phase 8 source.
Final obligation-4-only verification returned PASS with no remaining BLOCKER,
IMPORTANT, or MINOR finding. All 48 obligations are independently satisfied,
and Master Chat formally accepted the corrected Phase 8 implementation.
Implementation checkpoint `8098343` was committed, pushed, and remotely
verified. The bounded feasibility result `FEASIBLE WITH MATERIAL RUNTIME COST`
was accepted. The later fixed-run authorization stopped correctly before work
because pre-registration authority was absent and exposed a genuine
self-referential `code_commit` contract defect. Its documentation correction
passed Gate 46 independent review with no remaining BLOCKER or IMPORTANT
finding and exact recommendation `ACCEPT PROVENANCE CONTRACT CORRECTION`.
Master Chat formally accepted the corrected provenance contract at SHA-256
`1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd` without
changing DEC-0020 or the accepted implementation.

## Completed work

- Phases 0–7 are formally remotely closed.
- Phase 7 closure commit `33d4510`, accepted contract `60b2a9c`, and accepted
  implementation `3139b17` are recorded and synchronized.
- The Phase 8 continuity inspection established inherited authority, open
  choices, checkpoint gaps, and the Phase 8/9 boundary without accessing the
  sealed test.
- DEC-0020 accepts unchanged dataset/tokenizer/MiniGPT authority; CPU float32;
  context 256; logical rank-one batches of eight; deterministic epoch shuffle;
  constant AdamW with learning rate `3e-4`, betas `(0.9, 0.999)`, epsilon
  `1e-8`, and weight decay `0.01`; global-norm clipping at `1.0`; ten epochs
  without validation early stopping; complete initialized/end-epoch evaluation;
  latest and best-validation checkpoints; complete stochastic/progress state;
  exact resume; durable experiment evidence; and no Phase 9 or sealed-test use.
- Accepted `docs/TRAINING_CHECKPOINTING_SPEC.md` defines stride-256 exact-once
  document-local windows with retained unpadded tails, order seed 8001,
  target-token-weighted gradient accumulation, final partial logical batches,
  exact AdamW/update/clipping/evaluation order, ten runtime RNG states, complete
  versioned checkpoint payloads, content-addressed immutable objects, one
  atomic catalog commit point, fail-closed restoration, exact-resume equality,
  experiment records, fitting evidence, numerical/failure behavior, focused
  tests, feasibility boundary, and separate Phase 8 gates.
- Stale Phase 7 remote-closure and Phase 8 authorization wording is reconciled
  in current continuity documentation.
- The authorized detailed-contract correction resolves checkpoint-directory
  durability, caller-global-RNG load transactionality, the frozen runtime
  envelope, exact PyTorch clipping, complete API/record/error/checkpoint/log
  schemas, catalog/payload agreement, the local filesystem trust boundary,
  live repository provenance, latest-only continuation, a complete-epoch real
  resume audit, irreversible gate separation, and deterministic permutation
  evidence. DEC-0020 is unchanged.
- The final targeted correction adds intrinsic supported AdamW
  `decoupled_weight_decay=True` state, exact 90-parameter step-tensor/progress
  equality, literal owning modules and module/package exports for every public
  Phase 8 object, and an unambiguous first-key nested configuration schema
  version. All prior corrections remain unchanged.
- Final focused independent re-review passed. All Phase 8 contract-review
  findings are resolved; no BLOCKER, IMPORTANT, or MINOR finding remains. The
  corrected specification is formally accepted at its exact SHA-256.
- Accepted contract checkpoint `09b2c2e` is pushed and remotely verified.
- Gate 25 implements the exact five Phase 8 modules, package exports, runtime,
  windows/order/batches, AdamW/gradients/evaluation, durable checkpoint/load,
  provenance, lifecycle, record schema, and synthetic exact-resume behavior.
- The targeted implementation correction resolves training-window update
  authority; latest/historical-best catalog semantics; complete non-coercing
  checkpoint schemas; full synthetic epoch-2 exact-resume audit publication and
  reload; runtime/pre-registration binding; post-clip gradients; filesystem
  publication mechanics; epoch-0 placeholder state; retained ignored evidence;
  RNG rollback; focused-test independence; and safe window repr.
- The final targeted correction adds complete literal mode-map validation for
  both latest and historical-best payloads, valid paired-empty logical batching,
  exact mode corruption matrices, direct prior-phase/static-boundary/checkpoint-
  content/public-record evidence, descriptor-specific fsync evidence, and a
  categorized substantive obligation-to-evidence audit.
- The final evidence-only correction adds direct static-boundary, checkpointed-
  ordering, evaluation-failure, publication-failure, restoration-order,
  durable-audit, bootstrap, descriptor-relative, and bounded-read evidence.
- The final two-obligation evidence correction adds a 19-category mutation-tested
  AST boundary oracle across all six modules and deliberate global/dropout RNG
  perturbation around an actual checkpointed ordering-stream round trip.
- The obligation-4 final correction adds qualified/assigned alias resolution,
  bounded string propagation, precise special-token and dtype-string detection,
  cross-module bypass mutations, and false-positive controls.
- Final independent Codex verification returned PASS, all implementation-review
  findings are resolved, and Master Chat formally accepted the exact corrected
  implementation/source/test checkpoint without changing DEC-0020 or the
  accepted Phase 8 specification.
- Accepted implementation checkpoint `8098343` is pushed and remotely verified.
- The bounded feasibility measurement is accepted as `FEASIBLE WITH MATERIAL
  RUNTIME COST`; it changed no policy or repository file and retained no state.
- The accepted pre-run stop exposed and adjudicated the planned-record commit
  self-reference. The accepted provenance correction separates immutable
  implementation `Code commit` `8098343` from the future external
  `pre_registration_commit`, binds launch `HEAD` to the latter, and preserves
  exact implementation ancestry/bytes and planned-record immutability.
- Gate 46 independent review returned PASS / NO ISSUE, confirmed the original
  contradiction was real, found no remaining provenance BLOCKER or IMPORTANT
  finding, and recommended `ACCEPT PROVENANCE CONTRACT CORRECTION`. Master Chat
  formally accepted the corrected provenance contract without changing
  DEC-0020 or accepted Phase 8 production source/tests.
- Focused Phase 8 tests pass 87/87 with all 48 accepted obligations explicitly
  mapped to behavioral/literal evidence. Relevant accepted Phase 3/5/6/7
  regressions pass 174/174. The complete 530-test suite has 529 passes, one expected
  restricted-context MPS skip, and zero failures.

## Current work

Gates 45–48 are complete. The corrected provenance contract is accepted in
`docs/TRAINING_CHECKPOINTING_SPEC.md` at SHA-256
`1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd`.
It is byte-frozen, unstaged, uncommitted, and unpushed pending separate
corrected-contract checkpoint commit authorization.

## Current model status

The accepted Phase 7 MiniGPT remains unchanged: one rank-one CPU-long sequence
of length `1..256` produces CPU-float32 logits `(T, 81)`. It contains 90 unique
trainable tensors and 63,825 parameters, four accepted blocks, eight local
dropout generators, final normalization, and an untied biased head.

Focused tests construct only ephemeral synthetic Phase 8 windows, models,
AdamW instances, gradients, checkpoint payloads/catalogs, temporary filesystem
trees, evaluations, and resume controls. All temporary artifacts are removed.
No retained project checkpoint, real Phase 8 training state, corpus-derived
window/tensor, generated sample, or Phase 7 model instance exists. The accepted
Phase 4 trained instance was not retained.

## Last verified working state

Immediately before this documentation correction, local `HEAD`, `main`,
`origin/main`, and actual remote `main` matched accepted implementation commit
`8098343` at ahead/behind `0/0` with a clean worktree/index.

Latest behavioral verification remains: Phase 8 focused tests pass 87/87;
accepted Phase 3/5/6/7 regressions pass 174/174; and the complete suite runs 530 tests with
529 passes, one expected restricted-context MPS skip, and zero failures. The
accepted provenance-corrected specification remains byte-identical at SHA-256
`1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd`; DEC-0020,
accepted Phase 8 production source/tests, and accepted Phase 7 identities remain
unchanged. Acceptance-bookkeeping `git diff --check` passes.

## Next exact step

Obtain separate Gate 49 corrected-contract checkpoint commit authorization from
Master Chat.

## Known issues

- Accepted feasibility is `FEASIBLE WITH MATERIAL RUNTIME COST`: training epoch
  approximately 78.44 seconds, full training evaluation approximately 58.69
  seconds, validation evaluation approximately 7.28 seconds, complete epoch
  cycle approximately 144.41 seconds, accepted runner including exact-resume
  audit branch approximately 27.58 minutes compute excluding checkpoint I/O,
  and measured peak RSS approximately 336.6 MB.
- Optional NumPy interoperability is unavailable and intentionally unnecessary.
- MPS may be unavailable in restricted contexts; Phase 8 is CPU-only regardless.
- Processed corpus files are ignored reproducible derivatives.
- Checkpoint binaries are excluded from Git by accepted policy.
- Content-addressed checkpoint objects may be orphaned by a failure before the
  atomic catalog replacement; the accepted contract preserves them for
  diagnosis rather than deleting uncertain state.

## Important constraints

- DEC-0020 remains unchanged. The provenance-corrected detailed contract at
  SHA-256 `1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd` is
  accepted and byte-frozen pending separate checkpoint commit authorization.
- The Phase 8 implementation is accepted and remotely synchronized at
  `8098343`; do not alter source/tests without separate authorization.
- Do not change context 256, logical batch capacity 8, CPU float32, model
  architecture, optimizer, or training policy without a new explicit decision.
- Preserve the accepted tokenizer/vocabulary identity and all accepted Phase
  3, 5, 6, and 7 source, tests, contracts, and behavior.
- Never access, tokenize, window, score, inspect, or derive statistics from the
  sealed test work.
- Do not rerun or tune `EXP-20260909-01`.
- Do not implement the production provenance correction or change Phase 8
  source/tests without separate authorization.
- Do not stage, commit, or push without the applicable separate authorization.
- Do not rerun feasibility; create pre-registration; perform real training or
  the real-run audit; retain experiment checkpoints; generate; or sample.
- Do not begin Phase 9 generation or final evaluation.

## Open questions

- No Phase 8 detailed-contract review finding remains unresolved.
- No conceptual DEC-0020 question remains open.
- No provenance-contract BLOCKER or IMPORTANT finding remains; Gate 46 review
  and Master Chat acceptance are complete.
- Runtime feasibility is accepted as `FEASIBLE WITH MATERIAL RUNTIME COST`.

## Session handoff

Phase 7 is formally remotely closed at `33d4510`; accepted Phase 7 contract and
implementation checkpoints remain `60b2a9c` and `3139b17`. Phase 8
learning/design is authorized, its continuity inspection is complete, and
DEC-0020 records the accepted consolidated policy. Accepted
`docs/TRAINING_CHECKPOINTING_SPEC.md` defines exact windows, deterministic
ordering, rank-one logical batches, token-weighted gradients, AdamW, clipping,
evaluation, progress, the frozen runtime, all RNG state, crash-durable atomic
checkpointing, transactional load, exact schemas, latest-only continuation,
complete-epoch exact resume, experiment evidence, failures, tests, feasibility
limits, and 74 gates. Independent review returned three BLOCKER, eight
IMPORTANT, and one MINOR finding; all targeted documentation corrections are
complete without changing DEC-0020. Focused re-review then found one remaining
AdamW-state BLOCKER and two ownership/configuration-schema IMPORTANT findings;
their final targeted corrections passed final focused re-review with no new or
remaining finding. Master Chat accepted the pre-provenance corrected detailed contract at SHA-256
`0b3e2a79de7038e2233560c0836101d2f5757f9a5e3c3e89e6ccb62e7cc6fffd` without
changing DEC-0020. Contract checkpoint `09b2c2e` is pushed and remotely verified;
Master Chat authorized implementation, and Gate 25 produced five source modules,
exact package exports, and 48 initial focused tests. Fresh independent review
then returned five BLOCKER, six IMPORTANT, and one MINOR finding; their targeted
correction is complete without changing DEC-0020 or the accepted contract.
Focused re-review then found one remaining historical-best mode-map BLOCKER and
two IMPORTANT zero-window/evidence findings. Their final correction is complete:
both payload roles validate the exact 53-entry mode map before cross-validation,
paired empty batching yields no batches, and every accepted obligation maps to
categorized substantive evidence. Final focused re-review accepted those
production behaviors and found one remaining evidence-quality gap. The
evidence-only correction changed no source and added direct matrices for static
exclusions, ordering restoration, evaluation/publication failures, restoration
order, durable audit roots, bootstrap, descriptor safety, and bounded reads.
Evidence re-review then left only static-boundary mutation coverage and
unrelated-RNG ordering perturbation incomplete. Their final test-only correction
changed no source and directly resolves both obligations. Final verification
accepted obligation 9 and left only obligation 4's analyzer bypassable. The
alias-aware, constant-propagating, mutation-tested final correction resolves
those bypasses while preserving safe negative controls. Final independent
verification returned PASS with no remaining BLOCKER, IMPORTANT, or MINOR
finding and exact recommendation `ACCEPT CORRECTED IMPLEMENTATION`; Master Chat
formally accepted the implementation. Focused tests pass 87/87, accepted Phase
3/5/6/7 regressions pass 174/174, and the complete 530-test suite has 529
passes, one expected MPS skip, and zero failures. Synthetic checkpoint objects
remain temporary. No feasibility benchmark, real training, real-run audit,
retained project checkpoint, generation, sealed test, staging, commit, or push
occurred. The sole next action is separate implementation-checkpoint commit
authorization from Master Chat.

Implementation checkpoint `8098343` was subsequently committed, pushed, and
remotely verified. Accepted bounded feasibility is `FEASIBLE WITH MATERIAL
RUNTIME COST` at an estimated 27.58 minutes compute excluding checkpoint I/O.
The attempted run authorization correctly stopped before corpus/model/training
because the required planned record did not exist; adjudication and Gate 46
review confirmed the contract's self-referential Git authority contradiction
was real. Gate 46 returned PASS / NO ISSUE with no remaining BLOCKER or
IMPORTANT provenance finding and exact recommendation
`ACCEPT PROVENANCE CONTRACT CORRECTION`; Master Chat formally accepted it.
Planned/checkpoint/result `Code commit` is immutable accepted implementation
anchor `8098343`; the future planned-record commit is separate external
`pre_registration_commit` authority after its own commit/push/remote verification;
and run-launch `HEAD` must equal that authority while exact accepted code/test/
specification identities and ancestry remain enforced. The accepted corrected
specification SHA-256 is
`1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd`.
DEC-0020 and accepted Phase 8 production source/tests remain unchanged. No
production provenance implementation correction, pre-registration record,
training, evaluation, checkpoint, audit, generation, sealed-test access,
staging, commit, or push occurred or is authorized. The sole next action is
separate Gate 49 corrected-contract checkpoint commit authorization from Master
Chat.
