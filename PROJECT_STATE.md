# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 6 — Transformer Block (technically COMPLETE; closure bookkeeping local)

## Current milestone

Phase 5 closure commit `6519d8c813b7e5bc899b73d6b1fa8c38a6819750`,
`Complete Phase 5 self-attention`, is pushed and independently remote-verified.
Local `HEAD`, `origin/main`, and actual remote `main` were verified at that exact
commit with ahead/behind `0/0` and a clean worktree/index. All four Phase 5 exit
criteria remain satisfied, and Phase 5 is formally remotely closed. Sebastien
subsequently authorized Phase 6 learning/design only. The read-only continuity
inspection is complete and accepted. Sebastien completed the learning/design
discussion, accepted the consolidated conceptual architecture in DEC-0018, and
authorized a documentation-only detailed contract. The proposed contract is now
drafted in `docs/TRANSFORMER_BLOCK_SPEC.md`. Master Chat sanity review returned
PASS; fresh Codex independent review returned FAIL with four MUST-FIX and one
SHOULD-FIX finding; and Master Chat accepted all five without changing DEC-0018.
The authorized documentation-only corrections completed Gate 9. Gate 10
focused re-review returned PASS with all
four MUST-FIX and the SHOULD-FIX finding resolved. Master Chat adjudicated PASS,
Sebastien explicitly accepted the corrected detailed contract without changing
DEC-0018. Accepted-contract checkpoint `2177c7e9aaf7a12790afb1fd25693723ac759170`
was committed, pushed, and independently remote-verified with ahead/behind
`0/0`. Sebastien then explicitly authorized Phase 6 implementation. Gate 16
implementation and focused synthetic testing are complete locally and await
Master Chat sanity review and fresh independent review. Independent review
returned FAIL with three MUST-FIX findings accepted by Master Chat: one narrow
standalone-dropout validation defect and two test-evidence gaps. The authorized
focused source/test corrections are complete locally without contract or
DEC-0018 changes. Gate 20 focused independent re-review returned PASS with all
three findings resolved, Master Chat adjudicated PASS, and Sebastien explicitly
accepted the corrected implementation and evidence suite. The dedicated
accepted-implementation checkpoint commit is authorized; push remains separate.
Accepted implementation checkpoint `c2624078eae5947e505d7f8093869e32c58e521b`
was then pushed and independently remote-verified with local `HEAD`, refreshed
`origin/main`, and actual remote `main` equal at ahead/behind `0/0`. The
educational inspection completed, independent Phase 6 exit review returned PASS
with no MUST-FIX findings, Master Chat adjudicated PASS, and Sebastien accepted
all three exit criteria and Phase 6 as technically complete. Documentation-only
closure bookkeeping is complete locally.

## Completed work

- Phases 0–3 are complete and remotely verified.
- Phase 4 is accepted and remotely closed at `e8b5c55`; do not reopen it.
- `EXP-20260909-01` is the sole completed Phase 4 experiment and passed both
  frozen training-loss predicates. Validation improvement remains observation
  only and is not evidence of generalization or test performance.
- Gate 30 explicitly authorized Phase 5 learning/design only.
- Sebastien accepted standalone width-32, single-sequence causal attention with
  one bias-free full-width head and four separately visible bias-free width-8
  heads followed by a bias-free width-32 output projection.
- Causal visibility is exactly `j <= i`; normal forward use returns contextual
  representations, and separate inspection exposes the exact post-softmax
  weights used for the output.
- DEC-0017 records the accepted architecture and
  `docs/SELF_ATTENTION_SPEC.md` proposes detailed mechanics and gates.
- Gate 6 Master Chat sanity review returned PASS; Gate 7 independent contract
  review returned FAIL, with all eight findings accepted for correction.
- Gate 8 resolved all five MUST-FIX and all three SHOULD-FIX findings; Gate 9
  focused re-review and Master Chat adjudication returned PASS.
- Sebastien explicitly accepted the corrected contract without changing
  DEC-0017. Its detailed mechanics are authoritative for later separately
  authorized implementation.
- The accepted contract checkpoint `9443dec` is pushed and independently
  remote-verified; Gate 12 explicitly authorized implementation.
- Gate 14 review found only two test-evidence gaps: incomplete all-head
  defensive-sentinel coverage and a missing explicit multi-input gradient
  finiteness assertion. Production and accepted architecture/contract passed.
- Both test-only corrections passed focused independent re-review. The
  production implementation remained byte-identical, the accepted contract
  remained unchanged, and DEC-0017's conceptual substance remained unchanged.
- Sebastien explicitly accepted the implementation and corrected evidence suite.
- Implementation checkpoint `f4b5a1d` is pushed and independently
  remote-verified with the accepted production and contract hashes unchanged.
- Small synthetic inspection demonstrated the exact one-token weight, causal
  three-position weights, future invariance, permitted historical influence,
  full shape trace, and four separately inspectable heads.
- Independent Phase 5 exit review returned PASS with no MUST-FIX findings;
  Master Chat adjudicated PASS and Sebastien accepted technical completion.
- Phase 5 closure commit `6519d8c` was pushed and independently remote-verified;
  Phase 5 is formally remotely closed.
- Sebastien explicitly authorized learning/design only for `Phase 6 —
  Transformer Block`, with exact goal `Combine attention and feed-forward
  computation into a stable reusable block.`
- The read-only Phase 6 continuity inspection is complete and accepted. At that
  inspection stage it appropriately selected no architecture and created no
  detailed contract.
- Sebastien subsequently completed the Phase 6 learning/design discussion and
  accepted the DEC-0018 conceptual architecture: one width-32, single-sequence,
  pre-norm block with accepted four-head attention, two independent explicit
  normalizations, a positionwise `32 -> 128 -> 32` GELU FFN, exactly two
  branch-output dropout sites at `p = 0.1`, and two residual paths.
- Documentation-only detailed-contract drafting is authorized and complete in
  proposed `docs/TRANSFORMER_BLOCK_SPEC.md`; no implementation is authorized.
- Master Chat Phase 6 contract sanity review returned PASS. Fresh Codex
  independent review returned FAIL with four MUST-FIX and one SHOULD-FIX
  finding, all accepted by Master Chat without changing DEC-0018.
- Authorized documentation-only corrections now define the complete two-stream
  dropout transaction, correct Phase 7/8 seed ownership, separate irreversible
  workflow gates, synchronize review history, add mode-isolation evidence, and
  incorporate the two optional clarifications.
- Gate 10 focused independent detailed-contract re-review returned PASS with all
  five prior findings resolved. Master Chat adjudicated PASS, and Sebastien
  explicitly accepted the corrected contract without changing DEC-0018.
- The accepted Phase 6 detailed mechanics are authoritative for later separately
  authorized implementation. One dedicated six-file accepted-contract
  checkpoint commit is authorized; push remains separate.
- Accepted-contract checkpoint `2177c7e` was pushed and independently
  remote-verified with local and remote `main` synchronized at ahead/behind
  `0/0`.
- Sebastien explicitly authorized Phase 6 implementation under the accepted
  contract.
- Gate 16 implementation is complete locally: explicit normalization, exact-erf
  GELU, the positionwise FFN, two dropout streams and their complete transaction,
  pre-norm residual composition with accepted Phase 5 attention, inspection,
  exports, and 50 focused synthetic tests.
- Independent Phase 6 implementation review returned FAIL with three accepted
  MUST-FIX findings: one narrow standalone-dropout seed-validation defect and
  two evidence gaps for exceptional-path global RNG isolation and parameter
  nonmutation during complete-gradient evidence.
- Focused correction is complete: standalone missing/corrupt seed state fails
  safely before snapshot/draw; exceptional transaction paths directly preserve
  global CPU RNG; and all 21 parameter objects, values, shapes, devices, and
  dtypes are proved unchanged by forward/backward.
- Gate 20 focused independent implementation re-review returned PASS with all
  three findings resolved. Master Chat adjudicated PASS, and Sebastien explicitly
  accepted the corrected implementation and evidence suite.
- The accepted contract and DEC-0018 remain unchanged. The dedicated eight-file
  accepted-implementation checkpoint commit is authorized; push and exit review
  remain separate.
- Accepted implementation checkpoint `c262407` was pushed and independently
  remote-verified with local and remote `main` synchronized at `0/0`.
- Small synthetic educational inspection demonstrated normalization, FFN,
  dropout, both residual value/gradient paths, complete shape preservation,
  causality, permitted history, same-call inspection, and gradient flow to all
  21 tensors and 12,576 parameters.
- Independent Phase 6 exit review returned PASS with no MUST-FIX findings;
  Master Chat adjudicated PASS and Sebastien accepted technical completion.

## Phase 5 exit criteria

Independent review passed and Sebastien accepted all four criteria as satisfied:

1. Queries, keys, values, scaling, masking, and attention weights are explainable. **PASS.**
2. Single-head causal self-attention is implemented transparently and tested. **PASS.**
3. Multi-head attention is built from understood components and tested. **PASS.**
4. Attention shapes and selected weights can be inspected. **PASS.**

## Phase 6 exit criteria

Independent review passed and Sebastien accepted all three criteria as satisfied:

1. Normalization, residual connections, feed-forward layers, and dropout are understood. **PASS.**
2. A Transformer block is implemented from explicit components. **PASS.**
3. Shape, causality, gradient, and residual-path behavior are tested. **PASS.**

## Current work

Phase 6 technical work is complete. This documentation-only closure bookkeeping
records the accepted exit result. It remains unstaged and uncommitted pending
separate Phase 6 closure commit authorization. Phase 6 is not yet formally
remotely closed.

## Current model status

The accepted Phase 4 model architecture remains unchanged; its trained Gate 24
instance was not retained. The Phase 5 implementation defines exactly three
tensors and 3,072 parameters for standalone single-head attention, or thirteen
tensors and 4,096 parameters for standalone four-head attention. Synthetic
tests construct only ephemeral modules; no model or parameter artifact is
retained. The Phase 6 implementation constructs an ephemeral block containing
the accepted 4,096-parameter multi-head attention plus 8,480 new parameters for
two normalizations and one positionwise FFN, totaling 21 tensors and 12,576
parameters. Tests construct only synthetic in-memory instances; no model or
parameter artifact is retained.

## Last verified working state

Local `HEAD`, refreshed `origin/main`, and actual remote `main` match accepted
Phase 6 implementation checkpoint `c262407` with ahead/behind `0/0` and a clean
worktree/index before this closure bookkeeping. Focused Phase 6 tests pass
51/51; relevant accepted Phase 3–5 regressions pass 223/223; and the complete
396-test suite has 395 passes, one expected restricted-context MPS skip, and
zero failures. Independent exit review returned PASS with no MUST-FIX findings.

## Next exact step

Obtain separate Phase 6 closure commit authorization.

## Known issues

- PyTorch warns that optional NumPy interoperability is unavailable. NumPy is
  intentionally absent because no accepted milestone requires it.
- MPS may be unavailable in restricted execution contexts; CPU float32 remains
  the accepted semantic baseline.
- Processed corpus files are ignored reproducible derivatives and must be
  regenerated only through the accepted publisher.
- The catalog and artifact report different update/Last-Modified dates; both
  remain recorded without assuming they describe the same revision mechanism.
- No production, contract, conceptual, or evidence issue remains after the
  focused PASS.
- No technical Phase 5 issue remains after the accepted exit review.
- No detailed-contract issue remains after focused independent re-review passed
  all five prior findings.
- No implementation or evidence issue remains after focused re-review passed
  all three prior findings.
- No technical Phase 6 issue remains after the accepted exit review.

## Important constraints

- Do not rerun or tune `EXP-20260909-01`.
- Never access, tokenize, window, score, inspect, or derive Phase 4 statistics
  from the sealed test work.
- Do not persist the completed run's parameters, gradients, logits,
  probabilities, losses, token/window/target data, or a checkpoint.
- Preserve the accepted tokenizer, vocabulary identity, Phase 3 representation,
  model architecture, frozen experiment settings, and evidence-limited claim.
- Do not modify the accepted implementation, tests, or technical contract.
- DEC-0018 is the accepted Phase 6 conceptual architecture; do not change it
  through detailed mechanics without explicit Master Chat adjudication and
  Sebastien approval.
- The corrected detailed contract is accepted; do not modify its mechanics or
  implement outside its boundary.
- Do not modify the accepted Phase 6 contract, implementation, tests, or
  evidence suite.
- Do not stage, commit, or push the closure bookkeeping without separate
  authorization.
- Do not begin Phase 7 stacking or integration, Phase 8 training/checkpointing,
  or Phase 9 generation/evaluation.

## Open questions

- Phase 5 has no open technical or closure question.
- No accepted Phase 6 conceptual or detailed-contract question remains open.
- Phase 6 has no open technical question.
- Closure commit authorization, closure push/remote verification, and Phase 7
  authorization remain separate future gates.
- Complete-model seed allocation for multiple distinct stacked blocks is
  explicitly deferred to Phase 7; it does not reopen Phase 5 or Phase 6.

## Session handoff

Phases 0–4 are remotely closed. Phase 5's accepted contract is `9443dec`, its
accepted implementation is `f4b5a1d`, and all four exit criteria remain
satisfied. Closure commit `6519d8c`, `Complete Phase 5 self-attention`, was
pushed and independently remote-verified with local `HEAD`, `origin/main`, and
actual remote `main` at that exact commit, ahead/behind `0/0`, and a clean
worktree/index. Phase 5 is formally remotely closed. Sebastien then authorized
Phase 6 learning/design only. The accepted continuity inspection verified the
exact phase name, goal, exit criteria, inherited interfaces, boundaries,
learning prerequisites, open architecture questions, and deferred mechanics.
Sebastien then completed the learning/design discussion, accepted the
consolidated architecture recorded in DEC-0018, and authorized documentation-
only detailed-contract drafting. Corrected `docs/TRANSFORMER_BLOCK_SPEC.md` is
accepted with exact component arithmetic, counts, initialization,
dropout, inspection, evidence, boundaries, and gates. Master Chat sanity review
returned PASS; fresh Codex review returned FAIL with four MUST-FIX and one
SHOULD-FIX finding; and Master Chat accepted all five without changing DEC-0018.
Authorized documentation-only corrections are complete locally. No source,
tests, parameters, experiment, or sealed-test access occurred. Gate 10 focused
re-review returned PASS, Master Chat adjudicated PASS, and Sebastien accepted
the corrected detailed contract without changing DEC-0018. Accepted contract
checkpoint `2177c7e` is pushed and independently remote-verified. Sebastien then
authorized implementation, and Gate 16 completed locally. Independent review
returned FAIL with three MUST-FIX findings accepted by Master Chat: one narrow
production validation defect and two test-evidence gaps. Focused source/test
corrections are complete with 51/51 focused tests, 223/223 relevant Phase 3–5
regressions, and a 396-test full suite with 395 passes, one expected MPS skip,
and zero failures. Gate 20 focused re-review returned PASS, Master Chat
adjudicated PASS, and Sebastien accepted the corrected implementation and
evidence suite. The accepted contract and DEC-0018 remain unchanged. The
accepted implementation checkpoint `c262407` is pushed and independently
remote-verified. Educational inspection and independent exit review passed all
three exact criteria with no MUST-FIX findings; Master Chat adjudicated PASS,
and Sebastien accepted Phase 6 as technically complete. No training experiment
was required. Closure bookkeeping is complete locally and uncommitted; Phase 6
is not yet formally remotely closed. Proceed only to separate Phase 6 closure
commit authorization. Phase 7 remains unauthorized.
