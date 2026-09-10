# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 5 — Self-Attention (implementation accepted; checkpoint commit authorized)

## Current milestone

Phase 4 closure commit `e8b5c55fb2f2f03b155c1a8b4308dd9df20d9e1c`
is pushed and independently remote-verified. Gate 30 authorized Phase 5
learning/design. Sebastien completed the learning discussion, accepted the
DEC-0017 conceptual architecture, and authorized a documentation-only detailed
contract. Gate 5 drafting completed, Gate 6 Master Chat sanity review passed,
and Gate 7 fresh independent review failed with five MUST-FIX and three
SHOULD-FIX findings. Gate 8 corrected all eight findings, Gate 9 focused
independent re-review returned PASS, Master Chat adjudicated PASS, and Sebastien
explicitly accepted the corrected detailed contract. The dedicated Gate 10
accepted-contract checkpoint was committed at `9443dec`, pushed, and
independently remote-verified. Gate 12 implementation authorization is complete,
and Gate 13 implementation plus focused tests are complete locally pending
review. Gate 14 independent implementation review returned FAIL on two
test-evidence gaps and found no production or contract defect. Master Chat
accepted both findings. The authorized Gate 15 test-only corrections are
complete. Focused independent re-review returned PASS, Master Chat adjudicated
PASS, and Sebastien explicitly accepted the Phase 5 implementation and corrected
evidence suite. The dedicated accepted-implementation checkpoint commit is
authorized; push remains separate.

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

## Phase 5 exit criteria

These exact roadmap criteria are not yet satisfied:

1. Queries, keys, values, scaling, masking, and attention weights are explainable.
2. Single-head causal self-attention is implemented transparently and tested.
3. Multi-head attention is built from understood components and tested.
4. Attention shapes and selected weights can be inspected.

## Current work

The accepted implementation, corrected 41-test evidence suite, exports, and
status documents are authorized for one dedicated checkpoint commit. Push and
Phase 5 exit review/closure remain unauthorized.

## Current model status

The accepted Phase 4 model architecture remains unchanged; its trained Gate 24
instance was not retained. The Phase 5 implementation defines exactly three
tensors and 3,072 parameters for standalone single-head attention, or thirteen
tensors and 4,096 parameters for standalone four-head attention. Synthetic
tests construct only ephemeral modules; no model or parameter artifact is
retained.

## Last verified working state

Before implementation, local `HEAD`, refreshed `origin/main`, and actual remote
`main` matched accepted contract checkpoint `9443dec` with ahead/behind `0/0`
and a clean worktree/index. Focused Phase 5 tests pass 41/41; relevant Phase 3/4
regressions pass 182/182; and the complete 345-test suite passes with 344
passes, one expected restricted-context MPS skip, and zero failures.

## Next exact step

Obtain separate authorization to push and independently remote-verify the accepted implementation checkpoint.

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

## Important constraints

- Do not rerun or tune `EXP-20260909-01`.
- Never access, tokenize, window, score, inspect, or derive Phase 4 statistics
  from the sealed test work.
- Do not persist the completed run's parameters, gradients, logits,
  probabilities, losses, token/window/target data, or a checkpoint.
- Preserve the accepted tokenizer, vocabulary identity, Phase 3 representation,
  model architecture, frozen experiment settings, and evidence-limited claim.
- The implementation and evidence suite are accepted; the dedicated checkpoint
  commit is authorized.
- Do not push, train, or run an experiment.
- Do not add batching, caller masks, padding, residual paths, normalization,
  feed-forward layers, dropout, Transformer blocks, stacking, checkpoints,
  generation, final evaluation, or Phase 6 work.
- Do not begin Phase 5 closure or Phase 6.

## Open questions

- No accepted conceptual or detailed-contract question remains open.
- Push/remote verification and Phase 5 exit-review authorization remain separate
  future gates.

## Session handoff

Phase 4 is remotely closed at `e8b5c55`. Gate 30 Phase 5 learning/design,
conceptual approval, and documentation-only contract authorization are
complete. DEC-0017, the proposed self-attention specification, and minimum
continuity corrections completed Gate 8. Gate 9 focused re-review and Master
Chat adjudication returned PASS, and Sebastien explicitly accepted the
corrected detailed contract without changing DEC-0017. Contract checkpoint
`9443dec` is remotely verified. Gate 14 review found two test-only evidence
gaps and no production or contract defect. Both corrections passed focused
independent re-review; Master Chat adjudicated PASS; and Sebastien accepted the
implementation and corrected evidence suite. Reviewed production and the
accepted contract remain byte-identical, and DEC-0017 remains unchanged. The
dedicated implementation checkpoint commit is authorized. Push, Phase 5 exit
review/closure, training, experiment, sealed-test access, and Phase 6 remain
unauthorized.
