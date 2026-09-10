# Project State

> Primary session anchor. Keep this document concise and current; it describes now, not project history.

## Current phase

Phase 5 — Self-Attention (technically COMPLETE; closure bookkeeping local)

## Current milestone

The accepted contract checkpoint `9443dec240ccdb5fdc1ebfcbe0f7b334893dbfff`
and accepted implementation checkpoint
`f4b5a1d8d29ab7ee6eb5b987fe150fb040c4544e` are pushed and independently
remote-verified. Educational inspection and independent Phase 5 exit review
returned PASS with no MUST-FIX findings. Master Chat adjudicated PASS, and
Sebastien explicitly accepted all four exit criteria and Phase 5 as technically
complete. Documentation-only closure bookkeeping is complete locally; formal
remote closure still requires a separately authorized closure commit, push,
and independent remote verification.

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

## Phase 5 exit criteria

Independent review passed and Sebastien accepted all four criteria as satisfied:

1. Queries, keys, values, scaling, masking, and attention weights are explainable. **PASS.**
2. Single-head causal self-attention is implemented transparently and tested. **PASS.**
3. Multi-head attention is built from understood components and tested. **PASS.**
4. Attention shapes and selected weights can be inspected. **PASS.**

## Current work

Phase 5 technical work is complete. This documentation-only closure bookkeeping
records the accepted exit result. It is unstaged and uncommitted pending
dedicated Phase 5 closure commit authorization. Phase 5 is not yet remotely
closed.

## Current model status

The accepted Phase 4 model architecture remains unchanged; its trained Gate 24
instance was not retained. The Phase 5 implementation defines exactly three
tensors and 3,072 parameters for standalone single-head attention, or thirteen
tensors and 4,096 parameters for standalone four-head attention. Synthetic
tests construct only ephemeral modules; no model or parameter artifact is
retained.

## Last verified working state

Local `HEAD`, refreshed `origin/main`, and actual remote `main` match accepted
implementation checkpoint `f4b5a1d` with ahead/behind `0/0`. Focused Phase 5
tests pass 41/41; relevant Phase 3/4 regressions pass 182/182; and the complete
345-test suite has 344 passes, one expected restricted-context MPS skip, and
zero failures. Independent exit review returned PASS with no MUST-FIX findings.

## Next exact step

Obtain dedicated Phase 5 closure commit authorization.

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

## Important constraints

- Do not rerun or tune `EXP-20260909-01`.
- Never access, tokenize, window, score, inspect, or derive Phase 4 statistics
  from the sealed test work.
- Do not persist the completed run's parameters, gradients, logits,
  probabilities, losses, token/window/target data, or a checkpoint.
- Preserve the accepted tokenizer, vocabulary identity, Phase 3 representation,
  model architecture, frozen experiment settings, and evidence-limited claim.
- Do not modify the accepted implementation, tests, or technical contract.
- Do not stage, commit, or push the closure bookkeeping without separate
  authorization.
- Do not add batching, caller masks, padding, residual paths, normalization,
  feed-forward layers, dropout, Transformer blocks, stacking, checkpoints,
  generation, final evaluation, or Phase 6 work.
- Do not begin Phase 6; it requires Phase 5 remote closure and separate explicit
  authorization.

## Open questions

- No accepted conceptual or detailed-contract question remains open.
- The closure commit, closure push/remote verification, and Phase 6
  authorization remain separate future gates.

## Session handoff

Phase 4 is remotely closed at `e8b5c55`. Gate 30 Phase 5 learning/design,
conceptual approval, and documentation-only contract authorization are
complete. DEC-0017, the proposed self-attention specification, and minimum
continuity corrections completed Gate 8. Gate 9 focused re-review and Master
Chat adjudication returned PASS, and Sebastien explicitly accepted the
corrected detailed contract without changing DEC-0017. Contract checkpoint
`9443dec` is remotely verified. Gate 14 review found two test-only evidence
gaps and no production or contract defect. Both corrections passed focused
independent re-review, and accepted implementation checkpoint `f4b5a1d` is
remotely verified. Educational inspection and independent exit review passed
all four criteria with no MUST-FIX findings; Master Chat adjudicated PASS and
Sebastien accepted Phase 5 as technically complete. No training experiment was
required. The closure bookkeeping is local and uncommitted; Phase 5 is not yet
remotely closed. Proceed only to dedicated closure commit authorization.
