# SebGPT Roadmap

The phases are sequential learning gates. Do not begin a later phase until the current phase's exit criteria are met, verified, and reflected in `PROJECT_STATE.md`. Refinements within a completed phase are allowed when a later discovery exposes a gap, but the reason should be documented.

## Phase 0 — Project and Environment (complete)

**Goal:** Establish durable project continuity and a minimal, reproducible development environment without implementing machine learning.

**Planned work:**

- Create and review the project documentation system and directory structure.
- Decide whether and how to initialize version control.
- Document supported platform, Python version, PyTorch version, and dependency-management approach.
- Create the environment only after those choices are reviewed.
- Add a minimal verification workflow for the environment.

**Exit criteria:**

- Continuity documents are present, internally consistent, and approved.
- Version-control and environment decisions are recorded.
- A fresh environment can be created from documented instructions.
- Python and PyTorch imports and a basic tensor operation are verified.
- `PROJECT_STATE.md` identifies the exact first step of Phase 1.

## Phase 1 — Dataset (complete)

**Goal:** Understand how raw text becomes a reproducible language-modeling dataset.

**Milestones:**

1. Define learning objectives and dataset selection criteria. **Complete.**
2. Approve the corpus design and document the source, manifest, split policy, and acquisition contract. **Complete.**
3. Acquire the approved raw source and record immutable provenance. **Complete.**
4. Inspect source structure and define exact extraction boundaries. **Complete.**
5. Implement and test deterministic extraction and splitting. **Complete: read-only preflight, deterministic in-memory extraction, the Models A/B cooperative-lock publisher, final expected-result ledger, and production publication are accepted and verified.**
6. Inspect corpus characteristics and audit validation/test Unicode code points against the training-only character inventory. **Complete: in-memory character inventory and zero-candidate unseen-character audit accepted and verified; occurrence-location reporting is not applicable for the pinned corpus.**

**Exit criteria:**

- A small, approved learning dataset is selected with source and licensing recorded.
- Raw and processed data boundaries are explicit.
- Loading, inspection, splitting, and deterministic reproduction are implemented and tested.
- Data characteristics and limitations are understood and documented.

## Phase 2 — Tokenization (complete)

**Goal:** Understand and implement the mapping between text and token IDs.

**Gates:**

1. Approve the conceptual tokenizer strategy. **Complete: corrected DEC-0013 defines the accepted boundary.**
2. Document the detailed tokenizer and vocabulary contract. **Complete: corrected DEC-0014 and `docs/TOKENIZER_SPEC.md` define the accepted contract.**
3. Independently review, correct, and accept the contract. **Complete: focused re-review passed with no must-fix issues and Sebastien accepted DEC-0014.**
4. Commit the accepted contract checkpoint. **Complete at `fd68bce66214ecf911dd643d2cc8ae35c3cdbb67`.**
5. Push the contract checkpoint and verify the remote commit. **Complete: local and remote `main` were verified at the Gate 4 commit.**
6. Explicitly authorize implementation. **Complete: implementation and synthetic deterministic tests were authorized.**
7. Implement the tokenizer and deterministic tests. **Complete locally: implementation follows DEC-0014 and focused tests pass; acceptance remains Gate 8.**
8. Independently review, correct, and accept the implementation. **Complete: the first review found one test-independence issue; focused re-review passed with no must-fix issues and Sebastien accepted the implementation.**
9. Commit the accepted implementation checkpoint. **Complete at `de7a7f096fbbd8607c944412eaef30be9b686b56`.**
10. Push the implementation checkpoint and verify the remote commit. **Complete: local and remote `main` were verified at the Gate 9 commit.**
11. Explicitly authorize production vocabulary construction and permitted statistics. **Complete: production construction and in-memory training/validation statistics were authorized.**
12. Construct and hash the vocabulary artifact and inspect permitted statistics. **Complete locally: the 81-entry artifact and permitted statistics are verified; acceptance remains Gate 13.**
13. Independently review, correct, and accept the production checkpoint. **Complete: focused re-review passed with no must-fix issues and Sebastien accepted the artifact and statistics checkpoint.**
14. Commit the accepted vocabulary checkpoint. **Complete at `0723474e9e7b3df8e853a148cce71f34bc004ca1`.**
15. Push the vocabulary checkpoint and verify the remote commit. **Complete: local and remote `main` were verified at the Gate 14 commit.**
16. Review every Phase 2 exit criterion. **Complete: independent review passed and Sebastien accepted that all technical exit criteria are satisfied.**
17. Create a final Phase 2 closure commit if required. **This authorized closure commit completes Gate 17.**
18. Push the final closure and verify the remote commit. **Authorized immediately after Gate 17 succeeds.**
19. Only then authorize Phase 3. **Complete for learning and contract design only through DEC-0015; implementation is not authorized.**

**Exit criteria:**

- An intentionally simple tokenizer is implemented without an opaque tokenizer framework. **Verified.**
- Vocabulary construction, encoding, decoding, unknown cases, and round trips are explained. **Verified.**
- Tests cover representative and edge cases. **Verified.**
- Tokenization statistics are inspected on the Phase 1 dataset. **Verified under the approved train/validation governance; the sealed test remained untouched.**

## Phase 3 — Embeddings (complete)

**Goal:** Understand learned token and positional representations.

The conceptual architecture and corrected detailed contract are accepted in
DEC-0015 and `docs/EMBEDDING_SPEC.md` after independent review, correction,
passing focused re-review, master-chat adjudication, and explicit Sebastien
acceptance. The contract commit is remotely verified, implementation was
explicitly authorized, and the accepted representation implementation and
tests are complete locally. Independent implementation review returned PASS,
no corrections were required, and Sebastien accepted the implementation. The
dedicated accepted-implementation commit is explicitly authorized and completes
Gate 12. Gate 13 remote verification completed, Gate 14 was explicitly skipped
as unnecessary, and Gate 15 independently verified all three exit criteria.
The documentation-only closure commit completed Gate 16. Gate 17 push and
independent remote verification are complete at
`a66d169e76f06bee18d4f32b6340206f9f45ee63`. Gate 18 explicitly authorized
Phase 4 learning and design, which is now complete.

**Exit criteria:**

- Token embeddings and an explicit positional representation are implemented. **Verified.**
- Tensor shapes and parameter roles are explainable and tested. **Verified.**
- Small examples demonstrate lookup, batching, and gradient flow. **Verified.**

## Phase 4 — Simple Neural Language Model

**Goal:** Build and train a small non-attention baseline to understand next-token prediction.

The accepted contract, shifted-example/governance slice, model/loss/update
slice, fixed runner, and safe seven-work corpus factory are committed and
remotely verified through `ba09d017e97a6d25317e160fc1a40a6304bcdd96`.
Master chat explicitly authorized one fixed execution. The first launcher
attempt stopped before imports and before Gate 24 because the documented
`PYTHONPATH=src` convention was absent; it performed no experiment work.
Master-chat adjudication classified that as a pre-experiment launcher abort and
authorized one corrected launch with no semantic change. The corrected launch
entered Gate 24, passed the accepted fifteen-stage preflight, constructed one
model, completed two passes and exactly 24,778 updates, and produced the valid
immutable result `EXP-20260909-01`. Training loss fell from
`4.426503102003006` to `2.5551429421586387`, below the exact `ln(81)` baseline
`4.394449154672439`; both frozen predicate clauses are true, so the recorded
result is PASS. Validation loss improved from `4.427434147633409` to
`2.565088013405899` as an observation only. Gate 25 independently returned PASS
with no MUST-FIX issues and verified the experiment evidence and every Phase 4
exit criterion. Gate 26 is not applicable because the valid experiment passed.
Gate 27 master-chat adjudication returned PASS, and Sebastien explicitly
accepted the result, all four exit criteria, and Phase 4 as technically
complete. Gate 28 closure commit and Gate 29 push/independent remote
verification are complete at
`e8b5c55fb2f2f03b155c1a8b4308dd9df20d9e1c`. Phase 4 is remotely closed and
must not be reopened. Gate 30 subsequently authorized Phase 5 learning/design.

**Exit criteria:**

- Inputs, targets, logits, probabilities, and cross-entropy loss are understood. **Verified and satisfied at Gates 25–27.**
- A simple neural language model is implemented and tested. **Verified and satisfied at Gates 25–27.**
- Backpropagation and parameter updates are inspected on a small example. **Verified and satisfied at Gates 25–27.**
- A reproducible run shows loss improving over a baseline. **Verified and satisfied by Gate 24 result `EXP-20260909-01` and Gates 25–27 review and acceptance.**

## Phase 5 — Self-Attention

**Goal:** Derive and implement causal self-attention before using it inside a Transformer.

Gate 30 learning/design authorization is complete. Sebastien completed the
first-principles discussion and accepted the conceptual architecture recorded
in DEC-0017: standalone, single-sequence causal self-attention over the
width-32 token-plus-position representation; one bias-free full-width head;
four separately visible bias-free width-8 heads plus a bias-free width-32
output projection; exact `j <= i` visibility; and explicit inspection of the
post-softmax weights used for the output. A documentation-only detailed
contract is proposed in `docs/SELF_ATTENTION_SPEC.md`. Gate 5 drafting is
complete, Gate 6 Master Chat sanity review returned PASS, and Gate 7 fresh
independent review returned FAIL with five MUST-FIX and three SHOULD-FIX
findings. Master Chat accepted all eight without changing DEC-0017. Gate 8
documentation-only corrections resolved every finding. Gate 9 focused
independent re-review returned PASS, Master Chat adjudicated PASS, and Sebastien
explicitly accepted the corrected detailed contract without changing DEC-0017.
The detailed mechanics are authoritative for later implementation, and the
dedicated accepted-contract checkpoint was committed, pushed, and independently
remote-verified at `9443dec240ccdb5fdc1ebfcbe0f7b334893dbfff`. Gate 12
explicitly authorized implementation. Gate 13 is complete locally with the
standalone single-head and four-head modules, exact inspection API, public
exports, and 40 focused synthetic tests. The complete 344-test suite passes
with 343 passes, one expected restricted-context MPS skip, and zero failures.
Gate 14 independent review found two test-only evidence gaps and no production,
contract, DEC-0017, or phase-boundary defect. Both authorized Gate 15 test-only
corrections are complete locally: focused tests pass 41/41 and the complete
345-test suite passes with 344 passes and one expected MPS skip. Focused
independent re-review returned PASS, Master Chat adjudicated PASS, and Sebastien
explicitly accepted the implementation and corrected evidence suite. Production
remained byte-identical, the accepted contract remained unchanged, and DEC-0017
remained unchanged. The dedicated accepted-implementation checkpoint commit is
`f4b5a1d8d29ab7ee6eb5b987fe150fb040c4544e`, pushed, and independently
remote-verified. Educational inspection supplied the accepted small synthetic
evidence. Independent Phase 5 exit review returned PASS with no MUST-FIX
findings, Master Chat adjudicated PASS, and Sebastien explicitly accepted all
four exact exit criteria and Phase 5 as technically complete. No training
experiment was required. The documentation-only closure commit
`6519d8c813b7e5bc899b73d6b1fa8c38a6819750`, `Complete Phase 5
self-attention`, was subsequently pushed and independently remote-verified.
Local `HEAD`, `origin/main`, and actual remote `main` matched that exact commit
with ahead/behind `0/0` and a clean worktree/index. All four exit criteria
remain satisfied, and Phase 5 is formally remotely closed. Sebastien then
explicitly authorized Phase 6 learning/design only.

**Exit criteria:**

- Queries, keys, values, scaling, masking, and attention weights are explainable. **Verified and satisfied.**
- Single-head causal self-attention is implemented transparently and tested. **Verified and satisfied.**
- Multi-head attention is built from understood components and tested. **Verified and satisfied.**
- Attention shapes and selected weights can be inspected. **Verified and satisfied.**

## Phase 6 — Transformer Block

**Goal:** Combine attention and feed-forward computation into a stable reusable block.

Phase 6 learning/design only is explicitly authorized, and the read-only
continuity inspection is complete and accepted. Sebastien completed the
learning/design discussion and accepted the consolidated conceptual architecture
recorded in DEC-0018: one width-32 single-sequence pre-norm block, accepted
four-head Phase 5 attention, two independent explicit LayerNorm-style
components, a positionwise `32 -> 128 -> 32` GELU feed-forward network,
dropout `p = 0.1` at exactly the two branch outputs, two residual paths, and
same-call educational inspection. Documentation-only detailed-contract drafting
is authorized and complete in proposed `docs/TRANSFORMER_BLOCK_SPEC.md`. Master
Chat sanity review returned PASS. Fresh Codex independent contract review
returned FAIL with four MUST-FIX and one SHOULD-FIX finding; Master Chat
accepted all five without changing DEC-0018. Authorized documentation-only
corrections completed Gate 9. Gate 10 focused independent re-review returned
PASS with all five findings resolved; Master Chat adjudicated PASS; and
Sebastien explicitly accepted the corrected detailed contract without changing
DEC-0018. Its mechanics are authoritative for later separately authorized
implementation. The dedicated six-file accepted-contract checkpoint is committed
at `2177c7e9aaf7a12790afb1fd25693723ac759170`, pushed, and independently
remote-verified with local and remote `main` synchronized at ahead/behind `0/0`.
Sebastien then explicitly authorized Phase 6 implementation. Gate 16 is complete
locally with the explicit block components, accepted Phase 5 attention
composition, complete two-stream dropout transaction, exports, and 50 focused
synthetic tests. Relevant Phase 3–5 regressions pass 223/223, and the complete
395-test suite has 394 passes, one expected MPS skip, and zero failures. The
implementation remains unaccepted, unstaged, uncommitted, and unpushed pending
Master Chat sanity review and fresh independent review. Independent review
subsequently returned FAIL with three MUST-FIX findings accepted by Master Chat:
one narrow standalone-dropout seed-validation defect and two test-evidence gaps.
Focused source/test corrections are complete locally without changing the
accepted contract or DEC-0018. Corrected focused tests pass 51/51, relevant
Phase 3–5 regressions pass 223/223, and the complete 396-test suite has 395
passes, one expected MPS skip, and zero failures. Focused independent
implementation re-review subsequently returned PASS with all three findings
resolved. Master Chat adjudicated PASS, and Sebastien explicitly accepted the
corrected implementation and evidence suite. The accepted contract and
DEC-0018 remain unchanged. The dedicated eight-file accepted-implementation
checkpoint commit is authorized and completes Gate 23. Push and independent
remote verification remain separate; Phase 6 exit review/closure remains
unauthorized until they complete. The accepted implementation checkpoint was
subsequently committed as `c2624078eae5947e505d7f8093869e32c58e521b`, pushed,
and independently remote-verified. Educational inspection demonstrated the
accepted normalization, FFN, dropout, residual, shape, causality, historical
influence, same-call inspection, gradient, and parameter-count evidence.
Independent Phase 6 exit review returned PASS with no MUST-FIX findings, Master
Chat adjudicated PASS, and Sebastien explicitly accepted all three exact exit
criteria and Phase 6 as technically complete. No training experiment was
required or performed. The documentation-only closure was subsequently
committed as `e84b364a7955ddba86b30938babd5dae9e829935`, `Complete Phase 6
transformer block`, pushed, and independently remote-verified with local and
remote `main` synchronized at ahead/behind `0/0`. Phase 6 is formally remotely
closed. Sebastien then explicitly authorized Phase 7 learning/design only.

**Exit criteria:**

- Normalization, residual connections, feed-forward layers, and dropout are understood. **Verified and satisfied.**
- A Transformer block is implemented from explicit components. **Verified and satisfied.**
- Shape, causality, gradient, and residual-path behavior are tested. **Verified and satisfied.**

## Phase 7 — Complete Mini-GPT

**Goal:** Assemble a decoder-only autoregressive Transformer.

Phase 6 is formally remotely closed at
`e84b364a7955ddba86b30938babd5dae9e829935`. Sebastien explicitly authorized
Phase 7 learning/design only, and the read-only continuity inspection is
complete and accepted. Sebastien subsequently accepted the consolidated
conceptual architecture in DEC-0019: one nonempty single sequence of length
`1..256`, the unchanged accepted Phase 3 representation, exactly four distinct
and non-shared accepted Phase 6 blocks, one final explicit width-32
LayerNorm-style normalization, one untied biased `(32, 81)` language-model
head, logits-only forward, separate next-token loss, same-call complete-model
inspection, and exactly 63,825 parameters. Documentation-only detailed-contract
drafting and continuity reconciliation are authorized and complete locally in
proposed `docs/MINI_GPT_SPEC.md`. Fresh independent review returned CORRECT
BEFORE ACCEPTANCE with five IMPORTANT findings and no BLOCKER. Master Chat
accepted all five without changing DEC-0019 and authorized only targeted
documentation correction. The five corrections are complete locally: safe
parameter-validation order, an exact public/error oracle, equal-length
causality evidence, stack-wide stochastic failure-state evidence without outer
rollback, and fully separated irreversible-action gates. The proposal remains
unaccepted at that correction checkpoint pending focused independent re-review
and final acceptance. Focused independent re-review subsequently returned PASS with all five findings
resolved and no BLOCKER, IMPORTANT, or MINOR findings. Codex recommended
`ACCEPT CORRECTED CONTRACT`, and Master Chat formally accepted the corrected
`docs/MINI_GPT_SPEC.md` detailed contract without changing DEC-0019. Complete-
model authority remains exactly 90 learned-parameter tensors and 63,825
parameters. The accepted contract checkpoint was subsequently committed as
`60b2a9cce55da79ccc9fbd03fad014cb2a939290`, pushed, and independently
remote-verified with local and remote `main` synchronized at ahead/behind
`0/0`. Master Chat then explicitly authorized Phase 7 implementation. Gate 20
is complete locally with the exact four-block model, final normalization,
untied biased head, logits-only forward, same-call inspection, required exports,
and 44 focused synthetic tests. Focused tests pass 44/44; the complete 440-test
suite has 439 passes, one expected restricted-context MPS skip, and zero
failures. Implementation remains unaccepted, unstaged, uncommitted, and
unpushed pending Master Chat sanity review and fresh independent review.
Independent review subsequently returned CORRECT BEFORE ACCEPTANCE with three
IMPORTANT findings and no BLOCKER. Master Chat accepted all three without
changing DEC-0019 or the accepted contract and authorized targeted correction.
The corrections are complete locally: exact child-module type-error ownership,
literal complete public-signature evidence, and exact failure-path object
handoff evidence. Corrected focused tests pass 47/47; Phase 3, 5, and 6 focused
regressions pass 35/35, 41/41, and 51/51; and the complete 443-test suite has
442 passes, one expected MPS skip, and zero failures. Implementation remains
unaccepted, unstaged, uncommitted, and unpushed pending focused independent
re-review. Focused re-review subsequently returned PASS with all three prior
IMPORTANT findings resolved and no BLOCKER, IMPORTANT, or MINOR findings. Codex
recommended `ACCEPT CORRECTED IMPLEMENTATION`, and Master Chat formally
accepted the corrected implementation, exact file identities, 63,825-parameter
and 90-tensor counts, and evidence suite without changing DEC-0019 or the
accepted contract. The implementation is accepted locally but remains
unstaged, uncommitted, and unpushed pending separate implementation-checkpoint
commit authorization. The accepted implementation was subsequently committed as
`3139b1736f005fe903e2ea111d91934478b5a683`, pushed, and independently
remote-verified with local and remote `main` synchronized at ahead/behind
`0/0`. All four exact Phase 7 exit criteria are satisfied. Phase 7 is
technically complete and ready for formal closure. Documentation-only closure
bookkeeping is complete locally; the separate closure commit remains
unauthorized. No training, experiment, sealed-test access, Phase 8, or Phase 9
work is authorized.

**Exit criteria:**

- Embeddings, stacked blocks, final normalization, and language-model head are integrated. **Verified and satisfied.**
- Parameter count and configuration are explicit. **Verified and satisfied.**
- Forward pass, loss path, causality, and shape behavior are tested. **Verified and satisfied.**
- The architecture is documented and matches recorded decisions. **Verified and satisfied.**

## Phase 8 — Training and Checkpointing

**Goal:** Create a reproducible training loop and durable model state.

**Exit criteria:**

- Batching, optimization, evaluation intervals, seeds, and device behavior are explicit.
- Training and validation loss are recorded reproducibly.
- Checkpoints save and restore model, optimizer, configuration, and progress.
- At least one fully documented training run can be resumed successfully.

## Phase 9 — Generation and Evaluation

**Goal:** Generate text and evaluate model behavior with appropriately modest claims.

**Exit criteria:**

- Autoregressive generation and sampling controls are implemented and explained.
- Deterministic and stochastic generation paths are tested.
- Quantitative metrics and qualitative samples are recorded.
- Limitations, failure modes, and comparisons to simple baselines are documented.

## Phase 10 — Interpretability / Black-box Inspection

**Goal:** Investigate learned behavior and internal representations using controlled observations.

**Exit criteria:**

- Activations, embeddings, attention patterns, and selected predictions can be inspected.
- At least one hypothesis-driven interpretability experiment is run reproducibly.
- Findings clearly separate observation, interpretation, and speculation.
- The project's core learning outcomes are reviewed against the charter.

## Optional later phases

Optional work requires an explicit decision after the core phases. Candidate areas include:

- better tokenization methods
- larger and more varied datasets
- GPU training and performance engineering
- instruction tuning
- a conversational interface

Each optional phase must define its own learning goal, prerequisites, risks, and exit criteria before implementation begins.
