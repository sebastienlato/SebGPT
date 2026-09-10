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

The twelve conceptual architecture decisions and corrected detailed contract
are accepted in DEC-0016 and `docs/SIMPLE_LANGUAGE_MODEL_SPEC.md`. Independent
Gate 6 review returned FAIL with seven must-fix findings; Gate 7 corrected all
seven, focused Gate 8 re-review returned PASS, master-chat adjudication returned
PASS, and Sebastien explicitly accepted the contract. Gate 9, the dedicated
accepted-contract commit, and Gate 10 push/remote verification are complete at
`3fcf007ce819ca1a45aa75b48fa19f10311f2819`. Gate 11 explicitly authorized
only shifted-example and Shakespeare-governance implementation with focused
tests. Gate 13 independent review passed production but returned FAIL for two
test-only coverage blockers. Gate 14 corrections retain all original tests and
expand the focused suite to 44 passing tests with production source unchanged.
Focused re-review and master-chat adjudication returned PASS, and Sebastien
explicitly accepted the slice. Gate 15, the dedicated implementation commit,
and Gate 16 push/remote verification are complete at
`266797f891e9980b57bb35a633c91bef838d4111`. Gate 17 explicitly authorized the
model/loss/update slice. Gate 19 independent review failed only on
`clear_gradients()` parameter-validation ordering and its missing regression
proof. Both are corrected locally at Gate 20 with 48 focused tests passing and
focused independent re-review returned PASS. Master-chat adjudication also
returned PASS, and Sebastien explicitly accepted the model/loss/update slice.
Gates 17–20 are complete. Gate 21 committed the accepted model/loss/update slice
as `497ecde3577677903f14669722d61dcdf8caa1d6`, and Gate 22 pushed and remotely
verified it. An attempted Gate 23 authorization stopped before model
construction or corpus traversal because accepted experiment-orchestration
machinery was missing; no Gate 24 run or experiment result exists. Sebastien
authorized only that prerequisite implementation. Independent review returned
FAIL on four focused issues: inherited model authority, governance before
tensorization, durable result completeness, and test coverage. All four are
corrected locally with 41 synthetic tests and await focused re-review. The
focused re-review passed all production corrections but found one final
lifecycle-role test-only blocker. The test now derives measurement roles from
distinct split sentinel identities; final focused re-review and master-chat
adjudication passed, and Sebastien explicitly accepted the runner and its 41
tests. Its dedicated commit and push/remote verification remain separately
gated in that historical status; both are now complete at
`a4da629a59584a0185a2ec286b0e770b7940940e`. A fresh experiment authorization
then stopped before Gate 24 because the runner had no accepted safe constructor
for its seven-work `Phase4PermittedCorpus`. Sebastien authorized only that
factory prerequisite, which is complete locally with 12 focused tests and
awaited independent review. That review returned FAIL on one path-containment
defect, three test-strength gaps, and stale README wording. All five were
corrected locally with 14 focused tests. Those corrections passed final focused
re-review and master-chat
adjudication, and Sebastien explicitly accepted the factory and its 14 tests.
Its dedicated commit and push/remote verification remain separately gated. The
bounded experiment remains unauthorized pending those checkpoints and a fresh
authorization.

**Exit criteria:**

- Inputs, targets, logits, probabilities, and cross-entropy loss are understood.
- A simple neural language model is implemented and tested.
- Backpropagation and parameter updates are inspected on a small example.
- A reproducible run shows loss improving over a baseline.

## Phase 5 — Self-Attention

**Goal:** Derive and implement causal self-attention before using it inside a Transformer.

**Exit criteria:**

- Queries, keys, values, scaling, masking, and attention weights are explainable.
- Single-head causal self-attention is implemented transparently and tested.
- Multi-head attention is built from understood components and tested.
- Attention shapes and selected weights can be inspected.

## Phase 6 — Transformer Block

**Goal:** Combine attention and feed-forward computation into a stable reusable block.

**Exit criteria:**

- Normalization, residual connections, feed-forward layers, and dropout are understood.
- A Transformer block is implemented from explicit components.
- Shape, causality, gradient, and residual-path behavior are tested.

## Phase 7 — Complete Mini-GPT

**Goal:** Assemble a decoder-only autoregressive Transformer.

**Exit criteria:**

- Embeddings, stacked blocks, final normalization, and language-model head are integrated.
- Parameter count and configuration are explicit.
- Forward pass, loss path, causality, and shape behavior are tested.
- The architecture is documented and matches recorded decisions.

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
