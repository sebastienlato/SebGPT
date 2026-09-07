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

## Phase 1 — Dataset (current)

**Goal:** Understand how raw text becomes a reproducible language-modeling dataset.

**Milestones:**

1. Define learning objectives and dataset selection criteria. **Complete.**
2. Approve the corpus design and document the source, manifest, split policy, and acquisition contract. **Complete.**
3. Acquire the approved raw source and record immutable provenance. **Complete.**
4. Inspect source structure and define exact extraction boundaries. **Complete.**
5. Implement and test deterministic extraction and splitting. **Read-only preflight and deterministic in-memory extraction accepted; filesystem publication not authorized.**
6. Inspect corpus characteristics and audit validation/test Unicode code points against the training-only character inventory. **In-memory character inventory accepted and tracked; occurrence-location audit not implemented.**

**Exit criteria:**

- A small, approved learning dataset is selected with source and licensing recorded.
- Raw and processed data boundaries are explicit.
- Loading, inspection, splitting, and deterministic reproduction are implemented and tested.
- Data characteristics and limitations are understood and documented.

## Phase 2 — Tokenization

**Goal:** Understand and implement the mapping between text and token IDs.

**Exit criteria:**

- An intentionally simple tokenizer is implemented without an opaque tokenizer framework.
- Vocabulary construction, encoding, decoding, unknown cases, and round trips are explained.
- Tests cover representative and edge cases.
- Tokenization statistics are inspected on the Phase 1 dataset.

## Phase 3 — Embeddings

**Goal:** Understand learned token and positional representations.

**Exit criteria:**

- Token embeddings and an explicit positional representation are implemented.
- Tensor shapes and parameter roles are explainable and tested.
- Small examples demonstrate lookup, batching, and gradient flow.

## Phase 4 — Simple Neural Language Model

**Goal:** Build and train a small non-attention baseline to understand next-token prediction.

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
