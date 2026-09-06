# SebGPT Project Charter

## Purpose

SebGPT exists to help Sebastien understand how modern language models work by progressively building a small GPT-style language model from first principles in Python and PyTorch.

The primary outcome is understanding. Model quality, scale, and convenience are secondary until the relevant ideas have been learned and implemented transparently.

## Philosophy

- Learn each core concept before hiding it behind an abstraction.
- Prefer small, inspectable implementations over sophisticated opaque ones.
- Introduce complexity only when the current phase's foundations are understood and verified.
- Use experiments to connect theory, code, observations, and conclusions.
- Keep the repository understandable enough that a future session can resume without relying on chat history.
- Treat failures and surprising results as learning material, not merely obstacles.

## Scope

The core project will progressively cover:

- datasets and data preparation
- tokenization
- embeddings
- language-modeling objectives and loss
- gradient descent and backpropagation
- attention, self-attention, and multi-head attention
- Transformer blocks
- a complete mini-GPT
- training and checkpointing
- inference, generation, and evaluation
- interpretability and inspection of internal representations

## Learning-first constraints

- Do not use pretrained models or external model APIs as a substitute for building the model.
- Do not use high-level frameworks that conceal a concept currently being learned.
- PyTorch may provide tensors, autograd, optimization, and low-level neural-network primitives.
- Implement the educational model architecture within this repository before considering higher-level equivalents.
- Do not silently delegate architectural choices to tools or coding agents.
- Do not skip phases merely because a later implementation is faster or more capable.
- Keep core machine-learning code readable, explainable, and testable.

## Out of scope for the initial project

- competing with production-scale or frontier language models
- wrapping an existing pretrained model or hosted model API
- premature optimization for scale or hardware
- instruction tuning or a conversational product before the base language model is understood
- production deployment before the educational milestones are complete

These items may be reconsidered only in an explicitly approved later phase.

## Authority and continuity

This repository is the authoritative source of project truth. Chat history can provide context, but it must not be the only record of current state, decisions, experiments, or next steps.

- `PROJECT_STATE.md` records the current state and exact next action.
- `ROADMAP.md` defines phases and their exit criteria.
- `DECISIONS.md` preserves architectural decisions append-only.
- `LEARNING_NOTES.md` records concepts actually covered.
- `EXPERIMENT_LOG.md` indexes reproducible experiments and training runs.
- `docs/SESSION_PROTOCOL.md` defines how sessions begin and end.

## Long-term success criteria

SebGPT succeeds when Sebastien can:

1. Explain the path from raw text to tokens, embeddings, logits, probabilities, and generated text.
2. Explain and inspect loss, gradients, parameter updates, attention, Transformer blocks, and autoregressive generation.
3. Implement and test a complete small GPT-style language model whose core architecture is not hidden behind a high-level model framework.
4. Train, checkpoint, restore, generate from, and evaluate the model reproducibly.
5. Inspect internal representations and use evidence to reason about model behavior.
6. Reconstruct why important design choices were made from repository documentation.
7. Resume work safely after a long gap by following the repository's continuity process.

## Change policy

This charter is intended to be stable. Material changes to purpose, philosophy, scope, constraints, or success criteria require an explicit decision recorded in `DECISIONS.md` and a corresponding charter update.
