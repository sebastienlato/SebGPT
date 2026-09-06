# Instructions for Coding Agents

SebGPT is a learning-first educational project. The repository, not prior chat context, is the authority for project state.

## Required beginning-of-work sequence

Before changing anything:

1. Read `PROJECT_STATE.md`.
2. Read the current phase and exit criteria in `ROADMAP.md`.
3. Read `PROJECT_CHARTER.md` when project purpose or scope is relevant.
4. Read relevant entries in `DECISIONS.md`.
5. When training or experiments are involved, read the latest relevant entry in `EXPERIMENT_LOG.md`.
6. Inspect the current code, tests, project tree, and Git state.
7. Confirm that the requested work belongs to the current phase before making significant changes.

## Working rules

1. Never skip educational phases or their exit criteria.
2. Never silently change the model architecture, training objective, tokenization strategy, dataset, or other consequential design choice.
3. Do not replace educational implementations with opaque abstractions.
4. Keep core machine-learning code small, readable, inspectable, and explainable.
5. Use PyTorch only at the abstraction level permitted by the current learning phase.
6. Do not introduce pretrained models, hosted model APIs, high-level model frameworks, or unrelated infrastructure without explicit approval and a recorded decision.
7. Make the smallest coherent change that advances the current milestone.
8. Test changes in proportion to their risk, including educational invariants such as shapes, causality, determinism, and gradient flow when relevant.
9. Preserve experiment reproducibility: record configuration, data identity, seeds, environment, code/version reference, metrics, artifacts, and conclusions.
10. Treat `DECISIONS.md` and `EXPERIMENT_LOG.md` as append-only. Supersede or correct entries by appending new entries.
11. Update `LEARNING_NOTES.md` only for concepts actually covered with Sebastien.
12. Keep `PROJECT_STATE.md` concise and current rather than accumulating project history.
13. Do not claim a phase or milestone is complete until its exit criteria are verified.
14. Preserve user changes and avoid destructive operations.

## Required end-of-work sequence

Before handing off:

1. Verify code and tests when applicable.
2. Record experiments when applicable.
3. Append architectural decisions when applicable.
4. Update learning notes only when a concept was actually covered.
5. Update `PROJECT_STATE.md` with the verified current state, known issues, and one next exact step.
6. Report exactly what changed, what was verified, and any assumptions or unrequested decisions.
7. Leave a concise session handoff that another session can follow without chat history.

Follow the detailed procedure in `docs/SESSION_PROTOCOL.md`.
