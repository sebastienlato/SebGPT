# Session Protocol

This protocol makes project sessions deterministic and keeps the repository sufficient for continuity.

## Beginning of session

1. Read `PROJECT_STATE.md` first. Identify the current phase, milestone, last verified state, known issues, constraints, and next exact step.
2. Read `PROJECT_CHARTER.md` whenever purpose, scope, learning philosophy, or success criteria are relevant.
3. Read the current phase in `ROADMAP.md`, including its exit criteria. Do not assume the next phase has started merely because related code exists.
4. Read the entries in `DECISIONS.md` relevant to the proposed work. Later entries supersede earlier ones only when they say so explicitly.
5. If training, evaluation, or experimentation is involved, read the latest relevant entry in `EXPERIMENT_LOG.md` and inspect its referenced artifacts.
6. Inspect the repository tree, relevant code and tests, and Git status before making changes. Preserve unrelated work and uncommitted user changes.
7. Compare the request with the current milestone and constraints. Confirm the current state before significant changes; if the work would skip a phase or alter architecture, stop and obtain an explicit decision.

## During session

- Keep work inside the current approved phase and milestone.
- Make consequential choices explicit. Append an entry to `DECISIONS.md` rather than silently embedding a decision in code.
- Prefer understandable, inspectable code and small tests that expose the concept being learned.
- Record reproducibility information as an experiment is designed or run; do not rely on memory at session end.
- If repository evidence conflicts with chat context, surface the conflict. Do not silently choose one.
- Update the current plan when verification changes what is known.

## End of session

1. Run the checks relevant to the work and record what was actually verified. Do not describe an unrun check as passing.
2. If an experiment was planned or run, append its entry or a new status update to `EXPERIMENT_LOG.md`; never rewrite an existing entry.
3. If an architectural or process decision was made, append it to `DECISIONS.md` with rationale, alternatives, and consequences.
4. Update `LEARNING_NOTES.md` only for concepts actually covered with Sebastien, including unresolved questions.
5. Rewrite `PROJECT_STATE.md` so it concisely reflects now: current phase and milestone, completed and current work, model status, last verified state, next exact step, known issues, constraints, open questions, and session handoff.
6. Ensure the next exact step is a single actionable step, not a vague goal or a list of possibilities.
7. Provide a concise handoff stating exactly what changed, what was verified, decisions or assumptions made, and what should happen next.

## Document ownership rules

- `PROJECT_CHARTER.md`: stable purpose and principles; change only through an explicit recorded decision.
- `PROJECT_STATE.md`: current truth; replace stale state rather than keeping a chronology.
- `ROADMAP.md`: ordered learning plan and phase exit criteria.
- `DECISIONS.md`: append-only durable decisions.
- `LEARNING_NOTES.md`: concepts actually learned, in human-readable language.
- `EXPERIMENT_LOG.md`: append-only reproducibility index for experiments and runs.
- `README.md`: concise orientation and only commands that currently exist and have been verified.
