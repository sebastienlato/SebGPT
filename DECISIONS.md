# Architectural Decision Log

This is an append-only record. Add new entries to supersede earlier decisions; do not delete or silently rewrite historical entries. Corrections should be added as clearly labeled amendments.

## DEC-0001 — Learning-first implementation strategy

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Build a small GPT-style language model progressively from first principles in Python and PyTorch. PyTorch may supply tensors, autograd, optimization, and low-level neural-network primitives, but the educational architecture will be implemented explicitly before higher-level frameworks are considered.
- **Why:** The project's purpose is to understand modern language models, not to maximize model capability or implementation speed.
- **Alternatives considered:** Wrap a pretrained model or API; use a high-level training/model framework; begin with a complete Transformer and explain it afterward.
- **Consequences:** Progress will be deliberately phased. Some standard components will be reimplemented for clarity. Core architecture changes require explicit discussion and a new decision entry.

## DEC-0002 — Repository as continuity authority

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Treat repository documentation and code as the authoritative source of truth across sessions, with `PROJECT_STATE.md` as the primary current-state anchor.
- **Why:** The project is expected to span many weeks or months and multiple Work and Codex sessions; chat history alone is not a durable or deterministic handoff mechanism.
- **Alternatives considered:** Rely on conversation history; maintain a single unstructured project note; rely on code alone.
- **Consequences:** Sessions must follow `docs/SESSION_PROTOCOL.md`. Current state, durable decisions, learning notes, and experiments have distinct documents and update rules.

## DEC-0003 — Phase gates prevent premature implementation

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Use the ordered phases and exit criteria in `ROADMAP.md` as learning gates. Do not start later-phase implementation until the current phase is verified and its completion is recorded.
- **Why:** A working advanced implementation could conceal gaps in the foundational concepts the project exists to teach.
- **Alternatives considered:** Build a complete mini-GPT immediately and study components afterward; allow agents to choose implementation order opportunistically.
- **Consequences:** Work may pause for review between phases. The exact next step must remain explicit in `PROJECT_STATE.md`, and exceptions require a recorded decision.

## DEC-0004 — Native Python.org runtime and project-local virtual environment

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Use the existing native ARM64 Python.org CPython 3.14.4 installation to create a `.venv` virtual environment in the project root. Do not add a Python version manager while the installed interpreter remains suitable.
- **Why:** The interpreter is already installed, native to the Apple Silicon architecture, includes `venv` and pip, and is supported by the selected stable PyTorch wheel. A project-local environment provides dependency isolation without introducing another management layer.
- **Alternatives considered:** Use Apple's Xcode-managed Python 3.9.6; install Homebrew Python; install pyenv, uv, Conda, or Mamba.
- **Consequences:** Development commands run through `.venv`, which is excluded from Git. Reproducing this verified environment requires CPython 3.14.4; a future interpreter change requires a new decision and full verification.

## DEC-0005 — Minimal pinned dependency workflow

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Declare only `torch==2.14.0` in `requirements.in` and record the complete resolved environment in `requirements.lock`. Use Python's standard-library `unittest` framework initially. Do not add NumPy or other optional packages until an approved learning phase requires them.
- **Why:** Separating the direct requirement from the resolved lock keeps dependency intent visible while allowing exact environment reproduction. PyTorch 2.14.0 provides a stable CPython 3.14, macOS 14+, ARM64 wheel and supplies the low-level tensor, autograd, optimization, and device features SebGPT needs.
- **Alternatives considered:** Keep only an unpinned requirements file; use only a full `pip freeze`; introduce a packaging or lock-file tool; add common ML, notebook, plotting, or testing packages preemptively.
- **Consequences:** PyTorch's transitive packages are pinned even though they are not direct SebGPT choices. PyTorch may warn that optional NumPy interoperability is unavailable; this does not affect the verified tensor, autograd, CPU, or MPS operations. Dependency changes require explicit approval and lock regeneration.

## DEC-0006 — Git baseline and generated-artifact policy

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Initialize the repository on a `main` branch after the Phase 0 environment is verified, then create one baseline commit containing the project scaffold, documentation, dependency manifests, and environment test. Exclude the virtual environment, caches, local datasets, checkpoints, model-weight files, and generated experiment artifacts.
- **Why:** The baseline gives later educational work a clean, reviewable starting point without storing machine-local or large generated content.
- **Alternatives considered:** Commit before environment verification; defer Git initialization; commit the virtual environment or generated artifacts.
- **Consequences:** Empty source directories use tracked `.gitkeep` placeholders. Dataset provenance and experiment metadata may later be committed, but downloaded data, checkpoints, weights, and generated artifacts remain untracked.
