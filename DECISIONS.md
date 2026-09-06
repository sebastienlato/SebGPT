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

## DEC-0007 — First corpus source, manifest, and leakage boundary

- **Date:** 2026-09-06
- **Status:** Accepted
- **Decision:** Build the first corpus from the UTF-8 plain-text artifact for Project Gutenberg eBook #100, *The Complete Works of William Shakespeare*. Target approximately 170,000 words using a fixed eight-play manifest. Train on *Hamlet*, *Romeo and Juliet*, *Macbeth*, *A Midsummer Night's Dream*, *Much Ado About Nothing*, and *Henry V*; validate on *The Tempest*; and reserve *Twelfth Night* for test. Split only at whole-work boundaries. Derive the initial vocabulary from training data only, audit validation/test for unseen characters before tokenization, and report those characters without modifying or adding them.
- **Why:** A single Gutenberg source provides consistent editorial conventions and clearer provenance than the canonical third-party Tiny Shakespeare copy. Multiple plays provide about one million characters of controlled genre variety, while whole-work evaluation tests transfer to unseen text and avoids overlapping-passage leakage. Training-only vocabulary construction prevents validation/test information from influencing learned preprocessing.
- **Alternatives considered:** Use the canonical Tiny Shakespeare file; use one Project Gutenberg novel; use a TinyStories subset; download separate Shakespeare editions; split a concatenated corpus by random or contiguous character ranges; derive the character vocabulary from all splits.
- **Consequences:** Validation and test measurements will be noisier and harder than within-work evaluation. Source acquisition, extraction, counts, hashing, and unseen-character handling require explicit later gates. Validation/test-only characters remain unchanged and may cause an unresolved representation problem until Phase 2 makes a visible decision.

## DEC-0008 — Raw-source preservation and Phase 1 boundary clarifications

- **Date:** 2026-09-06
- **Status:** Accepted; amends DEC-0006 and clarifies DEC-0007
- **Decision:** Git-track exactly `data/raw/gutenberg-ebook-100/complete-works.txt`, the unmodified small public-domain plain-text master acquired for the approved Phase 1 corpus. This is the sole exception to DEC-0006's general dataset-ignore policy and does not authorize any other raw source, processed dataset, generated metadata, or future dataset. Preserve its bytes from Git line-ending conversion. Use `docs/data/shakespeare-eight-play-manifest.json` as the authoritative machine-readable record for concrete provenance and processing facts. Phase 1 defines a training-only Unicode character inventory, not a token vocabulary or tokenizer. Automated pre-evaluation access to *Twelfth Night* is limited to approved structural markers and machine-readable code-point diagnostics without surrounding prose; Phase 2 must choose a generic unseen-input policy that is not tailored to observed test characters.
- **Why:** A mutable upstream URL plus a hash can verify bytes but cannot recover them. The narrow Git exception makes the exact educational source reproducible for another person without weakening the default large-data policy. Precise character terminology and test-access limits preserve the Phase 2 learning gate and reduce evaluation leakage.
- **Alternatives considered:** Keep the raw file ignored and rely on the mutable upstream URL; use an external content-addressed archive; broadly allow raw datasets in Git; treat the Phase 1 character inventory as a tokenizer vocabulary; expose test context during the unseen-character audit.
- **Consequences:** The raw master and completed provenance manifest must be committed together after authorized acquisition and verification. All processed data and unlisted dataset files remain ignored. Generated local metadata is subordinate to the tracked manifest. Extraction rules still require a post-acquisition design review, and no tokenization or unseen-input representation decision is made in Phase 1.
