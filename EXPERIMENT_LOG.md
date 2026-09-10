# Experiment Log

This is an append-only index of experiments and training runs. Do not rewrite existing entries; append a status update, correction, or follow-up entry when needed. Store substantial run artifacts under `experiments/` and checkpoints under `checkpoints/` rather than embedding them here.

## Experiment ID convention

Use `EXP-YYYYMMDD-NN`, where `NN` is a two-digit sequence for that date.

## Entry template

### EXP-YYYYMMDD-NN — Short descriptive name

- **Date:**
- **Status:** planned | running | completed | failed | stopped
- **Question or hypothesis:**
- **Code/version reference:**
- **Random seed(s):**
- **Model configuration:**
- **Dataset and data version:**
- **Parameter count:**
- **Training configuration:**
- **Hardware/device:**
- **Initial loss:**
- **Final loss:**
- **Generated samples or artifact paths:**
- **Observations:**
- **Conclusions:**
- **Follow-up:**

## Experiments

No experiments have been run.

_Historical pre-run state; superseded by the append-only completed entry below._

### EXP-20260909-01 — Fixed Phase 4 positionwise baseline

- **Date:** `2026-09-10T01:26:06Z` (UTC, second resolution)
- **Status:** completed; pre-registered result `PASS`
- **Question or hypothesis:** Does the fixed two-pass positionwise baseline lower aggregate training loss below both `ln(81)` and its initialized value?
- **Authorization and predecessor:** Fresh one-shot execution was authorized from the clean, remotely verified safe-corpus checkpoint. An earlier launcher attempt stopped before imports with `ModuleNotFoundError: No module named 'sebgpt'`; it invoked the runner zero times, constructed no corpus or model, consumed no stochastic model state, accessed no sealed content, and produced no experiment result. Master-chat classified it as a pre-experiment launcher abort and explicitly authorized this single corrected launch. The pre-registration was not amended and there is no failed-run predecessor.
- **Code/version reference:** runner execution checkpoint `ba09d017e97a6d25317e160fc1a40a6304bcdd96`; accepted fixed-runner checkpoint `a4da629a59584a0185a2ec286b0e770b7940940e`; inherited model/loss/update checkpoint `497ecde3577677903f14669722d61dcdf8caa1d6`; shifted/governance checkpoint `266797f891e9980b57bb35a633c91bef838d4111`; Phase 4 contract checkpoint `3fcf007ce819ca1a45aa75b48fa19f10311f2819`; embedding decision `DEC-0015`; embedding contract `0e458cc8b8bce9d8e89b23abbb8ce6385669b7d5`; embedding implementation `68b47bb404d55357cadce35b97036c72c5876d62`; tokenizer implementation `de7a7f096fbbd8607c944412eaef30be9b686b56`.
- **Runtime:** Python `3.14.4`; PyTorch `2.14.0`; `macOS-26.6.2-arm64-arm-64bit-Mach-O`; machine `arm64`; device `cpu`; dtype `torch.float32`.
- **Random seeds and initialization:** embedding seed `1337`; output-head seed `4004`; token and position embeddings initialized `Normal(0,1/sqrt(32))`; output weight initialized `Normal(0,1/sqrt(32))`; output bias initialized to zero; initialization order `token_then_position_then_output_weight_then_bias`; separate local CPU generators left the global RNG unchanged.
- **Model configuration:** `SimpleNeuralLanguageModel`; architecture `positionwise_linear_non_attention`; vocabulary size `81`; embedding dimension `32`; maximum positions `256`; context/stride `64/64`; variable tail enabled; no padding; no weight tying; exactly four trainable tensors: `output_weight (32,81)`, `output_bias (81,)`, `representation.token_embeddings (81,32)`, and `representation.position_embeddings (256,32)`; exactly `13,457` trainable parameters.
- **Vocabulary authority:** `shakespeare-code-point-v1`, schema `1`, artifact `artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json`, SHA-256 `9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e`.
- **Dataset and provenance:** dataset `shakespeare-eight-play`; Phase 1 manifest SHA-256 `157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb`; processing-manifest SHA-256 `bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc`. The safe factory loaded only Hamlet `281070a10b0fe06b877a6e9342cedecad1e5f65657d58fc51b14ce378c70a243`, Romeo and Juliet `59bfe35a3f7ccb63353aa4b8823f1a33fca4bd27c47943515ecc5fb8d6775775`, Macbeth `fb24ddc7c0c35f7e7d6e0989858e3d9cdb30d00208f2122938d7cd900a3699c6`, A Midsummer Night's Dream `a9314798205c72c7cb7988813eb9df3d36e9731e45d882f761c907f31e178bd1`, Much Ado About Nothing `fbb46df4f5297329599a4e58b160d7182b30262b1c497c0de4002e27f536ce7b`, Henry V `eb8c7e711ac4182800d5b6280d1cf5c0c1df98f43c727d48cd04ca8288321560`, and validation work The Tempest `68a090b9d905967817948397410045f00592e4ea80b4115a13c860f67e1ca617`.
- **Training configuration:** manifest order then ascending start; no shuffle; one example at a time; manual SGD with learning rate `0.05`; gradients set to `None` before each example and after each update; mean loss scaled by `L/64` for backward; aggregate losses target-token weighted using `math.fsum`; exactly two passes.
- **Observed counts:** training source tokens `792,705`, targets `792,699`, examples `12,389` per pass; validation source tokens `98,296`, targets `98,295`, examples `1,536`; pass 1 updates `12,389`; pass 2 updates `12,389`; total updates `24,778`.
- **Initial identity and gradient state:** one model constructed; model identity `4565272288`; parameter identities `(4565274304, 4565274976, 4564333360, 4564334064)`; all four gradients were `None` at measurement boundaries; initial canonical raw-byte parameter digest `969c9f4606ff823a027324cb5ba6bdb76e72da8f4f903c42958703e0a233db20`.
- **Initial loss:** training `4.426503102003006`; validation `4.427434147633409`.
- **Final loss:** training `2.5551429421586387`; validation `2.565088013405899`.
- **Baseline and predicates:** exact uniform baseline `math.log(81) = 4.394449154672439`; `final_training_loss < math.log(81)` is `true`; `final_training_loss < initial_training_loss` is `true`; combined pre-registered result `PASS`.
- **Validation observation:** `improved`; validation is observation only and is not part of the primary predicate.
- **Identity and no-update measurement proofs:** model and parameter identities remained stable. Final parameter digest `c8ce1ae08c4c3466d74b30068fb6b77eaf2142c28a5b914d64bf0277d63bac72`. Initial-training and initial-validation measurements each preserved digest `969c9f4606ff823a027324cb5ba6bdb76e72da8f4f903c42958703e0a233db20`; final-training and final-validation measurements each preserved digest `c8ce1ae08c4c3466d74b30068fb6b77eaf2142c28a5b914d64bf0277d63bac72`; measurement-boundary gradients were all `None`.
- **Lifecycle:** `model_constructed`, `initial_training_measured`, `initial_validation_measured`, `training_pass_1_completed`, `training_pass_2_completed`, `final_training_measured`, `final_validation_measured`.
- **Sealed test and artifacts:** *Twelfth Night* was not accessed. No sealed-test metric exists. No model, checkpoint, logits, probabilities, gradients, token/window/target data, or other experiment artifact was retained.
- **Observations:** Final training loss was below both `ln(81)` and its initialized value. Validation loss improved observationally. The corrected launcher emitted the accepted warning that optional NumPy interoperability is unavailable; NumPy was not required and nothing was installed or changed.
- **Conclusions:** The fixed run met its pre-registered training-loss predicate. This does not establish generalization or test performance.
- **Follow-up:** Proceed only to Gate 25 independent experiment and Phase 4 exit review; do not begin Phase 5.
- **Post-result verification:** Safe corpus `14/14`; orchestration `41/41`; model/loss/update `48/48`; shifted-example/governance `44/44`; complete suite `304` total with `303` passes, one expected restricted-context MPS skip, and zero failures. Source and test bytes remained unchanged; `git diff --check` and final artifact/status checks are recorded in the session handoff.
