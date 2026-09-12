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

### EXP-20260912-01 — Phase 8 fixed training

- **Experiment ID:** `EXP-20260912-01`
- **Entry kind:** `planned`
- **Recorded at UTC:** `2026-09-12T17:08:32Z`
- **Status:** `planned`
- **Question:** `Does the fixed Phase 8 run satisfy the accepted training-loss improvement predicate while producing valid latest and best-validation checkpoints and passing exact resume?`
- **Authorization:** `Master Chat authorized planned-run pre-registration proposal creation only; real run remains unauthorized`
- **Code commit:** `809834323d53407cb4a54ae539585bb3d78856eb`
- **Phase 7 authority:** {"mini_gpt_source_sha256":"6b8db96577a0ad1f5daa4649d7ca9375e59873d4b635c3f76e75a6751f74f12d","mini_gpt_spec_sha256":"3e1987658d4c9ece59bebbea7061f939c1138aaa73751a523f659f8b3217c022","mini_gpt_test_sha256":"95ec62d1164c48525e59e33f3f90fada6d482c84a31ca16edba38178b16b2cab","phase7_closure_commit":"33d4510421107848c4aa8a6014f4a7b1e391065a","phase7_contract_commit":"60b2a9cce55da79ccc9fbd03fad014cb2a939290","phase7_implementation_commit":"3139b1736f005fe903e2ea111d91934478b5a683"}
- **Phase 8 contract authority:** {"phase8_contract_commit":"c50d77ac935bdf924b9b429a5776c419982a5d11","phase8_spec_sha256":"1630f9c7a113ff4af6db709ba2c356e8dab450eb4d918a54d64b734084e70efd"}
- **Runtime identity:** {"cpu_autocast_enabled":false,"default_device":"cpu","default_dtype":"torch.float32","deterministic_algorithms":true,"deterministic_warn_only":false,"device":"cpu","float32_matmul_precision":"highest","grad_mode_enabled":true,"inference_mode_enabled":false,"machine":"arm64","mkldnn_enabled":false,"operating_system":"Darwin","operating_system_release":"25.6.0","parameter_dtype":"torch.float32","processor":"arm","python_implementation":"CPython","python_version":"3.14.4","requirements_lock_sha256":"8ddf7a11bf67d36bce8ba58abffc5416a0fb778cdc20d03c077d318715122cfa","schema_version":1,"token_dtype":"torch.long","torch_build_config_sha256":"7087d129a9b4d0dad0f49ccb08b46952ceac534033de37a2791e4a7870f75bd2","torch_inter_op_threads":1,"torch_intra_op_threads":1,"torch_parallel_info_sha256":"f374b265f279a73db67700c37cada26235861d10ef91328bf12a992806c43b61","torch_version":"2.14.0"}
- **Dataset authority:** {"dataset_id":"shakespeare-eight-play","manifest_sha256":"157e324c41c6aee756b9c554ae465388a892ea9a0f8fb8296ef0d986a0c9f6fb","processing_manifest_sha256":"bbf938e565022dde72f26470e2fa7214f2fe1735afbc62ace320f1e6ebf372cc","training_works":[{"manifest_order":1,"processed_sha256":"281070a10b0fe06b877a6e9342cedecad1e5f65657d58fc51b14ce378c70a243","split":"train","work_id":"hamlet"},{"manifest_order":2,"processed_sha256":"59bfe35a3f7ccb63353aa4b8823f1a33fca4bd27c47943515ecc5fb8d6775775","split":"train","work_id":"romeo-and-juliet"},{"manifest_order":3,"processed_sha256":"fb24ddc7c0c35f7e7d6e0989858e3d9cdb30d00208f2122938d7cd900a3699c6","split":"train","work_id":"macbeth"},{"manifest_order":4,"processed_sha256":"a9314798205c72c7cb7988813eb9df3d36e9731e45d882f761c907f31e178bd1","split":"train","work_id":"a-midsummer-nights-dream"},{"manifest_order":5,"processed_sha256":"fbb46df4f5297329599a4e58b160d7182b30262b1c497c0de4002e27f536ce7b","split":"train","work_id":"much-ado-about-nothing"},{"manifest_order":6,"processed_sha256":"eb8c7e711ac4182800d5b6280d1cf5c0c1df98f43c727d48cd04ca8288321560","split":"train","work_id":"henry-v"}],"validation_works":[{"manifest_order":7,"processed_sha256":"68a090b9d905967817948397410045f00592e4ea80b4115a13c860f67e1ca617","split":"validation","work_id":"the-tempest"}]}
- **Tokenizer authority:** {"artifact_path":"artifacts/tokenizers/shakespeare-code-point-v1/vocabulary.json","artifact_sha256":"9f4235f9dab3e0361221a90c5ae7ca7fee540760fcc1708964368e6f17d6d70e","implementation_commit":"de7a7f096fbbd8607c944412eaef30be9b686b56","schema_version":1,"tokenizer_id":"shakespeare-code-point-v1","vocabulary_size":81}
- **Model configuration:** {"block_parameter_seeds":[7001,7002,7003,7004],"dropout_probability":0.1,"embedding_seed":1337,"head_width":8,"hidden_width":128,"layer_norm_epsilon":1e-05,"max_sequence_length":256,"model_width":32,"number_of_blocks":4,"number_of_heads":4,"output_head_seed":7005,"parameter_count":63825,"parameter_device":"cpu","parameter_dtype":"torch.float32","trainable_tensor_count":90}
- **Training configuration:** {"allow_final_partial_batch":true,"amsgrad":false,"best_comparison":"strict_lower","betas":[0.9,0.999],"capturable":false,"catalog_schema_version":1,"checkpoint_schema_version":1,"context_length":256,"differentiable":false,"epsilon":1e-08,"evaluate_initialized_state":true,"evaluation_interval_epochs":1,"evaluation_split_order":["train","validation"],"foreach":false,"fused":false,"gradient_clip_epsilon":1e-06,"gradient_clip_max_norm":1.0,"gradient_clip_norm_type":2.0,"learning_rate":0.0003,"logical_batch_capacity":8,"maximize":false,"maximum_catalog_bytes":16384,"maximum_checkpoint_object_bytes":67108864,"maximum_epochs":10,"optimizer_name":"AdamW","order_algorithm":"torch_randperm_local_cpu_generator","order_seed":8001,"parameter_device":"cpu","parameter_dtype":"torch.float32","retain_unpadded_tail":true,"runtime":{"cpu_autocast_enabled":false,"default_device":"cpu","default_dtype":"torch.float32","deterministic_algorithms":true,"deterministic_warn_only":false,"device":"cpu","float32_matmul_precision":"highest","grad_mode_enabled":true,"inference_mode_enabled":false,"machine":"arm64","mkldnn_enabled":false,"operating_system":"Darwin","operating_system_release":"25.6.0","parameter_dtype":"torch.float32","processor":"arm","python_implementation":"CPython","python_version":"3.14.4","requirements_lock_sha256":"8ddf7a11bf67d36bce8ba58abffc5416a0fb778cdc20d03c077d318715122cfa","schema_version":1,"token_dtype":"torch.long","torch_build_config_sha256":"7087d129a9b4d0dad0f49ccb08b46952ceac534033de37a2791e4a7870f75bd2","torch_inter_op_threads":1,"torch_intra_op_threads":1,"torch_parallel_info_sha256":"f374b265f279a73db67700c37cada26235861d10ef91328bf12a992806c43b61","torch_version":"2.14.0"},"scheduler_name":null,"schema_version":1,"stride":256,"token_dtype":"torch.long","validation_early_stopping":false,"warmup_steps":0,"weight_decay":0.01}
- **Checkpoint configuration:** {"catalog_schema_version":1,"checkpoint_schema_version":1,"maximum_catalog_bytes":16384,"maximum_checkpoint_object_bytes":67108864}
- **Feasibility evidence:** {"classification":"FEASIBLE WITH MATERIAL RUNTIME COST","estimated_complete_epoch_cycle_seconds":144.41,"estimated_runner_compute_minutes_excluding_checkpoint_io":27.58,"estimated_training_epoch_seconds":78.44,"estimated_training_evaluation_seconds":58.69,"estimated_validation_evaluation_seconds":7.28,"measured_peak_rss_mb":336.6,"scope":"planning measurement only; no training occurred"}
- **Success predicate:** `min(training_loss at completed epochs 1..10) < initialized training_loss`
- **Sealed-test policy:** `none`
- **Generation policy:** `none`
- **Next gate:** `Gate 66 fresh independent pre-registration review`
