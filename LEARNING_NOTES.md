# Learning Notes

This document grows only as concepts are actually covered with Sebastien. It is not a prewritten textbook or a list of topics merely planned for later.

## Entry format

For each concept covered, record:

- **Date**
- **Concept**
- **Explanation in our own words**
- **Small example or observation**
- **What remains unclear**
- **Related code, test, decision, or experiment**

## Concepts covered

### 2026-09-06 — Dataset suitability and leakage-resistant splits

- **Explanation in our own words:** A language model learns next-token patterns from the distribution it is shown, so a dataset defines the language, structure, subject matter, and limitations the model can learn. A useful first dataset is legally and technically traceable, coherent, small enough to inspect, large enough to contain repeated patterns, and reproducible from immutable raw material. Training data teaches parameters; validation data supports development choices; test data is reserved for a final unbiased check.
- **Small example or observation:** If lines from one play are randomly placed into every split, validation can contain nearby or repeated material from a play the model trained on. Assigning complete plays to one split each instead asks whether patterns learned from six plays transfer to an unseen play.
- **What remains unclear:** The actual source counts and exact extraction markers are unknown until acquisition. Validation/test may contain Unicode code points absent from training; Phase 1 will report them unchanged under a sealed-test diagnostic contract, and Phase 2 will choose a generic unseen-input policy.
- **Related code, test, decision, or experiment:** `docs/DATASET_SPEC.md`; DEC-0007 and DEC-0008. No data or implementation exists yet.
