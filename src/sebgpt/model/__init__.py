"""Transparent model primitives introduced phase by phase."""

from sebgpt.model.embeddings import (
    EmbeddingContractError,
    EmbeddingTypeError,
    TokenPositionEmbedding,
)
from sebgpt.model.simple_language_model import (
    Phase4ContractError,
    Phase4GovernanceError,
    Phase4TypeError,
    SimpleNeuralLanguageModel,
    clear_gradients,
    explicit_cross_entropy,
    manual_sgd_step,
    measure_example_loss,
    model_parameter_digest,
    probabilities_from_logits,
    scale_backward_loss,
)


__all__ = [
    "EmbeddingContractError",
    "EmbeddingTypeError",
    "Phase4ContractError",
    "Phase4GovernanceError",
    "Phase4TypeError",
    "SimpleNeuralLanguageModel",
    "TokenPositionEmbedding",
    "clear_gradients",
    "explicit_cross_entropy",
    "manual_sgd_step",
    "measure_example_loss",
    "model_parameter_digest",
    "probabilities_from_logits",
    "scale_backward_loss",
]
