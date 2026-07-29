"""Public v1 API for the offline car-insurance pricing mapper."""

from pricing_mapper.config import (
    AcquisitionConfig,
    ArtifactConfig,
    DomainConfig,
    EarlyStoppingConfig,
    EvaluationConfig,
    MapperConfig,
    ModelConfig,
    ProviderConfig,
    SamplingConfig,
    load_config,
)
from pricing_mapper.domain import CarQuoteInput, DomainSpec
from pricing_mapper.engine import PricingEngine
from pricing_mapper.exceptions import (
    ArtifactError,
    ConfigurationError,
    DomainValidationError,
    LegacyArtifactError,
    PersistenceError,
    PricingMapperError,
    ProviderError,
    ProviderRejected,
    ProviderUnavailable,
)
from pricing_mapper.orchestration import MappingResult, MappingRun
from pricing_mapper.types import Prediction, QuoteProvider

__all__ = [
    "AcquisitionConfig",
    "ArtifactConfig",
    "ArtifactError",
    "CarQuoteInput",
    "ConfigurationError",
    "DomainConfig",
    "DomainSpec",
    "DomainValidationError",
    "EarlyStoppingConfig",
    "EvaluationConfig",
    "LegacyArtifactError",
    "MapperConfig",
    "MappingResult",
    "MappingRun",
    "ModelConfig",
    "PersistenceError",
    "Prediction",
    "PricingEngine",
    "PricingMapperError",
    "ProviderConfig",
    "ProviderError",
    "ProviderRejected",
    "ProviderUnavailable",
    "QuoteProvider",
    "SamplingConfig",
    "load_config",
]

__version__ = "1.0.0"
