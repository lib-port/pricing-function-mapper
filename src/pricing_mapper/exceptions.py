"""Public exception hierarchy for pricing-mapper v1."""

from __future__ import annotations


class PricingMapperError(Exception):
    """Base class for all library-specific failures."""


class ConfigurationError(PricingMapperError, ValueError):
    """The v1 TOML configuration is invalid."""


class DomainValidationError(PricingMapperError, ValueError):
    """A quote input is malformed or outside the configured domain."""


class ProviderError(PricingMapperError):
    """Base class for quote-provider failures."""


class ProviderUnavailable(ProviderError):
    """A transient provider failure that may be retried."""


class ProviderRejected(ProviderError):
    """A permanent provider rejection that must not be retried."""


class AdvisorError(PricingMapperError, RuntimeError):
    """The opt-in acquisition advisor failed closed."""


class AdvisorUnavailable(AdvisorError):
    """The configured advisor endpoint could not complete a request."""


class AdvisorValidationError(AdvisorError):
    """The advisor returned untrusted or contract-invalid data."""


class AdvisorModelError(AdvisorError):
    """The installed advisor model or runtime differs from its pin."""


class PersistenceError(PricingMapperError):
    """The durable run journal is missing, corrupt, or incompatible."""


class ArtifactError(PricingMapperError):
    """A v1 artifact is incomplete, corrupt, or incompatible."""


class LegacyArtifactError(ArtifactError):
    """A v0 pickle or JSON-state artifact was supplied to a v1 operation."""
