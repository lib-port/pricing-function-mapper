"""Stable public protocols and output types."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pricing_mapper.domain import CarQuoteInput


@runtime_checkable
class QuoteProvider(Protocol):
    """Trusted local callable contract used by mapping runs."""

    def __call__(self, quote: CarQuoteInput) -> float:
        """Return a finite, non-negative premium or raise a provider exception."""


class Prediction(BaseModel):
    """One calibrated premium prediction."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    premium: float = Field(ge=0.0)
    lower: float = Field(ge=0.0)
    upper: float = Field(ge=0.0)
    model_version: str
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not self.model_version.strip():
            raise ValueError("model_version cannot be empty")
        if not self.lower <= self.premium <= self.upper:
            raise ValueError("prediction must satisfy lower <= premium <= upper")
        return self
