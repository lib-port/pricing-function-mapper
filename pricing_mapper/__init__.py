from importlib.metadata import PackageNotFoundError, version

from pricing_mapper.active_mapper import ActiveQuoteMapper, RunStats
from pricing_mapper.config import MapperConfig
from pricing_mapper.domain import build_comp_car_domain
from pricing_mapper.engine import PricingEngine

try:
    __version__ = version("pricing-function-mapper")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "ActiveQuoteMapper",
    "MapperConfig",
    "PricingEngine",
    "RunStats",
    "__version__",
    "build_comp_car_domain",
]
