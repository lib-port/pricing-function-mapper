from __future__ import annotations

from typing import Any

from pricing_mapper import __version__
from pricing_mapper.engine import PricingEngine


def create_app(engine: PricingEngine) -> Any:
    """Create the optional FastAPI application for a loaded engine."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI serving requires optional dependencies. " "Install with: pip install -e .[api]"
        ) from exc

    app = FastAPI(title="Pricing Engine API", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        return engine.model_info()

    @app.post("/price")
    def price(row: dict[str, Any]) -> dict[str, Any]:
        try:
            premium = engine.predict_row(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"premium": round(float(premium), 2)}

    @app.post("/price-batch")
    def price_batch(req: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        try:
            request_rows = req["rows"]
            rows = engine.predict_rows_with_inputs(request_rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"rows": rows, "count": len(rows)}

    return app


def serve_api(engine_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Load an engine and serve its FastAPI application with Uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI serving requires optional dependencies. " "Install with: pip install -e .[api]"
        ) from exc

    engine = PricingEngine.load(engine_path)
    app = create_app(engine)
    uvicorn.run(app, host=host, port=port)
