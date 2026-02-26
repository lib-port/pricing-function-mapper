from __future__ import annotations

from typing import Any

from pricing_mapper.engine import PricingEngine


def serve_api(engine_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except Exception as exc:
        raise RuntimeError(
            "FastAPI serving requires optional dependencies. "
            "Install with: pip install -e .[api]"
        ) from exc

    engine = PricingEngine.load(engine_path)
    app = FastAPI(title="Pricing Engine API", version="1.0.0")

    class PriceBatchRequest(BaseModel):
        rows: list[dict[str, Any]]

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
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"premium": round(float(premium), 2)}

    @app.post("/price-batch")
    def price_batch(req: PriceBatchRequest) -> dict[str, Any]:
        try:
            rows = engine.predict_rows_with_inputs(req.rows)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"rows": rows, "count": len(rows)}

    uvicorn.run(app, host=host, port=port)
