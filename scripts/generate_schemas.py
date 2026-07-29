"""Regenerate committed v1 JSON Schemas from public Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from pricing_mapper.config import config_json_schema
from pricing_mapper.domain import rows_json_schema


def main() -> None:
    target = Path("schema")
    target.mkdir(exist_ok=True)
    outputs = {
        "mapper-config.schema.json": config_json_schema(),
        "car-quote-input.schema.json": rows_json_schema(),
    }
    for name, schema in outputs.items():
        (target / name).write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
