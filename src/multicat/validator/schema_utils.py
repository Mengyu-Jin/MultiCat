from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"


@lru_cache(maxsize=None)
def load_schema(schema_filename: str) -> dict:
    schema_path = _SCHEMAS_DIR / schema_filename
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_with_schema(payload: dict, schema_filename: str, *, label: str) -> None:
    """Validate payload against a JSON schema file; raise ValueError on failure.

    label is used to name the agent/step in the error message (e.g. "Screening Agent").
    """
    schema = load_schema(schema_filename)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if not errors:
        return
    messages = [
        f"{'.'.join(str(part) for part in error.path) or '$'}: {error.message}" for error in errors
    ]
    raise ValueError(f"{label} result failed schema validation: " + "; ".join(messages))
