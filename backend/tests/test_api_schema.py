"""The committed OpenAPI file is the API's current one, so the TypeScript types generated from it are too."""

import json
from pathlib import Path

from app.main import app

OPENAPI = Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"


def test_the_committed_openapi_file_is_current():
    committed = json.loads(OPENAPI.read_text(encoding="utf-8"))

    assert committed == app.openapi(), "run: python scripts/export_openapi.py, then (in frontend/) npm run types"


def test_response_fields_with_a_default_are_required_in_the_schema():
    """So the generated types do not make the page check for fields that are always there."""
    source = app.openapi()["components"]["schemas"]["Source"]

    assert set(source["required"]) == set(source["properties"])
