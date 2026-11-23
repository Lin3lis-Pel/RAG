"""Write the API's OpenAPI description to frontend/openapi.json, from which the page's TypeScript types are generated.

Run after changing backend/app/schemas.py or a route (repo root, backend venv active), then regenerate the types:

    python scripts/export_openapi.py
    cd frontend && npm run types

A backend test fails when openapi.json is out of date, and CI fails when the generated types are.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402

OUT = ROOT / "frontend" / "openapi.json"

if __name__ == "__main__":
    OUT.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
