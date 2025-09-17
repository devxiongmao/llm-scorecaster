"""Script to generate OpenAPI schema and save it to docs/openapi.json."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import app  # pylint: disable=wrong-import-position


def main():
    """Generate OpenAPI schema and save to docs/openapi.json."""
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    openapi_schema = app.openapi()

    openapi_file = docs_dir / "openapi.json"
    with open(openapi_file, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"OpenAPI schema generated successfully at {openapi_file}")


if __name__ == "__main__":
    main()
