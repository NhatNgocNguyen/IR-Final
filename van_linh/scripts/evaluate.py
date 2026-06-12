from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from scripts.common import build_headers, print_response


def main() -> None:
    settings = get_settings()
    endpoint = f"{settings.teacher_base_url}/competition/evaluate"
    with httpx.Client(timeout=30) as client:
        response = client.post(endpoint, headers=build_headers())
    print_response(response)


if __name__ == "__main__":
    main()
