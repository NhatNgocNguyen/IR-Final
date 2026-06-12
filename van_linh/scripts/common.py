from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from app.config import get_settings


def build_headers() -> dict[str, str]:
    settings = get_settings()
    return {"X-Student-ID": settings.student_id}


def print_response(response: httpx.Response) -> None:
    print(f"HTTP {response.status_code}")
    try:
        data: Any = response.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)
