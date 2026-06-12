from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.schemas import EvaluateRequest
from scripts.common import build_headers, print_response


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("document_received must be true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document-received",
        type=_parse_bool,
        required=True,
        help="Set false on the first evaluation attempt to receive the document upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    endpoint = f"{settings.teacher_base_url}/competition/evaluate"
    payload = EvaluateRequest(document_received=args.document_received)
    with httpx.Client(timeout=settings.evaluate_timeout_seconds) as client:
        response = client.post(
            endpoint,
            headers=build_headers(),
            json=payload.model_dump(),
        )
    print_response(response)


if __name__ == "__main__":
    main()
