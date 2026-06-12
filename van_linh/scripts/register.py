from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.net import detect_lan_ip, resolve_teacher_host
from app.schemas import RegisterPayload
from scripts.common import build_headers, print_response


def resolve_server_url() -> str:
    settings = get_settings()
    if settings.server_public_ip:
        ip_address = settings.server_public_ip
    else:
        teacher_host = resolve_teacher_host(settings.teacher_base_url)
        ip_address = detect_lan_ip(teacher_host, 8000)

    return f"http://{ip_address}:{settings.server_port}"


def main() -> None:
    settings = get_settings()
    server_url = resolve_server_url()
    payload = RegisterPayload(server_url=server_url)
    endpoint = f"{settings.teacher_base_url}/competition/register"

    print(f"Resolved server_url: {server_url}")
    with httpx.Client(timeout=15) as client:
        response = client.post(
            endpoint,
            headers=build_headers(),
            json=payload.model_dump(),
        )
    print_response(response)


if __name__ == "__main__":
    main()
