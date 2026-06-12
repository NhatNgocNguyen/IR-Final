from __future__ import annotations

import socket
from urllib.parse import urlparse


def resolve_teacher_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.hostname:
        raise ValueError(f"Invalid TEACHER_BASE_URL: {base_url}")
    return parsed.hostname


def detect_lan_ip(target_host: str, target_port: int = 80) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target_host, target_port))
        ip_address = sock.getsockname()[0]
    finally:
        sock.close()

    if not ip_address or ip_address.startswith("127."):
        raise RuntimeError("Could not resolve a routable LAN IP address")
    return ip_address
