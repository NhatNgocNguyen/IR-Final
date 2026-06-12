"""
register.py — CLI helper to interact with the Teacher Server.

Usage:
  python register.py register               # đăng ký server URL
  python register.py evaluate               # bắt đầu thi (lần đầu → upload + 100 câu)
  python register.py evaluate --skip-upload # bỏ qua upload (VectorDB đã có)
  python register.py result                 # xem điểm hiện tại
  python register.py reset                  # reset điểm (dùng lại 1 trong 5 lần)
  python register.py start                  # register + evaluate (tiện lợi)
  python register.py start --skip-upload    # register + evaluate bỏ upload
"""

import argparse
import sys

import requests

from config import MY_SERVER_URL, STUDENT_ID, TEACHER_BASE_URL

BASE = TEACHER_BASE_URL
HEADERS = {"X-Student-ID": STUDENT_ID}


def _print(label: str, data: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for k, v in data.items():
        print(f"  {k}: {v}")
    print()


# ------------------------------------------------------------------

def register() -> dict:
    """POST /competition/register"""
    url  = f"{BASE}/competition/register"
    body = {"server_url": MY_SERVER_URL}
    print(f"Registering  {STUDENT_ID}  @  {MY_SERVER_URL}")
    r = requests.post(url, json=body, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    _print("REGISTER", data)
    return data


def evaluate(document_received: bool = False) -> dict:
    """
    POST /competition/evaluate
    document_received=False  → teacher sends /upload then 100x /ask
    document_received=True   → teacher sends 100x /ask only (faster re-submit)
    """
    url  = f"{BASE}/competition/evaluate"
    body = {"document_received": document_received}
    flag = "SKIP UPLOAD" if document_received else "FULL (upload + questions)"
    print(f"Calling /evaluate  [{flag}]")
    print("  ⚠  This will block until all 100 questions are answered…")
    r = requests.post(url, json=body, headers=HEADERS, timeout=600)
    r.raise_for_status()
    data = r.json()
    _print("EVALUATE RESULT", data)
    return data


def result() -> dict:
    """GET /competition/result"""
    url = f"{BASE}/competition/result"
    r   = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    _print("CURRENT RESULT", data)
    return data


def reset() -> dict:
    """POST /competition/reset"""
    url = f"{BASE}/competition/reset"
    print("Resetting exam state (uses 1 of 5 submission attempts)…")
    r   = requests.post(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    _print("RESET", data)
    return data


# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PTIT RAG Exam — Teacher Server CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["register", "evaluate", "result", "reset", "start"],
        help="Action to perform",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        default=False,
        help="Pass document_received=True to skip /upload (VectorDB already built)",
    )
    args = parser.parse_args()

    try:
        if args.command == "register":
            register()

        elif args.command == "evaluate":
            evaluate(document_received=args.skip_upload)

        elif args.command == "result":
            result()

        elif args.command == "reset":
            reset()

        elif args.command == "start":
            register()
            evaluate(document_received=args.skip_upload)

    except requests.HTTPError as exc:
        print(f"\n❌  HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.ConnectionError:
        print("\n❌  Cannot connect to Teacher Server. Check your IP / LAN.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
