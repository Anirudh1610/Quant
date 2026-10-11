#!/usr/bin/env python3
"""
scripts/generate_access_token.py
---------------------------------
Run this once each trading morning before pulling data or trading.

Opens the Kite login page in your browser. After you log in, Kite redirects
to your app's registered redirect URL with a `request_token` query param —
paste either the full redirected URL or just the token back here. This
script exchanges it for a fresh access_token and writes it into the repo
root .env (KITE_ACCESS_TOKEN=...), leaving every other line untouched.

Usage:
    python scripts/generate_access_token.py
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ENV_PATH
from core.session import KiteAuthError, generate_access_token, login_url, persist_access_token


def _extract_request_token(pasted: str) -> str:
    pasted = pasted.strip()
    if pasted.startswith("http"):
        qs = parse_qs(urlparse(pasted).query)
        token = qs.get("request_token", [None])[0]
        if not token:
            raise ValueError("No request_token found in that URL.")
        return token
    return pasted


def main() -> None:
    try:
        url = login_url()
    except KiteAuthError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\nOpening Kite login page:\n  {url}\n")
    webbrowser.open(url)

    pasted = input(
        "After logging in, paste the redirected URL (or just the "
        "request_token param):\n> "
    )
    try:
        request_token = _extract_request_token(pasted)
        access_token = generate_access_token(request_token)
    except Exception as e:
        print(f"[ERROR] Failed to generate access token: {e}")
        sys.exit(1)

    persist_access_token(access_token)
    print(f"\nAccess token saved to {ENV_PATH}. Valid until ~midnight IST today.")


if __name__ == "__main__":
    main()
