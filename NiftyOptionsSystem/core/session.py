"""
core/session.py
----------------
Kite Connect authentication.

Kite access tokens expire every day (around midnight IST), so a fresh
request_token -> access_token exchange is required each trading day. This
module provides:

  - get_kite_client()          : build an authenticated KiteConnect instance
                                  from env vars, for use by data modules.
  - login_url(api_key)         : the URL to open in a browser to log in.
  - generate_access_token(...) : exchange a request_token for an access_token.
  - persist_access_token(...)  : rewrite KITE_ACCESS_TOKEN in the repo-root
                                  .env file in place, without touching other
                                  lines.

The actual semi-automated login flow (open browser, prompt for the pasted
redirect URL/request_token, exchange, persist) lives in
scripts/generate_access_token.py — this module only holds the reusable
pieces so data/signal code can import it without pulling in CLI concerns.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from kiteconnect import KiteConnect

from config import ENV_PATH

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(ENV_PATH)
        _ENV_LOADED = True


class KiteAuthError(RuntimeError):
    """Raised when Kite credentials are missing or the access token is stale/invalid."""


def login_url(api_key: str | None = None) -> str:
    api_key = api_key or _require_env("KITE_API_KEY")
    return KiteConnect(api_key=api_key).login_url()


def generate_access_token(request_token: str, api_key: str | None = None,
                           api_secret: str | None = None) -> str:
    """Exchange a request_token (from the login redirect) for an access_token."""
    api_key = api_key or _require_env("KITE_API_KEY")
    api_secret = api_secret or _require_env("KITE_API_SECRET")
    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    return session_data["access_token"]


def persist_access_token(access_token: str, env_path: Path = ENV_PATH) -> None:
    """
    Rewrite KITE_ACCESS_TOKEN=... in the .env file in place. Adds the line if
    it doesn't exist yet. Leaves every other line untouched.
    """
    line_re = re.compile(r"^KITE_ACCESS_TOKEN\s*=.*$")
    new_line = f"KITE_ACCESS_TOKEN={access_token}"

    if env_path.exists():
        lines = env_path.read_text().splitlines()
    else:
        lines = []

    for i, line in enumerate(lines):
        if line_re.match(line):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)

    env_path.write_text("\n".join(lines) + "\n")
    os.environ["KITE_ACCESS_TOKEN"] = access_token


def _require_env(key: str) -> str:
    _ensure_env_loaded()
    val = os.environ.get(key)
    if not val:
        raise KiteAuthError(
            f"{key} is not set. Add it to {ENV_PATH} or export it in your shell."
        )
    return val


def get_kite_client() -> KiteConnect:
    """
    Build an authenticated KiteConnect client from KITE_API_KEY /
    KITE_ACCESS_TOKEN env vars. Does NOT validate the token against the API
    (that costs a request) — the first real call will raise
    kiteconnect.exceptions.TokenException if it's stale, which callers should
    treat as "re-run scripts/generate_access_token.py", not retry.
    """
    api_key = _require_env("KITE_API_KEY")
    access_token = _require_env("KITE_ACCESS_TOKEN")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite
