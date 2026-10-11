"""utils.py — Shared utilities: config loading, logging, date helpers."""

from __future__ import annotations

import logging
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# .env loader
# ─────────────────────────────────────────────────────────────────────────────

def load_env(env_path: str | Path | None = None) -> None:
    """
    Load a .env file into os.environ using python-dotenv.

    Search order (first found wins):
      1. Explicit env_path argument
      2. VolatilityStrategy/.env  (project root)
      3. Parent directories walking up (e.g. Quant/.env)
      4. Sibling subdirectories of the workspace root (e.g. Quant/Alpha/.env)
    """
    from dotenv import load_dotenv

    if env_path:
        load_dotenv(Path(env_path), override=False)
        return

    project = Path(__file__).parent  # VolatilityStrategy/

    # Walk up the tree from project root
    for candidate in [project, *project.parents]:
        env_file = candidate / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            return
        # Also check immediate subdirectories at each level
        try:
            for sibling in sorted(candidate.iterdir()):
                if sibling.is_dir() and sibling != project:
                    sib_env = sibling / ".env"
                    if sib_env.exists():
                        load_dotenv(sib_env, override=False)
                        return
        except PermissionError:
            pass
        # Stop after checking one level above the workspace root
        if candidate.parent == candidate:
            break


def get_kite_credentials() -> tuple[str, str, str | None]:
    """
    Return (api_key, access_token, api_secret) from environment variables.

    Expected .env keys:
        KITE_API_KEY      — your Zerodha app API key
        KITE_ACCESS_TOKEN — generated daily via --auth flow
        KITE_API_SECRET   — your Zerodha app API secret (needed for --auth only)
    """
    load_env()
    api_key = os.environ.get("KITE_API_KEY", "")
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
    api_secret = os.environ.get("KITE_API_SECRET")

    if not api_key:
        raise EnvironmentError(
            "KITE_API_KEY not set. Add it to your .env file:\n"
            "  KITE_API_KEY=your_key_here"
        )
    return api_key, access_token, api_secret


def save_access_token_to_env(token: str, env_path: str | Path | None = None) -> Path:
    """Write or update KITE_ACCESS_TOKEN in the .env file."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
        if not env_path.exists():
            # Try to find the existing .env file
            from dotenv import find_dotenv
            found = find_dotenv(usecwd=False)
            if found:
                env_path = Path(found)

    path = Path(env_path)
    lines: list[str] = []
    replaced = False

    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("KITE_ACCESS_TOKEN="):
                lines.append(f"KITE_ACCESS_TOKEN={token}")
                replaced = True
            else:
                lines.append(line)

    if not replaced:
        lines.append(f"KITE_ACCESS_TOKEN={token}")

    path.write_text("\n".join(lines) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, defaulting to config/config.yaml relative to this file."""
    if path is None:
        path = Path(__file__).parent / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_file_logger(name: str, log_path: str | Path, level: int = logging.DEBUG) -> logging.Logger:
    logger = get_logger(name, level)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_date(value: str | date | datetime | pd.Timestamp | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def date_range_months(start: date, end: date) -> list[tuple[date, date]]:
    """Return list of (month_start, month_end) covering [start, end]."""
    ranges = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = min(next_month - timedelta(days=1), end)
        ranges.append((max(cur, start), month_end))
        cur = next_month
    return ranges


def nse_expiry_month_year_pairs(start: date, end: date) -> list[tuple[int, int]]:
    """Return unique (year, month) pairs for NSE futures contracts spanning [start, end]."""
    pairs = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        pairs.append((cur.year, cur.month))
        next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        cur = next_month
    return pairs


NSE_MONTH_CODES = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
    5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def nifty_futures_symbol(year: int, month: int) -> str:
    """E.g. (2025, 7) → 'NIFTY25JULFUT'"""
    yy = str(year)[-2:]
    mon = NSE_MONTH_CODES[month]
    return f"NIFTY{yy}{mon}FUT"


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    return Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────────────────────────────────────

def annualized_rv(returns: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    """Rolling realized volatility, annualized."""
    return returns.rolling(window).std() * np.sqrt(trading_days)
