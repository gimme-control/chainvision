# client.py
"""
Minimal client for the KV Append Store.
Deps: requests  ->  pip install requests
"""

from typing import Any, Dict, Optional
import os
import requests

DEFAULT_BASE = os.environ.get("KV_BASE", "http://192.168.1.50:8000")
TIMEOUT = 5  # seconds


def _base(base_url: Optional[str]) -> str:
    return base_url or DEFAULT_BASE


def submit(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST /append  with a JSON dict."""
    r = requests.post(f"{_base(base_url)}/append", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch(base_url: str) -> Dict[str, Any]:
    """GET /dump  returns full snapshot as dict."""
    r = requests.get(f"{_base(base_url)}/dump", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def clear(base_url: Optional[str] = None) -> Dict[str, Any]:
    """POST /clear  wipe memory + SQLite. If base_url omitted, use DEFAULT_BASE."""
    r = requests.post(f"{_base(base_url)}/clear", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()