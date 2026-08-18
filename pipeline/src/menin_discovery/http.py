"""Resilient HTTP and atomic file-writing utilities for public data sources."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "menin-discovery/0.2 (+https://github.com/Shivansh-Sahni/Menin-Inhibitor-Discovery)"


def build_session(*, retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    """Return a session with bounded retries for transient API failures."""

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/csv, */*"})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_response(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int | float | tuple[int | float, int | float] = (10, 90),
    session: requests.Session | None = None,
) -> requests.Response:
    """GET a URL through the retrying session and require a successful response."""

    client = session or build_session()
    response = client.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write a text file atomically so interrupted downloads do not replace valid snapshots."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
