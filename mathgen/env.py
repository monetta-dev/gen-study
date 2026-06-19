"""Minimal .env.local loading for local configuration."""

from __future__ import annotations

import os
from pathlib import Path

_DOTENV_VALUES: dict[str, str] = {}


def load_dotenv(path: str | Path = ".env.local") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key not in os.environ or os.environ.get(key) == _DOTENV_VALUES.get(key):
            os.environ[key] = value
            _DOTENV_VALUES[key] = value
