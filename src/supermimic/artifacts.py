"""Strict, atomic JSON artifacts for configs, metrics and evaluation reports."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _encode(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def json_text(value, *, indent: int | None = None) -> str:
    return json.dumps(value, default=_encode, allow_nan=False, indent=indent)


def write_json(path: str | Path, value) -> None:
    destination = Path(path)
    payload = json_text(value, indent=2) + "\n"  # Validate before changing any file.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
