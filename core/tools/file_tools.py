"""Shared file and data tools for agents."""

from __future__ import annotations

import json
import csv
import os
from pathlib import Path
from typing import Any


def read_json(path: str) -> dict:
    """Read a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    """Write data to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_csv(path: str) -> list[dict]:
    """Read a CSV file into a list of dicts."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write a list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
