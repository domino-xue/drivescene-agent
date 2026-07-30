from __future__ import annotations

import re
from pathlib import Path


SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "data",
    "outputs",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def test_repository_has_no_plaintext_api_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    leaked_locations: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET_PATTERN.search(text):
            leaked_locations.append(str(path.relative_to(root)))

    assert leaked_locations == []
