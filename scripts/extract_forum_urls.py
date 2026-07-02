#!/usr/bin/env python3
"""Extract community.livekit.io topic URLs from a markdown export."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_FILE = ROOT / "data" / "livekit_community_export.md"
URLS_FILE = ROOT / "data" / "livekit_forum_urls.txt"

URL_PATTERN = re.compile(r"\((https://community\.livekit\.io/t/[^)]+)\)")


def extract_urls(text: str) -> list[str]:
    return sorted(set(URL_PATTERN.findall(text)))


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPORT_FILE
    if not path.exists():
        print(f"Missing export file: {path}", file=sys.stderr)
        return 1
    urls = extract_urls(path.read_text(encoding="utf-8"))
    URLS_FILE.parent.mkdir(exist_ok=True)
    URLS_FILE.write_text("\n".join(urls) + "\n", encoding="utf-8")
    print(f"Extracted {len(urls)} URLs -> {URLS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
