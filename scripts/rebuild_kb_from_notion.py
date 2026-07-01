#!/usr/bin/env python3
"""Batch-ingest Notion-sourced docs into the Railway RAG server."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

DEFAULT_BASE = "https://livekit-rag-production.up.railway.app"
LIVEKIT_INDEX = (
    "https://cautious-cirrus-710.notion.site/"
    "LiveKit-Docs-Index-352bd1f88e0e81fcb677e34c27f1e112"
)
VAPI_BOOK = (
    "https://cautious-cirrus-710.notion.site/"
    "Vapi-Book-352bd1f88e0e8024a439f7e9350154e6"
)
COMMUNITY_KB = (
    "https://cautious-cirrus-710.notion.site/"
    "0ba419733e2848ccba59317c5f05dcc9?v=cb4a39345e6d49869c4e13ebe6b7c898"
)

ROOT = Path(__file__).resolve().parents[1]
URLS_FILE = ROOT / "data" / "livekit_doc_urls.txt"


def fetch_livekit_urls() -> list[str]:
    if URLS_FILE.exists():
        return [line.strip() for line in URLS_FILE.read_text().splitlines() if line.strip()]
    text = requests.get(f"https://r.jina.ai/{LIVEKIT_INDEX}", timeout=120).text
    urls = sorted(set(re.findall(r"\((https://docs\.livekit\.io[^)]+)\)", text)))
    URLS_FILE.parent.mkdir(exist_ok=True)
    URLS_FILE.write_text("\n".join(urls) + "\n")
    return urls


def fetch_vapi_notion_urls() -> list[str]:
    text = requests.get(f"https://r.jina.ai/{VAPI_BOOK}", timeout=60).text
    urls = re.findall(r"\((https://cautious-cirrus-710\.notion\.site/[^)#]+)", text)
    cleaned = sorted({re.sub(r"\?pvs=\d+", "", u) for u in urls if "Part-" in u})
    return cleaned


def source(url: str, source_type: str, category: str, architecture: str = "both") -> dict:
    return {
        "url": url,
        "source_type": source_type,
        "category": category,
        "architecture": architecture,
        "fetch_mode": "jina",
    }


def post_ingest(base: str, sources: list[dict], timeout: int = 600) -> dict:
    resp = requests.post(
        f"{base.rstrip('/')}/ingest",
        json={"sources": sources},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def batched(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild KB from Notion sources")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--livekit-docs", action="store_true")
    parser.add_argument("--vapi-book", action="store_true")
    parser.add_argument("--community-kb", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.livekit_docs = args.vapi_book = args.community_kb = True

    if not (args.livekit_docs or args.vapi_book or args.community_kb):
        parser.error("Pick at least one of --livekit-docs, --vapi-book, --community-kb, or --all")

    sources: list[dict] = []

    if args.vapi_book or args.community_kb:
        sources.extend(
            [
                source(VAPI_BOOK, "vapi_book", "prompting"),
                *[
                    source(u, "vapi_book", "prompting")
                    for u in fetch_vapi_notion_urls()
                ],
            ]
        )
    if args.community_kb:
        sources.append(source(COMMUNITY_KB, "forum", "infra"))
    if args.livekit_docs:
        sources.extend(
            source(u, "docs", "infra") for u in fetch_livekit_urls()
        )

    total_ingested = 0
    total_batches = (len(sources) + args.batch_size - 1) // args.batch_size
    print(f"Ingesting {len(sources)} sources in {total_batches} batches...")

    for i, batch in enumerate(batched(sources, args.batch_size), start=1):
        print(f"[{i}/{total_batches}] {batch[0]['url'][:70]}... ({len(batch)} urls)", flush=True)
        try:
            result = post_ingest(args.base, batch)
            count = result.get("ingested_count", 0)
            total_ingested += count
            print(f"  -> +{count} chunks", result.get("errors"), flush=True)
        except Exception as exc:
            print(f"  !! batch failed: {exc}", file=sys.stderr, flush=True)
        if i < total_batches:
            time.sleep(args.sleep)

    print(f"Done. Total chunks ingested this run: {total_ingested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
