#!/usr/bin/env python3
"""
MCP server exposing LiveKit RAG search as a tool for Claude.
Deployed as an SSE server for remote access.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import cohere
import requests
import voyageai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
)
from starlette.requests import Request
from starlette.responses import JSONResponse

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = str(SCRIPT_DIR / "qdrant_storage")
COLLECTION_NAME = "livekit_kb"
VECTOR_SIZE = 512
PORT = int(os.getenv("PORT", "8080"))

EMBED_MODEL = "voyage-3-lite"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

VALID_CATEGORIES = ("prompting", "infra")
VALID_ARCHITECTURES = ("cascaded", "s2s", "both")

PROMPTING_SOURCES: list[dict[str, str]] = [
    {
        "url": "https://docs.livekit.io/agents/start/prompting.md",
        "source_type": "livekit_prompting",
        "category": "prompting",
        "architecture": "both",
        "fetch_mode": "markdown",
    },
    {
        "url": "https://developers.openai.com/cookbook/examples/realtime_prompting_guide",
        "source_type": "openai_realtime",
        "category": "prompting",
        "architecture": "s2s",
        "fetch_mode": "jina",
    },
    {
        "url": "https://docs.vapi.ai/prompting-guide.md",
        "source_type": "vapi_prompting",
        "category": "prompting",
        "architecture": "both",
        "fetch_mode": "markdown",
        "fallback_url": "https://docs.vapi.ai/prompting-guide",
    },
    {
        "url": "https://hamming.ai/resources/voice-agent-caller-identity-testing-checklist",
        "source_type": "hamming_identity",
        "category": "prompting",
        "architecture": "both",
        "fetch_mode": "jina",
    },
    {
        "url": "https://www.anthropic.com/engineering/writing-tools-for-agents",
        "source_type": "anthropic_tools",
        "category": "prompting",
        "architecture": "both",
        "fetch_mode": "jina",
    },
]

mcp = FastMCP("livekit-rag", host="0.0.0.0", port=PORT)

_qdrant_client = None
_voyage_client = None
_cohere_client = None

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def get_voyage_client() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY environment variable not set")
        _voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _voyage_client


def get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable not set")
        _cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    return _cohere_client


def ensure_payload_indexes(qdrant: QdrantClient) -> None:
    """Create keyword indexes for filterable payload fields (safe, idempotent)."""
    for field in ("source_type", "category", "architecture"):
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" not in msg and "already indexed" not in msg:
                print(f"[qdrant] payload index '{field}': {exc}", file=sys.stderr)


def ensure_collection(qdrant: QdrantClient) -> None:
    """Create the collection if missing. Never delete or recreate existing data."""
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        current_size = info.config.params.vectors.size
        if current_size != VECTOR_SIZE:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' has vector size {current_size}, "
                f"expected {VECTOR_SIZE}. Fix manually; auto-recreate is disabled."
            )
    except ValueError:
        raise
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"},
        )
    ensure_payload_indexes(qdrant)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def stable_point_id(url: str, chunk_index: int) -> int:
    digest = hashlib.sha256(f"{url}:{chunk_index}".encode()).hexdigest()
    return int(digest[:15], 16) & 0x7FFFFFFF


def title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def fetch_markdown(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    response.raise_for_status()
    text = response.text
    fallback = urlparse(url).path.rsplit("/", 1)[-1].replace(".md", "") or urlparse(url).netloc
    return text, title_from_markdown(text, fallback)


def fetch_jina(url: str) -> tuple[str, str]:
    response = requests.get(
        f"https://r.jina.ai/{url}",
        headers={**HTTP_HEADERS, "Accept": "text/plain"},
        timeout=60,
    )
    response.raise_for_status()
    text = response.text
    title = urlparse(url).path.rsplit("/", 1)[-1].replace("-", " ") or urlparse(url).netloc
    lines = text.splitlines()
    if lines and lines[0].lower().startswith("title:"):
        title = lines[0].split(":", 1)[1].strip()
    return text, str(title)


def fetch_html(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = str(soup.title.string) if soup.title and soup.title.string else urlparse(url).netloc
    return soup.get_text(separator="\n", strip=True), title


def fetch_url_content(
    url: str,
    fetch_mode: str = "auto",
    fallback_url: Optional[str] = None,
) -> tuple[str, str]:
    modes = [fetch_mode] if fetch_mode != "auto" else []
    if fetch_mode == "auto":
        if url.endswith(".md"):
            modes = ["markdown", "jina", "html"]
        else:
            modes = ["jina", "html"]

    urls_to_try = [url]
    if fallback_url:
        urls_to_try.append(fallback_url)

    last_error: Optional[Exception] = None
    for target_url in urls_to_try:
        target_modes = modes if target_url == url else ["jina", "markdown", "html"]
        for mode in target_modes:
            try:
                if mode == "markdown":
                    return fetch_markdown(target_url)
                if mode == "jina":
                    return fetch_jina(target_url)
                if mode == "html":
                    return fetch_html(target_url)
            except Exception as exc:
                last_error = exc
                print(f"[ingest] {mode} failed for {target_url}: {exc}", file=sys.stderr)

    raise last_error or RuntimeError(f"Failed to fetch {url}")


def build_search_filter(
    source_type: Optional[str] = None,
    category: Optional[str] = None,
    architecture: Optional[str] = None,
) -> Optional[Filter]:
    conditions = []
    if source_type:
        conditions.append(
            FieldCondition(key="source_type", match=MatchValue(value=source_type))
        )
    if category:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )
    if architecture:
        if architecture in ("s2s", "cascaded"):
            match_values = [architecture, "both"]
        else:
            match_values = [architecture]
        conditions.append(
            FieldCondition(key="architecture", match=MatchAny(any=match_values))
        )
    if not conditions:
        return None
    return Filter(must=conditions)


def vector_search(qdrant: QdrantClient, query_embedding: list[float], search_filter: Optional[Filter]):
    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=30,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False,
        )
        return response.points
    return qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=30,
        query_filter=search_filter,
        with_payload=True,
    )


def format_search_hit(payload: dict, score: float) -> dict:
    return {
        "content": payload.get("content", ""),
        "source_url": payload.get("source_url", ""),
        "source_type": payload.get("source_type", ""),
        "title": payload.get("title", ""),
        "section": payload.get("section", ""),
        "category": payload.get("category", ""),
        "architecture": payload.get("architecture", ""),
        "score": float(score),
    }


def ingest_source(qdrant: QdrantClient, voyage: voyageai.Client, source: dict[str, Any]) -> int:
    url = source["url"]
    source_type = source.get("source_type", "docs")
    category = source.get("category")
    architecture = source.get("architecture")
    fetch_mode = source.get("fetch_mode", "auto")
    fallback_url = source.get("fallback_url")

    print(f"[ingest] Fetching {url} ({fetch_mode})", flush=True)
    text, title = fetch_url_content(url, fetch_mode=fetch_mode, fallback_url=fallback_url)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    print(f"[ingest] Embedding {len(chunks)} chunks from {url}", flush=True)
    embeddings = voyage.embed(
        chunks,
        model=EMBED_MODEL,
        input_type="document",
    ).embeddings

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        payload = {
            "content": chunk,
            "source_url": url,
            "source_type": source_type,
            "title": str(title),
            "section": f"chunk_{i}",
        }
        if category:
            payload["category"] = category
        if architecture:
            payload["architecture"] = architecture

        points.append(
            PointStruct(
                id=stable_point_id(url, i),
                vector=embedding,
                payload=payload,
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[ingest] ✓ Ingested {len(chunks)} chunks from {url}", flush=True)
    return len(chunks)


def normalize_sources(body: dict) -> list[dict[str, Any]]:
    if body.get("sources"):
        return body["sources"]

    urls = body.get("urls", [])
    defaults = {
        "source_type": body.get("source_type", "docs"),
        "category": body.get("category"),
        "architecture": body.get("architecture"),
        "fetch_mode": body.get("fetch_mode", "auto"),
    }
    return [{"url": url, **defaults} for url in urls]


@mcp.tool()
def search_livekit_kb(
    query: str,
    top_k: int = 5,
    source_type: Optional[str] = None,
    category: Optional[str] = None,
    architecture: Optional[str] = None,
) -> list[dict]:
    """Search the voice-agent knowledge base (LiveKit docs, forum, prompting guides).

    Finds relevant chunks for building and prompting voice agents. Use filters to
    narrow results by content type or pipeline architecture.

    Args:
        query: Natural language search query (e.g., "How should I write the system prompt?")
        top_k: Number of results to return (1-20, default 5)
        source_type: Optional filter on source (e.g. docs, forum, vapi_book,
            livekit_prompting, openai_realtime, vapi_prompting, hamming_identity,
            anthropic_tools). Omit to search all sources.
        category: Optional filter — "prompting" for prompt-engineering content,
            "infra" for deployment/platform docs. Omit to include untagged legacy chunks.
        architecture: Optional filter — "cascaded" (STT→LLM→TTS), "s2s"
            (speech-to-speech / realtime), or "both". Chunks tagged "both" match
            cascaded and s2s filters.

    Returns:
        List of dicts with keys: content, source_url, source_type, title, section,
        category, architecture, score
    """

    def _err(msg: str) -> list[dict]:
        return [
            {
                "content": f"[ERROR] {msg}",
                "source_url": "",
                "source_type": "error",
                "title": "Search error",
                "section": "",
                "category": "",
                "architecture": "",
                "score": 0.0,
            }
        ]

    try:
        if not query or not query.strip():
            return []

        top_k = max(1, min(20, top_k))

        if category and category not in VALID_CATEGORIES:
            return _err(
                f"Invalid category '{category}'. Use 'prompting', 'infra', or omit."
            )
        if architecture and architecture not in VALID_ARCHITECTURES:
            return _err(
                f"Invalid architecture '{architecture}'. Use 'cascaded', 's2s', 'both', or omit."
            )

        qdrant = get_qdrant_client()
        voyage = get_voyage_client()
        cohere_client = get_cohere_client()

        try:
            qdrant.get_collection(COLLECTION_NAME)
        except Exception as e:
            return _err(f"Collection '{COLLECTION_NAME}' not found. Run ingest first. ({e})")

        ensure_payload_indexes(qdrant)

        query_embedding = voyage.embed(
            [query],
            model=EMBED_MODEL,
            input_type="query",
        ).embeddings[0]

        search_filter = build_search_filter(source_type, category, architecture)
        search_results = vector_search(qdrant, query_embedding, search_filter)

        if not search_results:
            return []

        docs_to_rank = [r.payload.get("content", "") for r in search_results]
        try:
            rerank_response = cohere_client.rerank(
                model="rerank-3.5",
                query=query,
                documents=docs_to_rank,
                top_n=top_k,
            )
            ranked_indices = [result.index for result in rerank_response.results]
            rerank_scores = {
                result.index: result.relevance_score for result in rerank_response.results
            }
            ranked_results = [
                (search_results[i], rerank_scores.get(i, 0.0)) for i in ranked_indices
            ]
        except Exception as e:
            print(f"[mcp] rerank failed, falling back to vector scores: {e}", file=sys.stderr)
            ranked_results = [(p, float(p.score)) for p in search_results[:top_k]]

        return [
            format_search_hit(point.payload or {}, score)
            for point, score in ranked_results
        ]

    except Exception as e:
        return _err(f"Search failed: {type(e).__name__}: {e}")


@mcp.tool()
def livekit_kb_stats() -> dict:
    """Get statistics about the LiveKit knowledge base index."""
    try:
        qdrant = get_qdrant_client()
        try:
            info = qdrant.get_collection(COLLECTION_NAME)
            return {
                "collection": COLLECTION_NAME,
                "points_count": info.points_count,
                "status": "ready",
                "vector_size": VECTOR_SIZE,
            }
        except Exception:
            return {
                "collection": COLLECTION_NAME,
                "status": "not_found",
                "message": "Index not created. Run ingest first.",
            }
    except Exception as e:
        return {"error": str(e)}


@mcp.custom_route("/ingest", methods=["POST"])
async def ingest_urls(request: Request) -> JSONResponse:
    """Ingest documents from URLs into the knowledge base.

    Legacy body:
    {
        "urls": ["https://example.com/doc"],
        "source_type": "docs",
        "category": "infra",
        "architecture": "cascaded"
    }

    Extended body (per-source metadata):
    {
        "sources": [
            {
                "url": "https://example.com/doc.md",
                "source_type": "livekit_prompting",
                "category": "prompting",
                "architecture": "both",
                "fetch_mode": "markdown"
            }
        ]
    }
    """
    try:
        body = await request.json()
        sources = normalize_sources(body)

        if not sources:
            return JSONResponse({"error": "No URLs provided"})

        qdrant = get_qdrant_client()
        voyage = get_voyage_client()
        ensure_collection(qdrant)

        ingested_count = 0
        errors = []

        for source in sources:
            url = source.get("url")
            if not url:
                errors.append("Source missing url field")
                continue
            try:
                ingested_count += ingest_source(qdrant, voyage, source)
            except Exception as e:
                error_msg = f"Failed to ingest {url}: {e}"
                print(f"[ingest] ✗ {error_msg}", file=sys.stderr, flush=True)
                errors.append(error_msg)

        return JSONResponse(
            {
                "status": "success",
                "ingested_count": ingested_count,
                "urls_processed": len(sources),
                "errors": errors if errors else None,
            }
        )

    except Exception as e:
        return JSONResponse({"error": f"Ingest failed: {str(e)}"})


@mcp.custom_route("/ingest/prompting", methods=["POST"])
async def ingest_prompting_sources(_request: Request) -> JSONResponse:
    """Ingest the 5 curated prompting sources (category=prompting)."""
    try:
        qdrant = get_qdrant_client()
        voyage = get_voyage_client()
        ensure_collection(qdrant)

        ingested_count = 0
        errors = []

        for source in PROMPTING_SOURCES:
            try:
                ingested_count += ingest_source(qdrant, voyage, source)
            except Exception as e:
                error_msg = f"Failed to ingest {source['url']}: {e}"
                print(f"[ingest] ✗ {error_msg}", file=sys.stderr, flush=True)
                errors.append(error_msg)

        return JSONResponse(
            {
                "status": "success",
                "ingested_count": ingested_count,
                "urls_processed": len(PROMPTING_SOURCES),
                "errors": errors if errors else None,
            }
        )

    except Exception as e:
        return JSONResponse({"error": f"Ingest failed: {str(e)}"})


if __name__ == "__main__":
    print(f"[mcp] Starting server on 0.0.0.0:{PORT}", flush=True)
    if QDRANT_URL:
        print(f"[mcp] Qdrant: remote ({QDRANT_URL})", flush=True)
    else:
        print(f"[mcp] Qdrant: local ({QDRANT_PATH})", flush=True)
    print(f"[mcp] VOYAGE_API_KEY set: {bool(VOYAGE_API_KEY)}", flush=True)
    print(f"[mcp] COHERE_API_KEY set: {bool(COHERE_API_KEY)}", flush=True)
    mcp.run(transport="streamable-http")
