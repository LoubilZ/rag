#!/usr/bin/env python3
"""
MCP server exposing LiveKit RAG search as a tool for Claude.
Deployed as an SSE server for remote access.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

import cohere
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

# Resolve paths relative to this script's location
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

# Initialize MCP server
mcp = FastMCP("livekit-rag", host="0.0.0.0", port=PORT)

# Global clients
_qdrant_client = None
_voyage_client = None
_cohere_client = None


def get_qdrant_client():
    """Lazy-load Qdrant client (remote if QDRANT_URL is set, else local disk)."""
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def get_voyage_client():
    """Lazy-load Voyage client."""
    global _voyage_client
    if _voyage_client is None:
        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY environment variable not set")
        _voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _voyage_client


def get_cohere_client():
    """Lazy-load Cohere client."""
    global _cohere_client
    if _cohere_client is None:
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable not set")
        _cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    return _cohere_client


def ensure_collection(qdrant: QdrantClient) -> None:
    """Create or recreate the collection with the expected vector size."""
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        current_size = info.config.params.vectors.size
        if current_size != VECTOR_SIZE:
            print(
                f"[ingest] Recreating collection (was {current_size}d, need {VECTOR_SIZE}d)",
                flush=True,
            )
            qdrant.delete_collection(COLLECTION_NAME)
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"},
            )
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"},
        )


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def fetch_and_parse_url(url: str) -> tuple[str, str]:
    """Fetch URL and extract text content."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get title
        title = str(soup.title.string) if soup.title and soup.title.string else urlparse(url).netloc
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        return text, title
    except Exception as e:
        print(f"[ingest] Error fetching {url}: {e}", file=sys.stderr)
        raise


@mcp.tool()
def search_livekit_kb(
    query: str,
    top_k: int = 5,
    source_type: Optional[str] = None
) -> list[dict]:
    """Search LiveKit documentation and community forum for information.

    Finds the most relevant chunks from official documentation and community
    forum threads to answer questions about building voice agents with LiveKit.

    Args:
        query: Natural language search query (e.g., "How do I detect turns?")
        top_k: Number of results to return (1-20, default 5)
        source_type: Optional filter — 'docs' for official docs, 'forum' for community threads, 'vapi_book' for Vapi Playbook, None for all

    Returns:
        List of dicts with keys: content, source_url, source_type, title, score
    """

    def _err(msg: str) -> list[dict]:
        return [{"content": f"[ERROR] {msg}", "source_url": "", "source_type": "error", "title": "Search error", "section": "", "score": 0.0}]

    try:
        if not query or not query.strip():
            return []

        top_k = max(1, min(20, top_k))

        if source_type and source_type not in ("docs", "forum", "vapi_book"):
            return _err(f"Invalid source_type '{source_type}'. Use 'docs', 'forum', 'vapi_book', or omit.")

        qdrant = get_qdrant_client()
        voyage = get_voyage_client()
        cohere_client = get_cohere_client()

        try:
            qdrant.get_collection(COLLECTION_NAME)
        except Exception as e:
            return _err(f"Collection '{COLLECTION_NAME}' not found. Run ingest first. ({e})")

        # 1. Embed query
        embedding_response = voyage.embed(
            [query],
            model="voyage-3-lite",
            input_type="query",
        )
        query_embedding = embedding_response.embeddings[0]

        # 2. Build optional source_type filter
        search_filter = None
        if source_type:
            search_filter = Filter(
                must=[FieldCondition(key="source_type", match=MatchValue(value=source_type))]
            )

        # 3. Vector search (support both old and new qdrant-client APIs)
        if hasattr(qdrant, "query_points"):
            query_response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=30,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
            )
            search_results = query_response.points
        else:
            search_results = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=30,
                query_filter=search_filter,
                with_payload=True,
            )

        if not search_results:
            return []

        # 4. Rerank with Cohere
        docs_to_rank = [r.payload.get("content", "") for r in search_results]
        try:
            rerank_response = cohere_client.rerank(
                model="rerank-3.5",
                query=query,
                documents=docs_to_rank,
                top_n=top_k,
            )
            ranked_indices = [result.index for result in rerank_response.results]
            rerank_scores = {result.index: result.relevance_score for result in rerank_response.results}
            ranked_results = [(search_results[i], rerank_scores.get(i, 0.0)) for i in ranked_indices]
        except Exception as e:
            print(f"[mcp] rerank failed, falling back to vector scores: {e}", file=sys.stderr)
            ranked_results = [(p, float(p.score)) for p in search_results[:top_k]]

        # 5. Format response
        output = []
        for point, score in ranked_results:
            payload = point.payload or {}
            output.append({
                "content": payload.get("content", ""),
                "source_url": payload.get("source_url", ""),
                "source_type": payload.get("source_type", ""),
                "title": payload.get("title", ""),
                "section": payload.get("section", ""),
                "score": float(score),
            })
        return output

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
    
    Request body:
    {
        "urls": ["https://example.com/doc1", "https://example.com/doc2"],
        "source_type": "docs"  # optional: docs, forum, vapi_book
    }
    """
    try:
        body = await request.json()
        urls = body.get("urls", [])
        source_type = body.get("source_type", "docs")
        
        if not urls:
            return JSONResponse({"error": "No URLs provided"})
        
        if not isinstance(urls, list):
            return JSONResponse({"error": "urls must be a list"})
        
        if source_type not in ("docs", "forum", "vapi_book"):
            return JSONResponse({"error": f"Invalid source_type '{source_type}'"})
        
        qdrant = get_qdrant_client()
        voyage = get_voyage_client()
        
        ensure_collection(qdrant)
        ingested_count = 0
        errors = []
        
        for url in urls:
            try:
                print(f"[ingest] Fetching {url}", flush=True)
                text, title = fetch_and_parse_url(url)
                
                # Chunk the text
                chunks = chunk_text(text, chunk_size=500, overlap=100)
                
                # Embed chunks
                print(f"[ingest] Embedding {len(chunks)} chunks from {url}", flush=True)
                embeddings_response = voyage.embed(
                    chunks,
                    model="voyage-3-lite",
                    input_type="document",
                )
                embeddings = embeddings_response.embeddings
                
                # Store in Qdrant
                points = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    point = PointStruct(
                        id=hash((url, i)) % (2**31),
                        vector=embedding,
                        payload={
                            "content": chunk,
                            "source_url": url,
                            "source_type": source_type,
                            "title": title,
                            "section": f"chunk_{i}",
                        }
                    )
                    points.append(point)
                
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                ingested_count += len(chunks)
                print(f"[ingest] ✓ Ingested {len(chunks)} chunks from {url}", flush=True)
                
            except Exception as e:
                error_msg = f"Failed to ingest {url}: {str(e)}"
                print(f"[ingest] ✗ {error_msg}", file=sys.stderr, flush=True)
                errors.append(error_msg)
        
        return JSONResponse({
            "status": "success",
            "ingested_count": ingested_count,
            "urls_processed": len(urls),
            "errors": errors if errors else None,
        })
    
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
