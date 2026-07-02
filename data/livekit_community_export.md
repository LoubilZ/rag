<user_query>
Voici le code complet à copier-coller dans tes fichiers locaux:

mcp_server.py:

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
from mcp.server.fastmcp import FastMCP

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_PATH = str(SCRIPT_DIR / "qdrant_storage")
COLLECTION_NAME = "livekit_kb"
PORT = int(os.getenv("PORT", "8080"))

# Initialize MCP server
mcp = FastMCP("livekit-rag", host="0.0.0.0", port=PORT)

# Global clients
_qdrant_client = None
_voyage_client = None
_cohere_client = None

def get_qdrant_client():
    """Lazy-load Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
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
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _cohere_client

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
        title = soup.title.string if soup.title else urlparse(url).netloc
        
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

        # 3. Vector search
        query_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=30,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False,
        )
        search_results = query_response.points

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
                "vector_size": 512,
            }
        except Exception:
            return {
                "collection": COLLECTION_NAME,
                "status": "not_found",
                "message": "Index not created. Run ingest first.",
            }
    except Exception as e:
        return {"error": str(e)}

@mcp.http_handler("POST", "/ingest")
async def ingest_urls(request) -> dict:
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
            return {"error": "No URLs provided"}
        
        if not isinstance(urls, list):
            return {"error": "urls must be a list"}
        
        if source_type not in ("docs", "forum", "vapi_book"):
            return {"error": f"Invalid source_type '{source_type}'"}
        
        qdrant = get_qdrant_client()
        voyage = get_voyage_client()
        
        # Create collection if it doesn't exist
        try:
            qdrant.get_collection(COLLECTION_NAME)
        except:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 1024, "distance": "Cosine"}
            )
        
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
        
        return {
            "status": "success",
            "ingested_count": ingested_count,
            "urls_processed": len(urls),
            "errors": errors if errors else None,
        }
    
    except Exception as e:
        return {"error": f"Ingest failed: {str(e)}"}

if __name__ == "__main__":
    print(f"[mcp] Starting server on 0.0.0.0:{PORT}", flush=True)
    print(f"[mcp] Qdrant path: {QDRANT_PATH}", flush=True)
    print(f"[mcp] VOYAGE_API_KEY set: {bool(VOYAGE_API_KEY)}", flush=True)
    print(f"[mcp] COHERE_API_KEY set: {bool(COHERE_API_KEY)}", flush=True)
    mcp.run(transport="streamable-http")

requirements.txt:

cohere==5.0.0
voyageai==0.3.0
qdrant-client==1.7.0
mcp==0.5.0
python-dotenv==1.0.0
requests==2.31.0
beautifulsoup4==4.12.0

Dockerfile:

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "mcp_server.py"]

.env.example:

VOYAGE_API_KEY=your_voyage_api_key
COHERE_API_KEY=your_cohere_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PORT=8080

Voilà. Copie tout ça dans tes fichiers locaux et tu es bon pour pusher.
</user_query>
<user_query>
vasi fait le
</user_query>
<user_query>
Une fois que tu as copié les fichiers:

Ouvre un terminal dans le dossier rag/:
cd /chemin/vers/ton/dossier/rag

Initialise git et pousse:
git init
git add .
git commit -m "Initial commit with ingest endpoint"
git branch -M main
git remote add origin https://github.com/LoubilZ/rag.git
git push -u origin main

Attends 2-3 minutes que Railway redéploie automatiquement

Teste l'endpoint:

curl -X POST https://livekit-rag-production.up.railway.app/ingest \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://docs.livekit.io"], "source_type": "docs"}'

Si ça marche, tu vas voir une réponse comme:

{
  "status": "success",
  "ingested_count": 42,
  "urls_processed": 1,
  "errors": null
}

Après, tu peux chercher dans ta KB avec:

curl https://livekit-rag-production.up.railway.app/sse \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I detect turns?"}'

T'as git installé?
</user_query>
<user_query>
vasi
</user_query>
<user_query>
ok et maintenant je peux faire quoi avec ce repo bien configuré ?
</user_query>
<user_query>
donne moi le prompt pour Qdrant stp
</user_query>
<user_query>
jai mis les variables
</user_query>
<user_query>
comment j'ajoute des documents
</user_query>
<user_query>
ok mais je peux te les donner là ?
</user_query>
<user_query>
CONTEXTE
Tu travailles sur le repo du serveur RAG "RAG-Voice Agent" (FastMCP + Qdrant),
déployé sur Railway. Il indexe de la doc sur les agents vocaux dans une collection
Qdrant `livekit_kb` (vecteurs 512 dim, ~10 494 points aujourd'hui) et l'expose via
2 outils MCP : `livekit_kb_stats` et `search_livekit_kb(query, source_type, top_k)`.
Il y a actuellement 3 valeurs de source_type : `docs`, `forum`, `vapi_book`.

OBJECTIF
On transforme ce RAG en base d'amélioration continue du PROMPTING d'agents vocaux.
Deux choses :
1) Ajouter 5 sources de haute qualité, centrées prompting.
2) Enrichir le schéma de métadonnées pour filtrer les recherches par (a) type de
   contenu et (b) architecture de pipeline — car la structure d'un prompt diffère
   selon qu'on est en pipeline cascadé STT-LLM-TTS ou en speech-to-speech (S2S).

ÉTAPE 1 — Comprendre l'ingest existant AVANT de coder
Repère la logique d'ingestion (grep `upsert` / `PointStruct` / `source_type` /
l'appel d'embedding). Détermine : boucle-t-elle sur des URLs, ou lit-elle des
fichiers locaux ? Réutilise EXACTEMENT le même chunker et le même modèle
d'embedding. Ne change PAS le modèle ni la dimension (512), sinon les vecteurs
seront incompatibles avec l'existant.

ÉTAPE 2 — Étendre le schéma de métadonnées
Ajoute 2 champs au payload de chaque chunk :
  - category      : "prompting" | "infra"
  - architecture  : "cascaded" | "s2s" | "both"

ÉTAPE 3 — Ajouter ces 5 sources (toutes category=prompting)
  URL                                                                        source_type          architecture
  https://docs.livekit.io/agents/start/prompting.md                         livekit_prompting    both
  https://developers.openai.com/cookbook/examples/realtime_prompting_guide  openai_realtime      s2s
  https://docs.vapi.ai/prompting-guide.md                                   vapi_prompting       both
  https://hamming.ai/resources/voice-agent-caller-identity-testing-checklist hamming_identity    both
  https://www.anthropic.com/engineering/writing-tools-for-agents            anthropic_tools      both

Nettoyage du contenu :
  - LiveKit : ingère directement le `.md` (déjà propre).
  - Vapi : prends le `.md` ci-dessus ; s'il 404, récupère le download markdown de
    la page /prompting-guide.
  - OpenAI / Hamming / Anthropic (HTML) : passe par un extracteur propre pour éviter
    le boilerplate (nav, code dupliqué). Option simple : préfixer l'URL par
    `https://r.jina.ai/` (Jina Reader) ; sinon trafilatura ou Firecrawl.

ÉTAPE 4 — Exposer les nouveaux filtres dans le tool
Ajoute 2 paramètres optionnels `category` et `architecture` à `search_livekit_kb`,
implémentés comme filtres de payload Qdrant (combinables avec source_type). Mets à
jour la docstring pour que le modèle sache qu'ils existent. Rends source_type
extensible (accepte les nouvelles valeurs, ne le fige pas sur les 3 anciennes).

ÉTAPE 5 (optionnel) — Documenter
Ajoute une courte note dans le README ou le CLAUDE.md du repo : à quoi servent ces
sources, le schéma de tags (category / architecture), et la liste des source_type.
Pour que la prochaine session ait le contexte.

CONTRAINTES — NE PAS CASSER LA PROD
  - NE recrée PAS la collection (`recreate_collection` détruit les données).
    Upsert dans la collection existante `livekit_kb`.
  - Garde le même modèle d'embedding et la dimension 512.
  - Préserve les 3 source_type existants et leur comportement.
  - Ingest idempotent : IDs de points déterministes (hash de url + index de chunk)
    pour ne pas dupliquer si on relance.
  - Les 10 494 points existants n'ont pas encore category/architecture : laisse-les
    tels quels (le rétro-tag est une tâche séparée). Donc les nouveaux filtres ne
    couvriront que les nouveaux chunks tant qu'on n'a pas rétro-taggé l'ancien corpus.

VALIDATION
  - `livekit_kb_stats` doit passer au-dessus de 10 494.
  - Un `search_livekit_kb` filtré (ex. category="prompting", architecture="s2s")
    doit remonter les nouveaux chunks avec les métadonnées correctes.
</user_query>
<user_query>
ok donc en gros tu n'as pas les documents quil y avait avant c'est ça ?
</user_query>
<user_query>
mais avant on avait pas qdrant
</user_query>
<user_query>
enfaite jai tout les anciens docs dans mon notion
</user_query>
<user_query>
https://cautious-cirrus-710.notion.site/Vapi-Book-352bd1f88e0e8024a439f7e9350154e6?source=copy_link
https://cautious-cirrus-710.notion.site/0ba419733e2848ccba59317c5f05dcc9?v=cb4a39345e6d49869c4e13ebe6b7c898&source=copy_link
https://cautious-cirrus-710.notion.site/LiveKit-Docs-Index-352bd1f88e0e81fcb677e34c27f1e112?source=copy_link
voici les liens des pages noptions mais en gros y'en a un c'est un glossaire de toute les pages live kit a inserer
</user_query>
<user_query>
oui vasi
</user_query>
<user_query>
alors ?
</user_query>
<user_query>
cest quoi les urls qui ne fonctionnes pas ?
</user_query>
<user_query>
rekance les a la fin
</user_query>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed). If there's no follow-ups needed, don't explicitly say that.</user_query>
<user_query>
ok et ducoup mtn il gaut que je rajoute les liens des forums livekit ou tu les as recupérés dans ma pagen otion ?
</user_query>
<user_query>
voici tous les liens : # LiveKit Community — Tous les liens

_Total : 1042 topics · export du 2026-07-02_

## Announcements (12)

- [LiveKit Agents for Node.js 1.5.0 is coming (Feedback Wanted)](https://community.livekit.io/t/livekit-agents-for-node-js-1-5-0-is-coming-feedback-wanted/1466) — 2026-06-20
- [Announcing LiveKit Dev Roundtable on 25th June 2026](https://community.livekit.io/t/announcing-livekit-dev-roundtable-on-25th-june-2026/1458) — 2026-06-18
- [Solving end-of-turn detection: LiveKit Turn Detector v1.0](https://community.livekit.io/t/solving-end-of-turn-detection-livekit-turn-detector-v1-0/1453) — 2026-06-17
- [Introducing LiveKit Portal: Production-Grade Stack for Teleoperation and Remote Inference on Robots](https://community.livekit.io/t/introducing-livekit-portal-production-grade-stack-for-teleoperation-and-remote-inference-on-robots/1412) — 2026-06-11
- [Introducing the LiveKit C++ SDK](https://community.livekit.io/t/introducing-the-livekit-c-sdk/1330) — 2026-06-03
- [Announcing LiveKit Dev Roundtable on 28th May 2026](https://community.livekit.io/t/announcing-livekit-dev-roundtable-on-28th-may-2026/1196) — 2026-05-21
- [Announcing Answering Machine Detection](https://community.livekit.io/t/announcing-answering-machine-detection/1116) — 2026-05-14
- [Announcing Agent Console](https://community.livekit.io/t/announcing-agent-console/870) — 2026-04-15
- [Announcing LiveKit Dev Roundtable on 9th April 2026](https://community.livekit.io/t/announcing-livekit-dev-roundtable-on-9th-april-2026/762) — 2026-04-02
- [Introducing Agents UI, an open-source shadcn component library](https://community.livekit.io/t/introducing-agents-ui-an-open-source-shadcn-component-library/443) — 2026-03-03
- [Welcome to the LiveKit community](https://community.livekit.io/t/welcome-to-the-livekit-community/49) — 2025-12-30
- [About the Announcements category](https://community.livekit.io/t/about-the-announcements-category/10) — 2025-12-22

## Getting Started (242)

- [Agent voice breaking in S3 composite recording but clean in LiveKit Observability](https://community.livekit.io/t/agent-voice-breaking-in-s3-composite-recording-but-clean-in-livekit-observability/1538) — 2026-07-01
- [LiveKit API secret consistently 43 characters instead of 44 — getting persistent 401 invalid token](https://community.livekit.io/t/livekit-api-secret-consistently-43-characters-instead-of-44-getting-persistent-401-invalid-token/1524) — 2026-06-29
- [I am not able to connect my agent there is no option to show my agent but my agent still running](https://community.livekit.io/t/i-am-not-able-to-connect-my-agent-there-is-no-option-to-show-my-agent-but-my-agent-still-running/1522) — 2026-06-28
- [Agent observability partial setting not working](https://community.livekit.io/t/agent-observability-partial-setting-not-working/1518) — 2026-06-26
- [Livekit experience improvement, lag in response, delay in generation …](https://community.livekit.io/t/livekit-experience-improvement-lag-in-response-delay-in-generation/1515) — 2026-06-26
- [Safari doesn’t allow joining the LiveKit room if the microphone is enabled](https://community.livekit.io/t/safari-doesn-t-allow-joining-the-livekit-room-if-the-microphone-is-enabled/1513) — 2026-06-26
- [What causes dropping pass-through signal — no stream available? Agent not receiving room closure events](https://community.livekit.io/t/what-causes-dropping-pass-through-signal-no-stream-available-agent-not-receiving-room-closure-events/1502) — 2026-06-24
- [EU AI Act ”you're talking to an AI” disclosure — how are you handling it on LiveKit? (Article 50, live Aug 2)](https://community.livekit.io/t/eu-ai-act-youre-talking-to-an-ai-disclosure-how-are-you-handling-it-on-livekit-article-50-live-aug-2/1497) — 2026-06-24
- [Do i need to open the ports](https://community.livekit.io/t/do-i-need-to-open-the-ports/1494) — 2026-06-23
- [Livekit sip unable to resolve PRACK and 100rel](https://community.livekit.io/t/livekit-sip-unable-to-resolve-prack-and-100rel/1491) — 2026-06-23
- [Fresh LiveKit Cloud project and API keys. Both Agent Worker startup and RoomService/ListRooms API calls return 401 Unauthorized. Using livekit-agents 1.6.2, livekit-api 1.1.0, Python 3.13. Looking for guidance on authentication or project configuration re](https://community.livekit.io/t/fresh-livekit-cloud-project-and-api-keys-both-agent-worker-startup-and-roomservice-listrooms-api-calls-return-401-unauthorized-using-livekit-agents-1-6-2-livekit-api-1-1-0-python-3-13-looking-for-guidance-on-authentication-or-project-configuration-re/1488) — 2026-06-23
- [STT via LiveKit Inference (Deepgram): two issues on SIP/phone calls](https://community.livekit.io/t/stt-via-livekit-inference-deepgram-two-issues-on-sip-phone-calls/1487) — 2026-06-23
- [After '500 no media node available', workers get no dispatch until manually restarted — why?](https://community.livekit.io/t/after-500-no-media-node-available-workers-get-no-dispatch-until-manually-restarted-why/1484) — 2026-06-22
- [SIP connection fails](https://community.livekit.io/t/sip-connection-fails/1483) — 2026-06-22
- [Cloud turn detector failed + late STT final warnings after enabling LiveKit inference audio turn detection/vad](https://community.livekit.io/t/cloud-turn-detector-failed-late-stt-final-warnings-after-enabling-livekit-inference-audio-turn-detection-vad/1472) — 2026-06-21
- [Question: Built-in LiveKit way to strip silence before STT?](https://community.livekit.io/t/question-built-in-livekit-way-to-strip-silence-before-stt/1471) — 2026-06-21
- [After-hours AI agent with warm transfer while keeping existing number](https://community.livekit.io/t/after-hours-ai-agent-with-warm-transfer-while-keeping-existing-number/1468) — 2026-06-20
- [Question about speech-to-speech models via LiveKit Inference](https://community.livekit.io/t/question-about-speech-to-speech-models-via-livekit-inference/1461) — 2026-06-19
- [Clarification on usage of vad/turn detector about whether only Agentsession supports or Agent also supports it?](https://community.livekit.io/t/clarification-on-usage-of-vad-turn-detector-about-whether-only-agentsession-supports-or-agent-also-supports-it/1460) — 2026-06-19
- [Room-composite egress migrates mid-call, finalizes early, and silently truncates the recording (captured ~30% of the call) + fires 'egress_ended' while the call is still live](https://community.livekit.io/t/room-composite-egress-migrates-mid-call-finalizes-early-and-silently-truncates-the-recording-captured-30-of-the-call-fires-egress-ended-while-the-call-is-still-live/1459) — 2026-06-18
- [Gpt realtime transcription misses](https://community.livekit.io/t/gpt-realtime-transcription-misses/1456) — 2026-06-18
- [Is it possible to increase adaptive interruption quota for a project?](https://community.livekit.io/t/is-it-possible-to-increase-adaptive-interruption-quota-for-a-project/1452) — 2026-06-17
- [Query about fastest TTFT livekit inference model](https://community.livekit.io/t/query-about-fastest-ttft-livekit-inference-model/1451) — 2026-06-17
- [EOT value leads to around 30% of e2e latency, Tried reducing min/max endpointint in turn delay plugin but no luck](https://community.livekit.io/t/eot-value-leads-to-around-30-of-e2e-latency-tried-reducing-min-max-endpointint-in-turn-delay-plugin-but-no-luck/1450) — 2026-06-17
- [Session.say() hangs indefinitely (~20s+) when using Gemini RealtimeModel(gemini 3.1) as llm together with google.beta.GeminiTTS as tts; session.input.set_audio_enabled() / aec_warmup_duration do not reliably block user interruption in this configuration](https://community.livekit.io/t/session-say-hangs-indefinitely-20s-when-using-gemini-realtimemodel-gemini-3-1-as-llm-together-with-google-beta-geminitts-as-tts-session-input-set-audio-enabled-aec-warmup-duration-do-not-reliably-block-user-interruption-in-this-configuration/1448) — 2026-06-17
- [Settings is not working](https://community.livekit.io/t/settings-is-not-working/1441) — 2026-06-16
- [Server-initiated migration fails to resume on agents 1.4.6 — subscriber + publisher PC fail, no recovery, process killed (expected fixed in >1.4.2 per agents #4705)](https://community.livekit.io/t/server-initiated-migration-fails-to-resume-on-agents-1-4-6-subscriber-publisher-pc-fail-no-recovery-process-killed-expected-fixed-in-1-4-2-per-agents-4705/1440) — 2026-06-16
- [Critical Egress Recording Discrepancies](https://community.livekit.io/t/critical-egress-recording-discrepancies/1438) — 2026-06-16
- [Circular structure to JSON when using Agent Builder](https://community.livekit.io/t/circular-structure-to-json-when-using-agent-builder/1436) — 2026-06-15
- [Make the agent announce prior to calling the tool (functionality not available)](https://community.livekit.io/t/make-the-agent-announce-prior-to-calling-the-tool-functionality-not-available/1434) — 2026-06-15
- [Title: 6–12s latency on the first model response after a pre-recorded greeting, even with early session initialization (Gemini 3.1 Flash Live)](https://community.livekit.io/t/title-6-12s-latency-on-the-first-model-response-after-a-pre-recorded-greeting-even-with-early-session-initialization-gemini-3-1-flash-live/1431) — 2026-06-15
- [Livekit inference GPT-5 mini does not works](https://community.livekit.io/t/livekit-inference-gpt-5-mini-does-not-works/1429) — 2026-06-14
- [Incorrect/false high e2e Latency > 203 seconds OR sometimes not present at all in chat item](https://community.livekit.io/t/incorrect-false-high-e2e-latency-203-seconds-or-sometimes-not-present-at-all-in-chat-item/1425) — 2026-06-12
- [Getting error on agent builder during the agent designing work](https://community.livekit.io/t/getting-error-on-agent-builder-during-the-agent-designing-work/1422) — 2026-06-12
- [”process is unresponsive, killing process” - happens intermittently on long Gemini 3.1 Flash Live calls, on_session_end shutdown callback often doesn't run](https://community.livekit.io/t/process-is-unresponsive-killing-process-happens-intermittently-on-long-gemini-3-1-flash-live-calls-on-session-end-shutdown-callback-often-doesnt-run/1420) — 2026-06-12
- [Downloading files to make a livekit agent - Pytorch and SSL](https://community.livekit.io/t/downloading-files-to-make-a-livekit-agent-pytorch-and-ssl/1413) — 2026-06-11
- [LiveKit Support for Transcript Normalization Before LLM Calls](https://community.livekit.io/t/livekit-support-for-transcript-normalization-before-llm-calls/1405) — 2026-06-11
- [Inbound SIP call: agent job dispatched ~8s after the INVITE reached LiveKit Cloud — caller had already hung up](https://community.livekit.io/t/inbound-sip-call-agent-job-dispatched-8s-after-the-invite-reached-livekit-cloud-caller-had-already-hung-up/1393) — 2026-06-09
- [Two to three seconds of no response](https://community.livekit.io/t/two-to-three-seconds-of-no-response/1392) — 2026-06-09
- [2 Months since VP8/VP9 is broken on macs](https://community.livekit.io/t/2-months-since-vp8-vp9-is-broken-on-macs/1391) — 2026-06-09
- [Livekit rtc audioframe](https://community.livekit.io/t/livekit-rtc-audioframe/1390) — 2026-06-09
- [Gemini Explicit Context Caching (cached_content) drops system_instruction in livekit.plugins.google — How to inject dynamic session variables?](https://community.livekit.io/t/gemini-explicit-context-caching-cached-content-drops-system-instruction-in-livekit-plugins-google-how-to-inject-dynamic-session-variables/1386) — 2026-06-09
- [Question about LiveKit Observability, Recording, and Data Residency in India](https://community.livekit.io/t/question-about-livekit-observability-recording-and-data-residency-in-india/1385) — 2026-06-09
- [Who to Contact Regarding Billing?](https://community.livekit.io/t/who-to-contact-regarding-billing/1382) — 2026-06-09
- [Question about LiveKit data transfer billing](https://community.livekit.io/t/question-about-livekit-data-transfer-billing/1380) — 2026-06-08
- [Signal connection times out on the ”v0 path” at agent join, forcing a fallback that adds 0.5–5s of call-setup latency](https://community.livekit.io/t/signal-connection-times-out-on-the-v0-path-at-agent-join-forcing-a-fallback-that-adds-0-5-5s-of-call-setup-latency/1377) — 2026-06-08
- [Open hanging session : cannot delete sessions](https://community.livekit.io/t/open-hanging-session-cannot-delete-sessions/1374) — 2026-06-08
- [How should lk agent deploy handle Node workspace agents in monorepos?](https://community.livekit.io/t/how-should-lk-agent-deploy-handle-node-workspace-agents-in-monorepos/1372) — 2026-06-08
- [What timezone do monthly quota resets use?](https://community.livekit.io/t/what-timezone-do-monthly-quota-resets-use/1370) — 2026-06-08
- [Will upgrading from free plan enable multiple participants per room for co-streaming?](https://community.livekit.io/t/will-upgrading-from-free-plan-enable-multiple-participants-per-room-for-co-streaming/1363) — 2026-06-07
- [How do I handle interruptions with a realtime openai model?](https://community.livekit.io/t/how-do-i-handle-interruptions-with-a-realtime-openai-model/1356) — 2026-06-05
- [How can I make my agent stop hallucinating dates and know the current date?](https://community.livekit.io/t/how-can-i-make-my-agent-stop-hallucinating-dates-and-know-the-current-date/1355) — 2026-06-05
- [Cutting LLM round-trip latency for the voice agent](https://community.livekit.io/t/cutting-llm-round-trip-latency-for-the-voice-agent/1350) — 2026-06-05
- [I need to ask about phone numbers and payment](https://community.livekit.io/t/i-need-to-ask-about-phone-numbers-and-payment/1347) — 2026-06-05
- [Gemini 3.1 Flash Live model giving 2 major issues](https://community.livekit.io/t/gemini-3-1-flash-live-model-giving-2-major-issues/1346) — 2026-06-05
- [Livekit is taking time to respond to calls](https://community.livekit.io/t/livekit-is-taking-time-to-respond-to-calls/1344) — 2026-06-05
- [Egress API returned 404 ”requested room does not exist” for ~25s — on rooms that were live (agents connected)](https://community.livekit.io/t/egress-api-returned-404-requested-room-does-not-exist-for-25s-on-rooms-that-were-live-agents-connected/1342) — 2026-06-04
- [Duplicate egress triggered for a room — duplicate egress_ended webhooks before room ended, and no way to tell which egress instance sent them](https://community.livekit.io/t/duplicate-egress-triggered-for-a-room-duplicate-egress-ended-webhooks-before-room-ended-and-no-way-to-tell-which-egress-instance-sent-them/1339) — 2026-06-04
- [Customer Number Not Displayed During SIP REFER Call Transfer - Plivo Provider](https://community.livekit.io/t/customer-number-not-displayed-during-sip-refer-call-transfer-plivo-provider/1336) — 2026-06-04
- [Issue with Agent Console](https://community.livekit.io/t/issue-with-agent-console/1331) — 2026-06-03
- [How Does LiveKit Route Agent Jobs Across Multiple EC2 Instances and Support Autoscaling?](https://community.livekit.io/t/how-does-livekit-route-agent-jobs-across-multiple-ec2-instances-and-support-autoscaling/1329) — 2026-06-03
- [Gemini live on livekit cloud](https://community.livekit.io/t/gemini-live-on-livekit-cloud/1318) — 2026-06-02
- [STT Accuracy Issues with Single-Word Answers](https://community.livekit.io/t/stt-accuracy-issues-with-single-word-answers/1317) — 2026-06-02
- [STT Audio Never Reaches Agent Despite Successful Track Subscription (Started ~May 31)](https://community.livekit.io/t/stt-audio-never-reaches-agent-despite-successful-track-subscription-started-may-31/1314) — 2026-06-02
- [WhatsApp Connector: inbound calls are answered inconsistently](https://community.livekit.io/t/whatsapp-connector-inbound-calls-are-answered-inconsistently/1311) — 2026-06-01
- [Enterprise TURN/STUN Connectivity Issues and LiveKit Migration Evaluation](https://community.livekit.io/t/enterprise-turn-stun-connectivity-issues-and-livekit-migration-evaluation/1307) — 2026-06-01
- [How to get a BAA signed with LiveKit - process, requirements & timeline?](https://community.livekit.io/t/how-to-get-a-baa-signed-with-livekit-process-requirements-timeline/1297) — 2026-06-01
- [Latency advice needed: OpenAI Realtime-quality conversation + Runway avatar via LiveKit Agent is too slow](https://community.livekit.io/t/latency-advice-needed-openai-realtime-quality-conversation-runway-avatar-via-livekit-agent-is-too-slow/1294) — 2026-06-01
- [AgentDispatchService/CreateDispatch does not auto-create room](https://community.livekit.io/t/agentdispatchservice-createdispatch-does-not-auto-create-room/1285) — 2026-05-29
- [What does this error mean ? And how can I add a fallback when this happens?](https://community.livekit.io/t/what-does-this-error-mean-and-how-can-i-add-a-fallback-when-this-happens/1284) — 2026-05-29
- [Livekit Outage - Production](https://community.livekit.io/t/livekit-outage-production/1273) — 2026-05-28
- [I can't use live mode when I go to www.agents-playground.livekit.io autocratically sends me to https://cloud.livekit.io/projects/ and because of that I can't use my agent that I'm trying to run on python please help](https://community.livekit.io/t/i-cant-use-live-mode-when-i-go-to-www-agents-playground-livekit-io-autocratically-sends-me-to-https-cloud-livekit-io-projects-and-because-of-that-i-cant-use-my-agent-that-im-trying-to-run-on-python-please-help/1263) — 2026-05-27
- [Cartesia TTS Error](https://community.livekit.io/t/cartesia-tts-error/1254) — 2026-05-26
- [Us-west-2 region support for livekit agents](https://community.livekit.io/t/us-west-2-region-support-for-livekit-agents/1235) — 2026-05-25
- [Bug Report: Session Showing ACTIVE Despite Already Ended](https://community.livekit.io/t/bug-report-session-showing-active-despite-already-ended/1233) — 2026-05-25
- [Sessions showing active on livekit portal even though in events we see an event of Room ended](https://community.livekit.io/t/sessions-showing-active-on-livekit-portal-even-though-in-events-we-see-an-event-of-room-ended/1230) — 2026-05-25
- [Unstability with livekit plugins for azure openai realtime](https://community.livekit.io/t/unstability-with-livekit-plugins-for-azure-openai-realtime/1226) — 2026-05-25
- [Please enable WhatsApp Connector for my project](https://community.livekit.io/t/please-enable-whatsapp-connector-for-my-project/1211) — 2026-05-23
- [Production Bug: Weird Background Noise in Agent Audio – LiveKit](https://community.livekit.io/t/production-bug-weird-background-noise-in-agent-audio-livekit/1207) — 2026-05-22
- [Data residency for Deepgram STT when using LiveKit Inference in India region](https://community.livekit.io/t/data-residency-for-deepgram-stt-when-using-livekit-inference-in-india-region/1203) — 2026-05-21
- [Dynamic LLM Model Switching During an Active LiveKit Voice Call](https://community.livekit.io/t/dynamic-llm-model-switching-during-an-active-livekit-voice-call/1180) — 2026-05-20
- [AI Agent async tool calls causes a lot of LLM conversation messages](https://community.livekit.io/t/ai-agent-async-tool-calls-causes-a-lot-of-llm-conversation-messages/1178) — 2026-05-20
- [Application-level turn-state bug](https://community.livekit.io/t/application-level-turn-state-bug/1162) — 2026-05-18
- [Room Session Not Finishing Despite no Participants](https://community.livekit.io/t/room-session-not-finishing-despite-no-participants/1139) — 2026-05-15
- [Livekit cloud connectivity test](https://community.livekit.io/t/livekit-cloud-connectivity-test/1136) — 2026-05-15
- [Question about free inference credits limit](https://community.livekit.io/t/question-about-free-inference-credits-limit/1131) — 2026-05-15
- [One of the session is not ending](https://community.livekit.io/t/one-of-the-session-is-not-ending/1128) — 2026-05-14
- [REMOVING OF https://agents-playground.livekit.io/](https://community.livekit.io/t/removing-of-https-agents-playground-livekit-io/1127) — 2026-05-14
- [Participant was audible in room/demo recording, but LiveKit Agent heard very low audio and STT/VAD pipeline missed most of the response](https://community.livekit.io/t/participant-was-audible-in-room-demo-recording-but-livekit-agent-heard-very-low-audio-and-stt-vad-pipeline-missed-most-of-the-response/1124) — 2026-05-14
- [Livekit-agents@1.5.9 Latest](https://community.livekit.io/t/livekit-agents-1-5-9-latest/1112) — 2026-05-13
- [Trying to understand how cloud limit on STT and TTS reset](https://community.livekit.io/t/trying-to-understand-how-cloud-limit-on-stt-and-tts-reset/1107) — 2026-05-13
- [Interruption handling model region/plan restrictions](https://community.livekit.io/t/interruption-handling-model-region-plan-restrictions/1099) — 2026-05-12
- [My agent doesn't work locally and in production](https://community.livekit.io/t/my-agent-doesnt-work-locally-and-in-production/1092) — 2026-05-12
- [Which version of Livekit should I use to not have any problems?](https://community.livekit.io/t/which-version-of-livekit-should-i-use-to-not-have-any-problems/1086) — 2026-05-12
- [WebRTC connection stability and background process interference on mobile](https://community.livekit.io/t/webrtc-connection-stability-and-background-process-interference-on-mobile/1081) — 2026-05-11
- [Agent not speaking](https://community.livekit.io/t/agent-not-speaking/1077) — 2026-05-11
- [Issue with a LiveKit-provided phone number](https://community.livekit.io/t/issue-with-a-livekit-provided-phone-number/1073) — 2026-05-10
- [Ai Voice not working](https://community.livekit.io/t/ai-voice-not-working/1069) — 2026-05-09
- [Get language from STT plugin](https://community.livekit.io/t/get-language-from-stt-plugin/1043) — 2026-05-06
- [Gemini 3.1 Flash Live Preview](https://community.livekit.io/t/gemini-3-1-flash-live-preview/1041) — 2026-05-06
- [Free plan minutes](https://community.livekit.io/t/free-plan-minutes/1033) — 2026-05-06
- [Does the ask-ai channel have any advantages over using the LiveKit Docs MCP server in an agentic coding session?](https://community.livekit.io/t/does-the-ask-ai-channel-have-any-advantages-over-using-the-livekit-docs-mcp-server-in-an-agentic-coding-session/1027) — 2026-05-05
- [Lowest-latency setup](https://community.livekit.io/t/lowest-latency-setup/1026) — 2026-05-05
- [Livekit Ship Plan](https://community.livekit.io/t/livekit-ship-plan/1021) — 2026-05-04
- [WebRTC track subscription failures and signal connection timeouts in high-concurrency mobile environments](https://community.livekit.io/t/webrtc-track-subscription-failures-and-signal-connection-timeouts-in-high-concurrency-mobile-environments/1017) — 2026-05-04
- [If you're building on LiveKit, I wanna talk to you!](https://community.livekit.io/t/if-youre-building-on-livekit-i-wanna-talk-to-you/1015) — 2026-05-04
- [Guys I have a problem](https://community.livekit.io/t/guys-i-have-a-problem/1010) — 2026-05-02
- [Inbound calls are failing with Room Subscription Timed Out error](https://community.livekit.io/t/inbound-calls-are-failing-with-room-subscription-timed-out-error/1004) — 2026-05-01
- [Billing plan updated but changes not reflecting](https://community.livekit.io/t/billing-plan-updated-but-changes-not-reflecting/995) — 2026-04-30
- [LiveKit + Deepgram STT not understanding Hindi voice (real-time issue)](https://community.livekit.io/t/livekit-deepgram-stt-not-understanding-hindi-voice-real-time-issue/989) — 2026-04-30
- [LLM Comparision](https://community.livekit.io/t/llm-comparision/988) — 2026-04-30
- [Request to enable WhatsApp Connector (Accept + Initiate) for project](https://community.livekit.io/t/request-to-enable-whatsapp-connector-accept-initiate-for-project/978) — 2026-04-28
- [403 ”whatsapp call is not enabled for this project”](https://community.livekit.io/t/403-whatsapp-call-is-not-enabled-for-this-project/959) — 2026-04-27
- [Livekit realtime using gemini-live-2.5-flash-native-audio](https://community.livekit.io/t/livekit-realtime-using-gemini-live-2-5-flash-native-audio/956) — 2026-04-27
- [Is it possible to add prefix for webhook token](https://community.livekit.io/t/is-it-possible-to-add-prefix-for-webhook-token/949) — 2026-04-25
- [Agent silence Gap](https://community.livekit.io/t/agent-silence-gap/947) — 2026-04-24
- [I don't understand](https://community.livekit.io/t/i-dont-understand/946) — 2026-04-23
- [Telephony Voice AI listening errors](https://community.livekit.io/t/telephony-voice-ai-listening-errors/939) — 2026-04-23
- [STT from livekit inference and assembly AI returning error](https://community.livekit.io/t/stt-from-livekit-inference-and-assembly-ai-returning-error/936) — 2026-04-23
- [WHIP Ingress ICE fails with “Failed to ping without candidate pairs” - UDP port range not initializing](https://community.livekit.io/t/whip-ingress-ice-fails-with-failed-to-ping-without-candidate-pairs-udp-port-range-not-initializing/934) — 2026-04-22
- [Problem with GetEmailTask](https://community.livekit.io/t/problem-with-getemailtask/933) — 2026-04-22
- [SIP inbound returns 503 but no record in Calls dashboard](https://community.livekit.io/t/sip-inbound-returns-503-but-no-record-in-calls-dashboard/930) — 2026-04-22
- [Auto STT model errors leads to stalled agent](https://community.livekit.io/t/auto-stt-model-errors-leads-to-stalled-agent/923) — 2026-04-21
- [LiveKit Cloud - Zombie session](https://community.livekit.io/t/livekit-cloud-zombie-session/919) — 2026-04-21
- [Text streams missing in front end](https://community.livekit.io/t/text-streams-missing-in-front-end/917) — 2026-04-21
- [Call visibility in livekit cloud](https://community.livekit.io/t/call-visibility-in-livekit-cloud/915) — 2026-04-20
- [Proactive Greeting fails with Gemini 3.1 Flash Live (agent silent until user speaks)](https://community.livekit.io/t/proactive-greeting-fails-with-gemini-3-1-flash-live-agent-silent-until-user-speaks/913) — 2026-04-20
- [Self-hosting only the Egress service while using LiveKit Cloud — has anyone tried this?](https://community.livekit.io/t/self-hosting-only-the-egress-service-while-using-livekit-cloud-has-anyone-tried-this/906) — 2026-04-20
- [Long prompt - help](https://community.livekit.io/t/long-prompt-help/895) — 2026-04-16
- [Where can I set region pinning?](https://community.livekit.io/t/where-can-i-set-region-pinning/877) — 2026-04-15
- [AI_coustics blocking pipeline (?)](https://community.livekit.io/t/ai-coustics-blocking-pipeline/876) — 2026-04-15
- [Enabling Observability 404 error](https://community.livekit.io/t/enabling-observability-404-error/852) — 2026-04-13
- [Connection Tester and JS client both fail with ‘invalid authorization token’ on new API key](https://community.livekit.io/t/connection-tester-and-js-client-both-fail-with-invalid-authorization-token-on-new-api-key/850) — 2026-04-13
- [Is there any configuration where publisher and consumer create peer to peer communication?](https://community.livekit.io/t/is-there-any-configuration-where-publisher-and-consumer-create-peer-to-peer-communication/845) — 2026-04-13
- [Had any one had actual success Live streaming](https://community.livekit.io/t/had-any-one-had-actual-success-live-streaming/840) — 2026-04-11
- [Discrepancy between session audio and what egress posted to our endpoint](https://community.livekit.io/t/discrepancy-between-session-audio-and-what-egress-posted-to-our-endpoint/836) — 2026-04-10
- [Agent does't reply in Android app from agent-starter-android repo](https://community.livekit.io/t/agent-doest-reply-in-android-app-from-agent-starter-android-repo/828) — 2026-04-10
- [Slow draining old versions](https://community.livekit.io/t/slow-draining-old-versions/826) — 2026-04-09
- [Can the agent hangup?](https://community.livekit.io/t/can-the-agent-hangup/815) — 2026-04-08
- [Pronunciation Dictionary Limitations in Inference (Cartesia Sonic 3)](https://community.livekit.io/t/pronunciation-dictionary-limitations-in-inference-cartesia-sonic-3/812) — 2026-04-08
- [Process memory usage is high (nodejs livekit agents) (using cloud livekit)](https://community.livekit.io/t/process-memory-usage-is-high-nodejs-livekit-agents-using-cloud-livekit/792) — 2026-04-07
- [Room is closed when the last “non programmatic” participant leaves](https://community.livekit.io/t/room-is-closed-when-the-last-non-programmatic-participant-leaves/791) — 2026-04-07
- [V1.2.3 proc_pool.js regression — numIdleProcesses=2 only spawns 1 child at boot due to initMutex lifetime change in procWatchTask](https://community.livekit.io/t/v1-2-3-proc-pool-js-regression-numidleprocesses-2-only-spawns-1-child-at-boot-due-to-initmutex-lifetime-change-in-procwatchtask/789) — 2026-04-07
- [Background Tool Execution with LiveKit Agents](https://community.livekit.io/t/background-tool-execution-with-livekit-agents/778) — 2026-04-06
- [When I use docker, agent fails to join after second time](https://community.livekit.io/t/when-i-use-docker-agent-fails-to-join-after-second-time/777) — 2026-04-05
- [With the new adaptive interruption feature, im seeing the same agent interruption behaviour resume_false_interruption True or False](https://community.livekit.io/t/with-the-new-adaptive-interruption-feature-im-seeing-the-same-agent-interruption-behaviour-resume-false-interruption-true-or-false/776) — 2026-04-05
- [How to check INBOUND CALL or OUTBOUND CALL in rtc.session entrypoint and divide logic](https://community.livekit.io/t/how-to-check-inbound-call-or-outbound-call-in-rtc-session-entrypoint-and-divide-logic/772) — 2026-04-04
- [Getting LiveKit Running Unity](https://community.livekit.io/t/getting-livekit-running-unity/768) — 2026-04-03
- [Gemini 3.1 Flash Live not working with LiveKit Agents 1.5.1](https://community.livekit.io/t/gemini-3-1-flash-live-not-working-with-livekit-agents-1-5-1/757) — 2026-04-02
- [”failed to connect to livekit” warnings](https://community.livekit.io/t/failed-to-connect-to-livekit-warnings/747) — 2026-04-01
- [Outbound SIP IP addresses](https://community.livekit.io/t/outbound-sip-ip-addresses/744) — 2026-04-01
- [Issue with lk.chat not triggering response in Agents 1.5+](https://community.livekit.io/t/issue-with-lk-chat-not-triggering-response-in-agents-1-5/741) — 2026-04-01
- [Gemini 3.1 Flash Lite Preview](https://community.livekit.io/t/gemini-3-1-flash-lite-preview/739) — 2026-04-01
- [Want to play a music while executing a tool it should play parallelly with the execution of the api clal](https://community.livekit.io/t/want-to-play-a-music-while-executing-a-tool-it-should-play-parallelly-with-the-execution-of-the-api-clal/733) — 2026-03-31
- [What's the difference of the plans?](https://community.livekit.io/t/whats-the-difference-of-the-plans/727) — 2026-03-31
- [Anyone else having issues with Live 3.1?](https://community.livekit.io/t/anyone-else-having-issues-with-live-3-1/717) — 2026-03-30
- [HTTP Tool Parameters](https://community.livekit.io/t/http-tool-parameters/707) — 2026-03-28
- [Multylanguage with ElevenLabs not working](https://community.livekit.io/t/multylanguage-with-elevenlabs-not-working/706) — 2026-03-28
- [Error message on all projects: Your project has exceeded its free tier limit](https://community.livekit.io/t/error-message-on-all-projects-your-project-has-exceeded-its-free-tier-limit/705) — 2026-03-28
- [Behaviour of Gemini Live 3.1 model in LiveKit (not consistent)](https://community.livekit.io/t/behaviour-of-gemini-live-3-1-model-in-livekit-not-consistent/704) — 2026-03-28
- [Trying gemini 3.1 flash live and I can't seem to make it start talking?](https://community.livekit.io/t/trying-gemini-3-1-flash-live-and-i-cant-seem-to-make-it-start-talking/690) — 2026-03-26
- [Gemini 3.1 flash live](https://community.livekit.io/t/gemini-3-1-flash-live/689) — 2026-03-26
- [Esp32 rtc error](https://community.livekit.io/t/esp32-rtc-error/682) — 2026-03-26
- [Add api ElevenLabs key to agents TTS](https://community.livekit.io/t/add-api-elevenlabs-key-to-agents-tts/680) — 2026-03-25
- [Sessions analytics populate, but Sessions table remains empty for agent calls](https://community.livekit.io/t/sessions-analytics-populate-but-sessions-table-remains-empty-for-agent-calls/678) — 2026-03-25
- [Missing feature in Sandbox? Unable to deploy Next.js Starter App as per documentation](https://community.livekit.io/t/missing-feature-in-sandbox-unable-to-deploy-next-js-starter-app-as-per-documentation/677) — 2026-03-25
- [High Latency (~2–3s) in LiveKit Voice Agent with Plivo](https://community.livekit.io/t/high-latency-2-3s-in-livekit-voice-agent-with-plivo/675) — 2026-03-25
- [Adaptive interruption error nodejs agents sdk](https://community.livekit.io/t/adaptive-interruption-error-nodejs-agents-sdk/670) — 2026-03-25
- [FallbackAdapter with aligned_transcript](https://community.livekit.io/t/fallbackadapter-with-aligned-transcript/663) — 2026-03-24
- [Livekit Inference no-thinking config for google gemini 2.5 flash model](https://community.livekit.io/t/livekit-inference-no-thinking-config-for-google-gemini-2-5-flash-model/662) — 2026-03-24
- [I have issue for connection of RTC](https://community.livekit.io/t/i-have-issue-for-connection-of-rtc/654) — 2026-03-24
- [The explanation of session events](https://community.livekit.io/t/the-explanation-of-session-events/644) — 2026-03-23
- [Latest livekit-server image is having some critical vulnerabilities](https://community.livekit.io/t/latest-livekit-server-image-is-having-some-critical-vulnerabilities/643) — 2026-03-23
- [My Secret Weapon for LiveKit Answers](https://community.livekit.io/t/my-secret-weapon-for-livekit-answers/640) — 2026-03-21
- [Is Adaptive interruption HIPAA eligible?](https://community.livekit.io/t/is-adaptive-interruption-hipaa-eligible/637) — 2026-03-20
- [Adaptive interruption for realtime model](https://community.livekit.io/t/adaptive-interruption-for-realtime-model/632) — 2026-03-20
- [Adaptive interruption for self hosted](https://community.livekit.io/t/adaptive-interruption-for-self-hosted/625) — 2026-03-20
- [Solving unwanted interruptions with Adaptive Interruption Handling](https://community.livekit.io/t/solving-unwanted-interruptions-with-adaptive-interruption-handling/620) — 2026-03-19
- [Double latency when using telephony with livekit](https://community.livekit.io/t/double-latency-when-using-telephony-with-livekit/609) — 2026-03-18
- [Access for the beta connector in my project](https://community.livekit.io/t/access-for-the-beta-connector-in-my-project/571) — 2026-03-16
- [Static IPs Correct?](https://community.livekit.io/t/static-ips-correct/570) — 2026-03-16
- [Whatsapp call is not enabled for this project](https://community.livekit.io/t/whatsapp-call-is-not-enabled-for-this-project/569) — 2026-03-15
- [How to initialize per-participant / per-user MCP server in LiveKit Agents (Python)](https://community.livekit.io/t/how-to-initialize-per-participant-per-user-mcp-server-in-livekit-agents-python/565) — 2026-03-15
- [M5Stack CoreS3 + AW88298/ES7210 shared I2S — no speaker output (capture works)](https://community.livekit.io/t/m5stack-cores3-aw88298-es7210-shared-i2s-no-speaker-output-capture-works/561) — 2026-03-15
- [WSServerHandshakeError: 401](https://community.livekit.io/t/wsserverhandshakeerror-401/559) — 2026-03-14
- [Session not found, not ending, consumed my entire free trial](https://community.livekit.io/t/session-not-found-not-ending-consumed-my-entire-free-trial/557) — 2026-03-14
- [Auto Egress with dispatch rules](https://community.livekit.io/t/auto-egress-with-dispatch-rules/552) — 2026-03-13
- [Google TTS Plugin Timeout with Gemini Model & Chirp_3 Streaming Error (livekit-plugins-google==1.4.4)](https://community.livekit.io/t/google-tts-plugin-timeout-with-gemini-model-chirp-3-streaming-error-livekit-plugins-google-1-4-4/550) — 2026-03-13
- [Getting started](https://community.livekit.io/t/getting-started/544) — 2026-03-13
- [Issue In Audio Call](https://community.livekit.io/t/issue-in-audio-call/539) — 2026-03-12
- [Error from STT while in console --text](https://community.livekit.io/t/error-from-stt-while-in-console-text/531) — 2026-03-11
- [LiveKit inference for gemini 3.1 flash lite when?](https://community.livekit.io/t/livekit-inference-for-gemini-3-1-flash-lite-when/522) — 2026-03-11
- [Is Assembly Universal Pro 3 streaming HIPAA compliant in livekit inference?](https://community.livekit.io/t/is-assembly-universal-pro-3-streaming-hipaa-compliant-in-livekit-inference/521) — 2026-03-11
- [Choose Hosting Region in Cloud](https://community.livekit.io/t/choose-hosting-region-in-cloud/518) — 2026-03-10
- [Agent’s spoken transcript is not saved to chat context and conversation_item_added event does not fire when user disconnects during speech](https://community.livekit.io/t/agent-s-spoken-transcript-is-not-saved-to-chat-context-and-conversation-item-added-event-does-not-fire-when-user-disconnects-during-speech/513) — 2026-03-10
- [Back button is ending the livesession on screenshare, it should not. in react native app,](https://community.livekit.io/t/back-button-is-ending-the-livesession-on-screenshare-it-should-not-in-react-native-app/505) — 2026-03-09
- [Best practice to answer FAQs: speed & accuracy](https://community.livekit.io/t/best-practice-to-answer-faqs-speed-accuracy/504) — 2026-03-09
- [Agents in userdata: best practice](https://community.livekit.io/t/agents-in-userdata-best-practice/503) — 2026-03-09
- [Genesys SIP Routing to Livekit cloud or Genesys Audio connector to Livekit Cloud](https://community.livekit.io/t/genesys-sip-routing-to-livekit-cloud-or-genesys-audio-connector-to-livekit-cloud/502) — 2026-03-09
- [Inbound call telnyx service use but not working](https://community.livekit.io/t/inbound-call-telnyx-service-use-but-not-working/486) — 2026-03-06
- [Latency issue how to fix this?](https://community.livekit.io/t/latency-issue-how-to-fix-this/479) — 2026-03-06
- [WhatsApp Voice via Twilio SIP: Call rings but agent is not dispatched](https://community.livekit.io/t/whatsapp-voice-via-twilio-sip-call-rings-but-agent-is-not-dispatched/475) — 2026-03-06
- [What do the LiveKit Agent Session Limits entail?](https://community.livekit.io/t/what-do-the-livekit-agent-session-limits-entail/474) — 2026-03-05
- [Inbound calls not reaching the agent](https://community.livekit.io/t/inbound-calls-not-reaching-the-agent/460) — 2026-03-04
- [Elevenlabs STT Realtime and SIlero VAD | Proper setup](https://community.livekit.io/t/elevenlabs-stt-realtime-and-silero-vad-proper-setup/458) — 2026-03-04
- [Trying the new websockets responses api from openai](https://community.livekit.io/t/trying-the-new-websockets-responses-api-from-openai/457) — 2026-03-04
- [Project Statistics says Agents deployed on LiveKit Cloud (peak usage) is 0 in the past 7 days](https://community.livekit.io/t/project-statistics-says-agents-deployed-on-livekit-cloud-peak-usage-is-0-in-the-past-7-days/450) — 2026-03-04
- [Plugins-ai-coustics-python manylinux_aarch64 architecture support](https://community.livekit.io/t/plugins-ai-coustics-python-manylinux-aarch64-architecture-support/446) — 2026-03-04
- [Status of Unity Plugins & Contribution Alignment](https://community.livekit.io/t/status-of-unity-plugins-contribution-alignment/423) — 2026-03-01
- [How do i edit a livekit cloud project via lk cli?](https://community.livekit.io/t/how-do-i-edit-a-livekit-cloud-project-via-lk-cli/422) — 2026-03-01
- [Integrating LiveKit Agents with Microsoft Teams](https://community.livekit.io/t/integrating-livekit-agents-with-microsoft-teams/419) — 2026-02-28
- [Voice optimization audioProcessor for Livekit](https://community.livekit.io/t/voice-optimization-audioprocessor-for-livekit/398) — 2026-02-26
- [Down Stream data transfer cost estimation](https://community.livekit.io/t/down-stream-data-transfer-cost-estimation/396) — 2026-02-26
- [Websocket API for OpenAI](https://community.livekit.io/t/websocket-api-for-openai/374) — 2026-02-24
- [How can I change user_away_timeout during a session?](https://community.livekit.io/t/how-can-i-change-user-away-timeout-during-a-session/369) — 2026-02-23
- [Connection Issue: 401 Unauthorized / Invalid API Key on LiveKit Cloud](https://community.livekit.io/t/connection-issue-401-unauthorized-invalid-api-key-on-livekit-cloud/366) — 2026-02-23
- [Unity dev - CLI agent works, but not cloud agent](https://community.livekit.io/t/unity-dev-cli-agent-works-but-not-cloud-agent/365) — 2026-02-23
- [Livekit/livekit_composite repo seems outdates](https://community.livekit.io/t/livekit-livekit-composite-repo-seems-outdates/328) — 2026-02-19
- [Livekit Transcript timestamp is not correct ( even in their own observability it is showing wrong)](https://community.livekit.io/t/livekit-transcript-timestamp-is-not-correct-even-in-their-own-observability-it-is-showing-wrong/326) — 2026-02-19
- [LiveKit Cloud but we’ve lost track of the email](https://community.livekit.io/t/livekit-cloud-but-we-ve-lost-track-of-the-email/316) — 2026-02-17
- [Billing issues, correct plan not populating](https://community.livekit.io/t/billing-issues-correct-plan-not-populating/315) — 2026-02-17
- [Latency for each response 2.5-3.5sec in stopwach ,livekitagent](https://community.livekit.io/t/latency-for-each-response-2-5-3-5sec-in-stopwach-livekitagent/311) — 2026-02-17
- [Support for Indian Phone Numbers in LiveKit?](https://community.livekit.io/t/support-for-indian-phone-numbers-in-livekit/309) — 2026-02-17
- [SIP Inbound call - Agent is not joining and](https://community.livekit.io/t/sip-inbound-call-agent-is-not-joining-and/307) — 2026-02-16
- [Tips for getting a LiveKit-based solution certified for the Salesforce AppExchange?](https://community.livekit.io/t/tips-for-getting-a-livekit-based-solution-certified-for-the-salesforce-appexchange/304) — 2026-02-15
- [Difference between context.wait_for_playout and speechhandle.wait_for_playout](https://community.livekit.io/t/difference-between-context-wait-for-playout-and-speechhandle-wait-for-playout/296) — 2026-02-13
- [Executing a Task multiple times result in RuntimeError: no activity context found, the agent is not running](https://community.livekit.io/t/executing-a-task-multiple-times-result-in-runtimeerror-no-activity-context-found-the-agent-is-not-running/279) — 2026-02-10
- [Webhook Is not getting triggered even after correct configuration](https://community.livekit.io/t/webhook-is-not-getting-triggered-even-after-correct-configuration/272) — 2026-02-09
- [High end-to-end latency in LiveKit voice agent](https://community.livekit.io/t/high-end-to-end-latency-in-livekit-voice-agent/269) — 2026-02-09
- [Audio Glitches with Gemini Live Plugin](https://community.livekit.io/t/audio-glitches-with-gemini-live-plugin/260) — 2026-02-06
- [Understanding LiveKit Cloud architecture for per-user agents](https://community.livekit.io/t/understanding-livekit-cloud-architecture-for-per-user-agents/194) — 2026-01-21
- [Does self-hosting agents save costs compared to LiveKit Cloud?](https://community.livekit.io/t/does-self-hosting-agents-save-costs-compared-to-livekit-cloud/186) — 2026-01-21
- [Recommendations for Indian language TTS, STT, and LLM pipeline](https://community.livekit.io/t/recommendations-for-indian-language-tts-stt-and-llm-pipeline/173) — 2026-01-21
- [How to provide dynamic user-specific context to an agent](https://community.livekit.io/t/how-to-provide-dynamic-user-specific-context-to-an-agent/155) — 2026-01-21
- [How to build a chat agent with LiveKit](https://community.livekit.io/t/how-to-build-a-chat-agent-with-livekit/115) — 2026-01-21
- [Is there a UI/frontend for testing deployed agents?](https://community.livekit.io/t/is-there-a-ui-frontend-for-testing-deployed-agents/109) — 2026-01-21
- [Can I use my own API keys for external vendors (STT, TTS, LLM)?](https://community.livekit.io/t/can-i-use-my-own-api-keys-for-external-vendors-stt-tts-llm/97) — 2026-01-21
- [Where to find the LiveKit API key and secret](https://community.livekit.io/t/where-to-find-the-livekit-api-key-and-secret/92) — 2026-01-21
- [How to make a Node.js backend (not Agent) join a LiveKit room and handle audio](https://community.livekit.io/t/how-to-make-a-node-js-backend-not-agent-join-a-livekit-room-and-handle-audio/89) — 2026-01-21
- [Performance and differences between the Python and Node.js Agents SDKs](https://community.livekit.io/t/performance-and-differences-between-the-python-and-node-js-agents-sdks/74) — 2026-01-21
- [Frequently Asked Questions (FAQ)](https://community.livekit.io/t/frequently-asked-questions-faq/46) — 2025-12-30
- [Read this before posting a new question](https://community.livekit.io/t/read-this-before-posting-a-new-question/20) — 2025-12-22

## Agents (432)

- [Managed Cloud agents can't reach Google STT regional endpoints (*-speech.googleapis.com) - Chirp STT fails with ”Connection error”](https://community.livekit.io/t/managed-cloud-agents-cant-reach-google-stt-regional-endpoints-speech-googleapis-com-chirp-stt-fails-with-connection-error/1534) — 2026-06-30
- [After upgrading to 1.6.4 from 1.5.8, getting: ”job process exceeded memory limit, killing it”](https://community.livekit.io/t/after-upgrading-to-1-6-4-from-1-5-8-getting-job-process-exceeded-memory-limit-killing-it/1530) — 2026-06-30
- [Prewarm_fnc can't pre-warm async resources tied to the event loop (e.g. DB clients) — first job per process pays full connection cost](https://community.livekit.io/t/prewarm-fnc-cant-pre-warm-async-resources-tied-to-the-event-loop-e-g-db-clients-first-job-per-process-pays-full-connection-cost/1512) — 2026-06-26
- [Retrieving the realtime livekit agents transcript and saving to file](https://community.livekit.io/t/retrieving-the-realtime-livekit-agents-transcript-and-saving-to-file/1509) — 2026-06-25
- [Bugs and proposed enhancements related to the generate_reply timeout following up actions](https://community.livekit.io/t/bugs-and-proposed-enhancements-related-to-the-generate-reply-timeout-following-up-actions/1508) — 2026-06-25
- [Generate reply timeout for gemini-live-2.5-flash-native-audio realtime model](https://community.livekit.io/t/generate-reply-timeout-for-gemini-live-2-5-flash-native-audio-realtime-model/1505) — 2026-06-25
- [Livekit room isn't closing](https://community.livekit.io/t/livekit-room-isnt-closing/1485) — 2026-06-22
- [Calls are not reaching our livekit agents application from Livekit cloud](https://community.livekit.io/t/calls-are-not-reaching-our-livekit-agents-application-from-livekit-cloud/1482) — 2026-06-22
- [Support for Live STT Partial Transcripts in Python SDK for OpenAI models](https://community.livekit.io/t/support-for-live-stt-partial-transcripts-in-python-sdk-for-openai-models/1480) — 2026-06-22
- [LiveKit Agents for Node.js 1.5.0 is coming (Feedback Wanted.)](https://community.livekit.io/t/livekit-agents-for-node-js-1-5-0-is-coming-feedback-wanted/1475) — 2026-06-22
- [Agent fails to publish its audio track at join](https://community.livekit.io/t/agent-fails-to-publish-its-audio-track-at-join/1467) — 2026-06-20
- [”could not establish signal connection: Failed to fetch”,](https://community.livekit.io/t/could-not-establish-signal-connection-failed-to-fetch/1463) — 2026-06-19
- [Audio Intelligence for Voice Agents](https://community.livekit.io/t/audio-intelligence-for-voice-agents/1457) — 2026-06-18
- [CreateDispatch succeeds, but how do we know a worker actually received the job?](https://community.livekit.io/t/createdispatch-succeeds-but-how-do-we-know-a-worker-actually-received-the-job/1445) — 2026-06-17
- [Full-stack LiveKit voice agent starter: worker + token server + React, web and SIP on one worker (open source)](https://community.livekit.io/t/full-stack-livekit-voice-agent-starter-worker-token-server-react-web-and-sip-on-one-worker-open-source/1442) — 2026-06-16
- [Reliable Agent Room Join Issue – Dispatched Agents Not Joining Room Despite Successful Request](https://community.livekit.io/t/reliable-agent-room-join-issue-dispatched-agents-not-joining-room-despite-successful-request/1433) — 2026-06-15
- [How to submit a plugin?](https://community.livekit.io/t/how-to-submit-a-plugin/1424) — 2026-06-12
- [Python livekit-agents 1.6.0 released](https://community.livekit.io/t/python-livekit-agents-1-6-0-released/1419) — 2026-06-12
- [(Solved) Session never closes when participant disconnects before AgentTask](https://community.livekit.io/t/solved-session-never-closes-when-participant-disconnects-before-agenttask/1411) — 2026-06-11
- [Feature request: updateInstructions() support in Node.js Agents SDK](https://community.livekit.io/t/feature-request-updateinstructions-support-in-node-js-agents-sdk/1410) — 2026-06-11
- [Room Ended But Session is still Active](https://community.livekit.io/t/room-ended-but-session-is-still-active/1404) — 2026-06-11
- [Warning: stt end of speech received while vad is still in a speech segment, flushing vad](https://community.livekit.io/t/warning-stt-end-of-speech-received-while-vad-is-still-in-a-speech-segment-flushing-vad/1399) — 2026-06-10
- [Sharing: a New real-time 3DGS avatar plugin for LiveKit Agents by Spatius.ai (PR #5821)](https://community.livekit.io/t/sharing-a-new-real-time-3dgs-avatar-plugin-for-livekit-agents-by-spatius-ai-pr-5821/1398) — 2026-06-10
- [Answering Machine Detection latency concerns](https://community.livekit.io/t/answering-machine-detection-latency-concerns/1381) — 2026-06-08
- [ElevenLabs TTS playback speed catching up](https://community.livekit.io/t/elevenlabs-tts-playback-speed-catching-up/1376) — 2026-06-08
- [Does anyone else see increase in timeouts from gpt-realtime since around 1st June 2026, particularly 5th June 2026 onwards?](https://community.livekit.io/t/does-anyone-else-see-increase-in-timeouts-from-gpt-realtime-since-around-1st-june-2026-particularly-5th-june-2026-onwards/1367) — 2026-06-07
- [Livekit agents sending sms?](https://community.livekit.io/t/livekit-agents-sending-sms/1361) — 2026-06-06
- [Livekit SIP error](https://community.livekit.io/t/livekit-sip-error/1345) — 2026-06-05
- [ElevenLabs volume variation](https://community.livekit.io/t/elevenlabs-volume-variation/1340) — 2026-06-04
- [Enforce PR descriptions](https://community.livekit.io/t/enforce-pr-descriptions/1333) — 2026-06-03
- [Agent didn't followup after tool call](https://community.livekit.io/t/agent-didnt-followup-after-tool-call/1332) — 2026-06-03
- [Publisher connection times out mid-call, then retries ~11×/sec until hangup (agent goes silent)](https://community.livekit.io/t/publisher-connection-times-out-mid-call-then-retries-11x-sec-until-hangup-agent-goes-silent/1327) — 2026-06-03
- [Intermittent repeated agent responses with Azure OpenAI in LiveKit Agents](https://community.livekit.io/t/intermittent-repeated-agent-responses-with-azure-openai-in-livekit-agents/1319) — 2026-06-02
- [Building a Real-Time Vision Agent (Continuous Feedback)](https://community.livekit.io/t/building-a-real-time-vision-agent-continuous-feedback/1308) — 2026-06-01
- [Cerebras plugin](https://community.livekit.io/t/cerebras-plugin/1292) — 2026-05-31
- [Agent Node SDK features ETA - EndCallTool, WarmTransferTask, MCP server](https://community.livekit.io/t/agent-node-sdk-features-eta-endcalltool-warmtransfertask-mcp-server/1290) — 2026-05-30
- [Build plan deployed agent stays Sleeping after successful CreateDispatch](https://community.livekit.io/t/build-plan-deployed-agent-stays-sleeping-after-successful-createdispatch/1288) — 2026-05-30
- [Alerts for when job dispatches fail](https://community.livekit.io/t/alerts-for-when-job-dispatches-fail/1283) — 2026-05-29
- [Agent Builder agent joins room and publishes audio, but no speech in preview or production (dispatch Jakt)](https://community.livekit.io/t/agent-builder-agent-joins-room-and-publishes-audio-but-no-speech-in-preview-or-production-dispatch-jakt/1282) — 2026-05-29
- [Deepgram flux vs nova-3](https://community.livekit.io/t/deepgram-flux-vs-nova-3/1276) — 2026-05-29
- [High latency of Turn detection](https://community.livekit.io/t/high-latency-of-turn-detection/1275) — 2026-05-29
- [Outbound SIP trunk returns 408 / ”0 intermediate responses” on every call — is outbound enabled on our account?](https://community.livekit.io/t/outbound-sip-trunk-returns-408-0-intermediate-responses-on-every-call-is-outbound-enabled-on-our-account/1274) — 2026-05-28
- [What are the egress IPs for Agents Cloud compute containers?](https://community.livekit.io/t/what-are-the-egress-ips-for-agents-cloud-compute-containers/1272) — 2026-05-28
- [Livekit Inference thinking configuration for gemini 2.5 and 3.5 flash](https://community.livekit.io/t/livekit-inference-thinking-configuration-for-gemini-2-5-and-3-5-flash/1268) — 2026-05-28
- [Realtime model is not working properly](https://community.livekit.io/t/realtime-model-is-not-working-properly/1267) — 2026-05-27
- [Did something change recently?](https://community.livekit.io/t/did-something-change-recently/1266) — 2026-05-27
- [All Livekit Inference Gemini LLMs return ”Completion_tokens=0” and stop responding , This suddenly started happening today without any code change](https://community.livekit.io/t/all-livekit-inference-gemini-llms-return-completion-tokens-0-and-stop-responding-this-suddenly-started-happening-today-without-any-code-change/1264) — 2026-05-27
- [Support persistent custom model options in openai.LLM.withTelnyx() / OpenAI-compatible LLMs](https://community.livekit.io/t/support-persistent-custom-model-options-in-openai-llm-withtelnyx-openai-compatible-llms/1262) — 2026-05-27
- [Built-in laptop mic causing incorrect Whisper transcriptions](https://community.livekit.io/t/built-in-laptop-mic-causing-incorrect-whisper-transcriptions/1251) — 2026-05-26
- [Inconsistent agent state events for gemini realtime model](https://community.livekit.io/t/inconsistent-agent-state-events-for-gemini-realtime-model/1241) — 2026-05-26
- [Livekit Gemini-Live Pause issue](https://community.livekit.io/t/livekit-gemini-live-pause-issue/1228) — 2026-05-25
- [𝐌𝐮𝐥𝐭𝐢-𝐝𝐞𝐯𝐢𝐜𝐞 𝐜𝐨𝐨𝐫𝐝𝐢𝐧𝐚𝐭𝐢𝐨𝐧 𝐮𝐬𝐢𝐧𝐠 𝐋𝐢𝐯𝐞𝐊𝐢𝐭 𝐒𝐃𝐊𝐬: 𝐚𝐜𝐨𝐮𝐬𝐭𝐢𝐜 𝐬𝐲𝐧𝐜 𝐭𝐨 𝐯𝐢𝐝𝐞𝐨 𝐬𝐲𝐧𝐜](https://community.livekit.io/t/topic/1222) — 2026-05-25
- [Dynamic endpointing in agents-js SDK](https://community.livekit.io/t/dynamic-endpointing-in-agents-js-sdk/1210) — 2026-05-23
- [Agent observability and region pinning](https://community.livekit.io/t/agent-observability-and-region-pinning/1206) — 2026-05-22
- [Add support for Danish/Swedish in turn detection model](https://community.livekit.io/t/add-support-for-danish-swedish-in-turn-detection-model/1198) — 2026-05-21
- [Billing page refresh frequency + TTS costs during live event](https://community.livekit.io/t/billing-page-refresh-frequency-tts-costs-during-live-event/1197) — 2026-05-21
- [Dispatcher Issue](https://community.livekit.io/t/dispatcher-issue/1191) — 2026-05-21
- [Improving accuracy](https://community.livekit.io/t/improving-accuracy/1187) — 2026-05-20
- [Feature request: Gemini thinkingLevel=minimal for faster voice-agent TTFT](https://community.livekit.io/t/feature-request-gemini-thinkinglevel-minimal-for-faster-voice-agent-ttft/1185) — 2026-05-20
- [How to correlate per plugin metrics event with ChatMessage.metrics to calculate total latency](https://community.livekit.io/t/how-to-correlate-per-plugin-metrics-event-with-chatmessage-metrics-to-calculate-total-latency/1182) — 2026-05-20
- [Best practice for task-specific agents in LiveKit workflows?](https://community.livekit.io/t/best-practice-for-task-specific-agents-in-livekit-workflows/1181) — 2026-05-20
- [Request for review: Respeecher TTS plugin](https://community.livekit.io/t/request-for-review-respeecher-tts-plugin/1177) — 2026-05-20
- [Has Gemini 3.5 Flash been officially added to LiveKit Inference yet?](https://community.livekit.io/t/has-gemini-3-5-flash-been-officially-added-to-livekit-inference-yet/1176) — 2026-05-19
- [Issue: (Python SDK) Network switch causes ”stuck-in-reconnecting” and ”ghost room” states in livekit-agents](https://community.livekit.io/t/issue-python-sdk-network-switch-causes-stuck-in-reconnecting-and-ghost-room-states-in-livekit-agents/1174) — 2026-05-19
- [Released Agents JS @1.4.3](https://community.livekit.io/t/released-agents-js-1-4-3/1173) — 2026-05-19
- [Roadmap for adaptive interruption handling in self-hosted LiveKit workers?](https://community.livekit.io/t/roadmap-for-adaptive-interruption-handling-in-self-hosted-livekit-workers/1172) — 2026-05-19
- [Usage Summary for realtime xai and Ultravox models](https://community.livekit.io/t/usage-summary-for-realtime-xai-and-ultravox-models/1170) — 2026-05-19
- [Delay in agent response after initial greeting](https://community.livekit.io/t/delay-in-agent-response-after-initial-greeting/1169) — 2026-05-19
- [Missing call in agent session](https://community.livekit.io/t/missing-call-in-agent-session/1154) — 2026-05-18
- [We need to talk about pre-tool-call speech](https://community.livekit.io/t/we-need-to-talk-about-pre-tool-call-speech/1152) — 2026-05-18
- [Does cartesia/sonic-3-latest still route to Sonic 3.5, or Sonic 3?](https://community.livekit.io/t/does-cartesia-sonic-3-latest-still-route-to-sonic-3-5-or-sonic-3/1149) — 2026-05-17
- [Gemini3.1 live preview model drops mid conversation for odd seconds](https://community.livekit.io/t/gemini3-1-live-preview-model-drops-mid-conversation-for-odd-seconds/1142) — 2026-05-16
- [Intermittent 429 QuotaStatusNetworkError from LiveKit Cloud log export](https://community.livekit.io/t/intermittent-429-quotastatusnetworkerror-from-livekit-cloud-log-export/1140) — 2026-05-15
- [Gemini 3 Flash Preview via LiveKit Inference has much higher TTFT/jitter than direct Vertex in same Agents workflow](https://community.livekit.io/t/gemini-3-flash-preview-via-livekit-inference-has-much-higher-ttft-jitter-than-direct-vertex-in-same-agents-workflow/1138) — 2026-05-15
- [Missing Old LiveKit Playground UI](https://community.livekit.io/t/missing-old-livekit-playground-ui/1133) — 2026-05-15
- [inference.TTS doesn't work with FallbackAdapter in node sdk](https://community.livekit.io/t/inference-tts-doesnt-work-with-fallbackadapter-in-node-sdk/1126) — 2026-05-14
- [Anthropic Claude models in the agent builder](https://community.livekit.io/t/anthropic-claude-models-in-the-agent-builder/1119) — 2026-05-14
- [Grok RealtimeModel not accepting custom xAI voice clone (only default voices work)](https://community.livekit.io/t/grok-realtimemodel-not-accepting-custom-xai-voice-clone-only-default-voices-work/1117) — 2026-05-14
- [Docker image size too large with Python LiveKit Agents (~2.7 GB) - is this normal for production?](https://community.livekit.io/t/docker-image-size-too-large-with-python-livekit-agents-2-7-gb-is-this-normal-for-production/1114) — 2026-05-14
- [Attaching custom attributes to user transcription stream — public hook in RoomOutputOptions?](https://community.livekit.io/t/attaching-custom-attributes-to-user-transcription-stream-public-hook-in-roomoutputoptions/1111) — 2026-05-13
- [AssignmentTimeoutError when accepting job requests](https://community.livekit.io/t/assignmenttimeouterror-when-accepting-job-requests/1110) — 2026-05-13
- [How to calculate the pricing for gemini tts plugin](https://community.livekit.io/t/how-to-calculate-the-pricing-for-gemini-tts-plugin/1109) — 2026-05-13
- [Agent Dispatch errors - Free Plan | Automatic Agent Dispatch](https://community.livekit.io/t/agent-dispatch-errors-free-plan-automatic-agent-dispatch/1106) — 2026-05-13
- [Adaptive Turn Handling Ignores Single Word Answers](https://community.livekit.io/t/adaptive-turn-handling-ignores-single-word-answers/1104) — 2026-05-13
- [Publisher pc state failed and the agent resumed frequently at around the 8-minute mark](https://community.livekit.io/t/publisher-pc-state-failed-and-the-agent-resumed-frequently-at-around-the-8-minute-mark/1093) — 2026-05-12
- [How does Azure gpt vs Open AI gpt has such latency difference?](https://community.livekit.io/t/how-does-azure-gpt-vs-open-ai-gpt-has-such-latency-difference/1088) — 2026-05-12
- [Why is GPT-5.4 pricing via LiveKit Inference about 2x OpenAI direct?](https://community.livekit.io/t/why-is-gpt-5-4-pricing-via-livekit-inference-about-2x-openai-direct/1087) — 2026-05-12
- [Livekit agent framework agnostic](https://community.livekit.io/t/livekit-agent-framework-agnostic/1075) — 2026-05-11
- [Is there a Node.js version or roadmap for WarmTransferTask?](https://community.livekit.io/t/is-there-a-node-js-version-or-roadmap-for-warmtransfertask/1074) — 2026-05-11
- [Gpt-realtime-2 + LiveKit: VAD does not work well](https://community.livekit.io/t/gpt-realtime-2-livekit-vad-does-not-work-well/1072) — 2026-05-10
- [Egress completed automatically after 163s](https://community.livekit.io/t/egress-completed-automatically-after-163s/1068) — 2026-05-09
- [Question: bounded evidence receipts from FunctionToolsExecutedEvent](https://community.livekit.io/t/question-bounded-evidence-receipts-from-functiontoolsexecutedevent/1066) — 2026-05-09
- [Cloud Agent Concurrency Failover](https://community.livekit.io/t/cloud-agent-concurrency-failover/1064) — 2026-05-08
- [Gpt-Realtime 2: Experience so far?](https://community.livekit.io/t/gpt-realtime-2-experience-so-far/1063) — 2026-05-08
- [Gpt-realtime-2 set reasoning_effort to none or very low](https://community.livekit.io/t/gpt-realtime-2-set-reasoning-effort-to-none-or-very-low/1062) — 2026-05-08
- [Cannot write to closing transport (Stale Connection Pool)](https://community.livekit.io/t/cannot-write-to-closing-transport-stale-connection-pool/1058) — 2026-05-08
- [GPT‑Realtime‑2 support](https://community.livekit.io/t/gpt-realtime-2-support/1055) — 2026-05-07
- [LiveKit Agent took 4 minutes to join once livekitSdk.dispatchAgentToRoom() was called](https://community.livekit.io/t/livekit-agent-took-4-minutes-to-join-once-livekitsdk-dispatchagenttoroom-was-called/1054) — 2026-05-07
- [Livekit integration with langfuse](https://community.livekit.io/t/livekit-integration-with-langfuse/1051) — 2026-05-07
- [Traffic routed to stale/ghost agent after deletion (not visible in CLI or dashboard) (US-East Outage Context)](https://community.livekit.io/t/traffic-routed-to-stale-ghost-agent-after-deletion-not-visible-in-cli-or-dashboard-us-east-outage-context/1047) — 2026-05-06
- [Agent Deployments Stuck in Scheduling](https://community.livekit.io/t/agent-deployments-stuck-in-scheduling/1039) — 2026-05-06
- [Managed deployment agent status](https://community.livekit.io/t/managed-deployment-agent-status/1030) — 2026-05-05
- [Optimizing Voice Agent Latency, Tool Calling Delays, and Audio Quality Issues with GPT-4o Mini, Sarvam V3 TTS, Deepgram Nova 3 STT, and LiveKit](https://community.livekit.io/t/optimizing-voice-agent-latency-tool-calling-delays-and-audio-quality-issues-with-gpt-4o-mini-sarvam-v3-tts-deepgram-nova-3-stt-and-livekit/1024) — 2026-05-04
- [Gemini Live speaks twice after a tool call](https://community.livekit.io/t/gemini-live-speaks-twice-after-a-tool-call/1022) — 2026-05-04
- [Livekit Inference Assembly AI u3-rt-pro - TLS: failed to verify certificate](https://community.livekit.io/t/livekit-inference-assembly-ai-u3-rt-pro-tls-failed-to-verify-certificate/1016) — 2026-05-04
- [How to controll use of RAM](https://community.livekit.io/t/how-to-controll-use-of-ram/1009) — 2026-05-02
- [ElevenLabs eleven_v3 with LiveKit Agents JS plugin fails with WebSocket 403](https://community.livekit.io/t/elevenlabs-eleven-v3-with-livekit-agents-js-plugin-fails-with-websocket-403/1007) — 2026-05-02
- [Manual or server VAD for Node.js Elevenlabs STT plugin?](https://community.livekit.io/t/manual-or-server-vad-for-node-js-elevenlabs-stt-plugin/1006) — 2026-05-02
- [JobRequest.reject(terminate=False) on Cloud Agents — does it reroute or fail?](https://community.livekit.io/t/jobrequest-reject-terminate-false-on-cloud-agents-does-it-reroute-or-fail/1005) — 2026-05-02
- [Elevenlabs STT plugin for Node.js](https://community.livekit.io/t/elevenlabs-stt-plugin-for-node-js/998) — 2026-04-30
- [Agent idle pending issues](https://community.livekit.io/t/agent-idle-pending-issues/997) — 2026-04-30
- [Support for Cartesia sonic-3-latest model](https://community.livekit.io/t/support-for-cartesia-sonic-3-latest-model/994) — 2026-04-30
- [Elevenlabs v3 tts inference erroring without language_code](https://community.livekit.io/t/elevenlabs-v3-tts-inference-erroring-without-language-code/992) — 2026-04-30
- [Upgrading python livekit-agents to 1.5.6 causing memory issues](https://community.livekit.io/t/upgrading-python-livekit-agents-to-1-5-6-causing-memory-issues/976) — 2026-04-28
- [“whatsapp call is not enabled for this project”](https://community.livekit.io/t/whatsapp-call-is-not-enabled-for-this-project/972) — 2026-04-28
- [How do you guys pick the right voice?](https://community.livekit.io/t/how-do-you-guys-pick-the-right-voice/967) — 2026-04-28
- [Any missing plugins on Livekit right now?](https://community.livekit.io/t/any-missing-plugins-on-livekit-right-now/960) — 2026-04-27
- [High Turn Detection Latency Issue](https://community.livekit.io/t/high-turn-detection-latency-issue/958) — 2026-04-27
- [Clarity on Avatar plugin flows - text/audio streaming](https://community.livekit.io/t/clarity-on-avatar-plugin-flows-text-audio-streaming/954) — 2026-04-27
- [Pre-warm the audio connection for iOS?](https://community.livekit.io/t/pre-warm-the-audio-connection-for-ios/953) — 2026-04-27
- [LiveKit Agents Runway plugin - Unrecognized key: /”livekit/”](https://community.livekit.io/t/livekit-agents-runway-plugin-unrecognized-key-livekit/948) — 2026-04-24
- [Static IP for South America (Brazil) with LiveAvatar (HeyGen)](https://community.livekit.io/t/static-ip-for-south-america-brazil-with-liveavatar-heygen/943) — 2026-04-23
- [Whats your current go-to LLM model?](https://community.livekit.io/t/whats-your-current-go-to-llm-model/941) — 2026-04-23
- [Available Elevenlabs Voices On Inference](https://community.livekit.io/t/available-elevenlabs-voices-on-inference/932) — 2026-04-22
- [How to update the MCP bearer authorization](https://community.livekit.io/t/how-to-update-the-mcp-bearer-authorization/927) — 2026-04-22
- [Released Agents (Python) 1.5.5 and Agents (NodeJS) 1.2.8](https://community.livekit.io/t/released-agents-python-1-5-5-and-agents-nodejs-1-2-8/918) — 2026-04-21
- [Channel closed error understanding](https://community.livekit.io/t/channel-closed-error-understanding/916) — 2026-04-20
- [xAI Grok Voice not working in Preview + 2 more issues](https://community.livekit.io/t/xai-grok-voice-not-working-in-preview-2-more-issues/914) — 2026-04-20
- [Review process for new integrations in livekit/agents](https://community.livekit.io/t/review-process-for-new-integrations-in-livekit-agents/910) — 2026-04-20
- [Turn Detection Latency Issue](https://community.livekit.io/t/turn-detection-latency-issue/909) — 2026-04-20
- [Response.prompt_cache_retention Input should be ‘in-memory’ or ‘24h](https://community.livekit.io/t/response-prompt-cache-retention-input-should-be-in-memory-or-24h/908) — 2026-04-20
- [Error: 429 Too Many Requests on agent-gateway.livekit.cloud](https://community.livekit.io/t/error-429-too-many-requests-on-agent-gateway-livekit-cloud/904) — 2026-04-19
- [Gpt-realtime-1.5 leaks audio control tokens (<|audio_text|>, <|caption_quality_N|>) into text stream when run with modalities=(”text”)](https://community.livekit.io/t/gpt-realtime-1-5-leaks-audio-control-tokens-audio-text-caption-quality-n-into-text-stream-when-run-with-modalities-text/902) — 2026-04-18
- [Tool timeout issue](https://community.livekit.io/t/tool-timeout-issue/896) — 2026-04-17
- [Livekit Cloud Log](https://community.livekit.io/t/livekit-cloud-log/894) — 2026-04-16
- [Released livekit-agents@1.5.3 and livekit-agents@1.5.4](https://community.livekit.io/t/released-livekit-agents-1-5-3-and-livekit-agents-1-5-4/887) — 2026-04-16
- [Preview of agent in agent building is not working](https://community.livekit.io/t/preview-of-agent-in-agent-building-is-not-working/883) — 2026-04-16
- [AI Coustics model usage very high](https://community.livekit.io/t/ai-coustics-model-usage-very-high/878) — 2026-04-15
- [Help with issue: livekit dropping screen share frames in prod only](https://community.livekit.io/t/help-with-issue-livekit-dropping-screen-share-frames-in-prod-only/867) — 2026-04-14
- [Bad LiveKit Inference ttft for gpt-4.1](https://community.livekit.io/t/bad-livekit-inference-ttft-for-gpt-4-1/866) — 2026-04-14
- [The agent is dispatched only for the first participant](https://community.livekit.io/t/the-agent-is-dispatched-only-for-the-first-participant/864) — 2026-04-14
- [Investigating the HTTP 200 unauthorized response issue](https://community.livekit.io/t/investigating-the-http-200-unauthorized-response-issue/858) — 2026-04-14
- [Stuck at 2.3s p50 after weeks of tuning - is the livekit.io homepage demo a classic chain or realtime speech?](https://community.livekit.io/t/stuck-at-2-3s-p50-after-weeks-of-tuning-is-the-livekit-io-homepage-demo-a-classic-chain-or-realtime-speech/851) — 2026-04-13
- [Chat-only mode in LiveKit without consuming agent minutes?](https://community.livekit.io/t/chat-only-mode-in-livekit-without-consuming-agent-minutes/849) — 2026-04-13
- [Only one 'AgentSession' can be the primary at a time Error when starting a session](https://community.livekit.io/t/only-one-agentsession-can-be-the-primary-at-a-time-error-when-starting-a-session/846) — 2026-04-13
- [Livekit gemini Plugin , Gemini 3.1 flash live preview is not supported on 1.5.2](https://community.livekit.io/t/livekit-gemini-plugin-gemini-3-1-flash-live-preview-is-not-supported-on-1-5-2/844) — 2026-04-12
- [Handling response latency: Playing fallback/filler audio if no response arrives within a timeout](https://community.livekit.io/t/handling-response-latency-playing-fallback-filler-audio-if-no-response-arrives-within-a-timeout/839) — 2026-04-11
- [Build plan shows 5 concurrent agent sessions, but only 1 live call works at a time](https://community.livekit.io/t/build-plan-shows-5-concurrent-agent-sessions-but-only-1-live-call-works-at-a-time/837) — 2026-04-11
- [Cross-process agent handoff — any plans for first-class support?](https://community.livekit.io/t/cross-process-agent-handoff-any-plans-for-first-class-support/835) — 2026-04-10
- [Unable to use ai_coustics in agents](https://community.livekit.io/t/unable-to-use-ai-coustics-in-agents/832) — 2026-04-10
- [Improvement to graceful teardown](https://community.livekit.io/t/improvement-to-graceful-teardown/831) — 2026-04-10
- [Elevenlabs Voice ID outside of Defaults - Livekit Inference](https://community.livekit.io/t/elevenlabs-voice-id-outside-of-defaults-livekit-inference/829) — 2026-04-10
- [429 too many requests when trying to preemptively connect user audio](https://community.livekit.io/t/429-too-many-requests-when-trying-to-preemptively-connect-user-audio/827) — 2026-04-10
- [Python livekit-agents@1.5.2 released](https://community.livekit.io/t/python-livekit-agents-1-5-2-released/819) — 2026-04-08
- [Livekit Agent Dispatch issue, hosted on livekit cloud](https://community.livekit.io/t/livekit-agent-dispatch-issue-hosted-on-livekit-cloud/811) — 2026-04-08
- [Livekit supports ElevenLabs TTS models, but why not Eleven v3](https://community.livekit.io/t/livekit-supports-elevenlabs-tts-models-but-why-not-eleven-v3/810) — 2026-04-08
- [Adding skills to agents](https://community.livekit.io/t/adding-skills-to-agents/795) — 2026-04-08
- [One-Way Audio — agent audio not reaching caller when SIP participant joins room before agent](https://community.livekit.io/t/one-way-audio-agent-audio-not-reaching-caller-when-sip-participant-joins-room-before-agent/794) — 2026-04-08
- [500 Internal Server Error uploading session report to /observability/recordings/v0](https://community.livekit.io/t/500-internal-server-error-uploading-session-report-to-observability-recordings-v0/788) — 2026-04-06
- [Audio visualizers vs. avatars — how are you thinking about visual presence for your agents?](https://community.livekit.io/t/audio-visualizers-vs-avatars-how-are-you-thinking-about-visual-presence-for-your-agents/787) — 2026-04-06
- [Tool Invocation Starts Event](https://community.livekit.io/t/tool-invocation-starts-event/784) — 2026-04-06
- [Agent-to-agent audio](https://community.livekit.io/t/agent-to-agent-audio/780) — 2026-04-06
- [Adaptive interruption disabled due to unrecoverable error, falling back to VAD-based interruption](https://community.livekit.io/t/adaptive-interruption-disabled-due-to-unrecoverable-error-falling-back-to-vad-based-interruption/775) — 2026-04-05
- [Does adaptive interruption handling add latency anywhere in the voice agent pipeline?](https://community.livekit.io/t/does-adaptive-interruption-handling-add-latency-anywhere-in-the-voice-agent-pipeline/774) — 2026-04-04
- [Agent Session Connection Timeout without Redispatch 4/3 3 PM PST](https://community.livekit.io/t/agent-session-connection-timeout-without-redispatch-4-3-3-pm-pst/769) — 2026-04-04
- [Does LiveKit inference have Deepgram flux multilingual?](https://community.livekit.io/t/does-livekit-inference-have-deepgram-flux-multilingual/765) — 2026-04-03
- [Still dealing with double and triple logging, anyone else have this problem?](https://community.livekit.io/t/still-dealing-with-double-and-triple-logging-anyone-else-have-this-problem/764) — 2026-04-02
- [Multi environment deployment best practice](https://community.livekit.io/t/multi-environment-deployment-best-practice/755) — 2026-04-02
- [Migrating from plugins to livekit inference](https://community.livekit.io/t/migrating-from-plugins-to-livekit-inference/751) — 2026-04-01
- [(sarvam tts) output_audio_codec=”wav” causes ”Invalid WAV file: missing RIFF/WAVE” error](https://community.livekit.io/t/sarvam-tts-output-audio-codec-wav-causes-invalid-wav-file-missing-riff-wave-error/749) — 2026-04-01
- [Unexpected Audio Degradation After Enabling BVC Noise Cancellation in LiveKit Voice Agent](https://community.livekit.io/t/unexpected-audio-degradation-after-enabling-bvc-noise-cancellation-in-livekit-voice-agent/745) — 2026-04-01
- [Intermitten connection issues with connecting to Livekit](https://community.livekit.io/t/intermitten-connection-issues-with-connecting-to-livekit/742) — 2026-04-01
- [Urgent: Candidates Unable to Connect to Room — Signal Connection Error](https://community.livekit.io/t/urgent-candidates-unable-to-connect-to-room-signal-connection-error/740) — 2026-04-01
- [User turn not recognized by the agent](https://community.livekit.io/t/user-turn-not-recognized-by-the-agent/735) — 2026-03-31
- ['metrics_collected' is deprecated — what's the right replacement?](https://community.livekit.io/t/metrics-collected-is-deprecated-whats-the-right-replacement/734) — 2026-03-31
- [Livekit agents worker on cloud getting shutdown automatically](https://community.livekit.io/t/livekit-agents-worker-on-cloud-getting-shutdown-automatically/732) — 2026-03-31
- [How to configure serviceTier for AWS Bedrock voice agents](https://community.livekit.io/t/how-to-configure-servicetier-for-aws-bedrock-voice-agents/722) — 2026-03-30
- [Chat History Not Recording All Turns — Only First Agent Transcript Captured](https://community.livekit.io/t/chat-history-not-recording-all-turns-only-first-agent-transcript-captured/721) — 2026-03-30
- [S3 Egress Error — ”Start Signal Not Received”](https://community.livekit.io/t/s3-egress-error-start-signal-not-received/719) — 2026-03-30
- [Validating architecture: Single agent, dynamic per-call config via external API (multi-tenant PSTN)](https://community.livekit.io/t/validating-architecture-single-agent-dynamic-per-call-config-via-external-api-multi-tenant-pstn/718) — 2026-03-30
- [Livekit based vapi or retell oss alternative](https://community.livekit.io/t/livekit-based-vapi-or-retell-oss-alternative/716) — 2026-03-29
- [Voxtral TTS API 1,230ms TTFB in real-time voice agent pipeline](https://community.livekit.io/t/voxtral-tts-api-1-230ms-ttfb-in-real-time-voice-agent-pipeline/715) — 2026-03-29
- [Voxtral TTS by Mistral](https://community.livekit.io/t/voxtral-tts-by-mistral/713) — 2026-03-29
- [Long-running Voice Session incur Significant Costs for WebSocket-based STT model](https://community.livekit.io/t/long-running-voice-session-incur-significant-costs-for-websocket-based-stt-model/710) — 2026-03-29
- [Sarvam TTS - error decoding WAV audio](https://community.livekit.io/t/sarvam-tts-error-decoding-wav-audio/709) — 2026-03-29
- [Generate_reply Times Out on First Dialogue with gemini-3.1-flash-live-preview](https://community.livekit.io/t/generate-reply-times-out-on-first-dialogue-with-gemini-3-1-flash-live-preview/708) — 2026-03-28
- [Gemini 3.1 Flash live generateReply() timeout issue nodejs](https://community.livekit.io/t/gemini-3-1-flash-live-generatereply-timeout-issue-nodejs/702) — 2026-03-28
- [Interruption inference timeout: ', status_code=408](https://community.livekit.io/t/interruption-inference-timeout-status-code-408/700) — 2026-03-27
- [Pinch Translation](https://community.livekit.io/t/pinch-translation/696) — 2026-03-27
- [Is TaskGroup the right primitive for Survey Agents with 20+ Questions](https://community.livekit.io/t/is-taskgroup-the-right-primitive-for-survey-agents-with-20-questions/694) — 2026-03-27
- [Agent Room not deleted](https://community.livekit.io/t/agent-room-not-deleted/693) — 2026-03-26
- [Agent Voice Not Playing on Mobile Devices — Critical Production Issue](https://community.livekit.io/t/agent-voice-not-playing-on-mobile-devices-critical-production-issue/688) — 2026-03-26
- [Failed to synthesize speech: Invalid response status (429 Too Many Requests)](https://community.livekit.io/t/failed-to-synthesize-speech-invalid-response-status-429-too-many-requests/687) — 2026-03-26
- [Agent didn't join even after 2 mins of room creation and no errors](https://community.livekit.io/t/agent-didnt-join-even-after-2-mins-of-room-creation-and-no-errors/686) — 2026-03-26
- [User_state stuck in speaking during the agent handoff](https://community.livekit.io/t/user-state-stuck-in-speaking-during-the-agent-handoff/685) — 2026-03-26
- [Inference STT WebSocket fails (APIConnectionError) while room connection works](https://community.livekit.io/t/inference-stt-websocket-fails-apiconnectionerror-while-room-connection-works/683) — 2026-03-26
- [Multi-agent turn coordination — text-stream ack protocol vs native SDK approach?](https://community.livekit.io/t/multi-agent-turn-coordination-text-stream-ack-protocol-vs-native-sdk-approach/674) — 2026-03-25
- [No Xai Grok Voice Options in Realtime Model](https://community.livekit.io/t/no-xai-grok-voice-options-in-realtime-model/673) — 2026-03-25
- [Does Gemini Vision work with H264?](https://community.livekit.io/t/does-gemini-vision-work-with-h264/672) — 2026-03-25
- [Add language for turn detection](https://community.livekit.io/t/add-language-for-turn-detection/668) — 2026-03-25
- [Live test call not starting](https://community.livekit.io/t/live-test-call-not-starting/664) — 2026-03-24
- [Recording Duration Exceeds Room Duration + onDisconnected Not Triggered on Room End](https://community.livekit.io/t/recording-duration-exceeds-room-duration-ondisconnected-not-triggered-on-room-end/655) — 2026-03-24
- [Not able to download room session](https://community.livekit.io/t/not-able-to-download-room-session/653) — 2026-03-24
- [Released livekit-agents@1.5.1](https://community.livekit.io/t/released-livekit-agents-1-5-1/649) — 2026-03-23
- [Agent Disconnects After session.shutdown(drain=True) but User Remains Stuck in Room — Production Issue](https://community.livekit.io/t/agent-disconnects-after-session-shutdown-drain-true-but-user-remains-stuck-in-room-production-issue/647) — 2026-03-23
- [Cannot run the AI agent](https://community.livekit.io/t/cannot-run-the-ai-agent/639) — 2026-03-21
- [Handle per-call logging/observability with LiveKit Agents on Cloud](https://community.livekit.io/t/handle-per-call-logging-observability-with-livekit-agents-on-cloud/636) — 2026-03-20
- [Increased latency because of ivr_detection set to True](https://community.livekit.io/t/increased-latency-because-of-ivr-detection-set-to-true/635) — 2026-03-20
- [LiveKit Agent not triggering backend call hang-up on keyword detection](https://community.livekit.io/t/livekit-agent-not-triggering-backend-call-hang-up-on-keyword-detection/631) — 2026-03-20
- [Urgent: on_session_end Not Called for Some Sessions — Need Reliable Transcription Delivery in Production](https://community.livekit.io/t/urgent-on-session-end-not-called-for-some-sessions-need-reliable-transcription-delivery-in-production/630) — 2026-03-20
- [Critcal : AI Agent Unexpectedly Ending Interview Mid-Session](https://community.livekit.io/t/critcal-ai-agent-unexpectedly-ending-interview-mid-session/626) — 2026-03-20
- [Reliably persisting transcriptions under moderate concurrency](https://community.livekit.io/t/reliably-persisting-transcriptions-under-moderate-concurrency/624) — 2026-03-20
- [How to supply keywords to Deepgram Nova 3 STT with LiveKit Inference?](https://community.livekit.io/t/how-to-supply-keywords-to-deepgram-nova-3-stt-with-livekit-inference/623) — 2026-03-19
- [I extended the Agent Builder with Workflows, Tasks & more](https://community.livekit.io/t/i-extended-the-agent-builder-with-workflows-tasks-more/622) — 2026-03-19
- [I built a Python framework to run multiple LiveKit voice agents in one worker process](https://community.livekit.io/t/i-built-a-python-framework-to-run-multiple-livekit-voice-agents-in-one-worker-process/621) — 2026-03-19
- [Python Agents 1.5.0 Released](https://community.livekit.io/t/python-agents-1-5-0-released/619) — 2026-03-19
- [Setting elevenlabs TTS voice speed with LiveKit Inference](https://community.livekit.io/t/setting-elevenlabs-tts-voice-speed-with-livekit-inference/618) — 2026-03-19
- [Urgent: on_session_end callback not triggering in Agents SDK](https://community.livekit.io/t/urgent-on-session-end-callback-not-triggering-in-agents-sdk/616) — 2026-03-19
- [Agent handoff fails when tool returning Agent runs in parallel with reply-producing tool (pipeline path)](https://community.livekit.io/t/agent-handoff-fails-when-tool-returning-agent-runs-in-parallel-with-reply-producing-tool-pipeline-path/615) — 2026-03-18
- [Runway Characters avatar support](https://community.livekit.io/t/runway-characters-avatar-support/613) — 2026-03-18
- [Agent Pricing End to End](https://community.livekit.io/t/agent-pricing-end-to-end/608) — 2026-03-18
- [Room ended reason](https://community.livekit.io/t/room-ended-reason/607) — 2026-03-18
- [Several names for an agent / synonym](https://community.livekit.io/t/several-names-for-an-agent-synonym/606) — 2026-03-18
- [Agent Not Connecting to Local Environment — Unable to Debug](https://community.livekit.io/t/agent-not-connecting-to-local-environment-unable-to-debug/603) — 2026-03-18
- [xAI TTS returns 403 and no audio frames pushed; plugin appears to use different WebSocket endpoint than xAI docs](https://community.livekit.io/t/xai-tts-returns-403-and-no-audio-frames-pushed-plugin-appears-to-use-different-websocket-endpoint-than-xai-docs/602) — 2026-03-17
- [Python livekit-agents@1.4.6 Released](https://community.livekit.io/t/python-livekit-agents-1-4-6-released/600) — 2026-03-17
- [Lk command crashes creating a new app](https://community.livekit.io/t/lk-command-crashes-creating-a-new-app/596) — 2026-03-16
- [Why livekit is not supporting Bengali Language for ElevenLabs Scribe v2 Realtime model](https://community.livekit.io/t/why-livekit-is-not-supporting-bengali-language-for-elevenlabs-scribe-v2-realtime-model/595) — 2026-03-16
- [Facing errors while calling update_chat_ctx when using azure open ai realtime llm](https://community.livekit.io/t/facing-errors-while-calling-update-chat-ctx-when-using-azure-open-ai-realtime-llm/594) — 2026-03-16
- [Getting failed to initialize audio filter](https://community.livekit.io/t/getting-failed-to-initialize-audio-filter/591) — 2026-03-16
- [Agent insight export](https://community.livekit.io/t/agent-insight-export/585) — 2026-03-16
- [Multiple Unstable Connection Errors During Active Sessions — Losing Confidence in Production Reliability](https://community.livekit.io/t/multiple-unstable-connection-errors-during-active-sessions-losing-confidence-in-production-reliability/582) — 2026-03-16
- [Voice_agent audio is silent on M5Stack CoreS3 (AW88298 + ES7210)](https://community.livekit.io/t/voice-agent-audio-is-silent-on-m5stack-cores3-aw88298-es7210/579) — 2026-03-16
- [Agents for Local dev](https://community.livekit.io/t/agents-for-local-dev/574) — 2026-03-16
- [Six Agent Related Questions](https://community.livekit.io/t/six-agent-related-questions/564) — 2026-03-15
- [Agent SDK states](https://community.livekit.io/t/agent-sdk-states/563) — 2026-03-15
- [How can I find agent deployment preview link?](https://community.livekit.io/t/how-can-i-find-agent-deployment-preview-link/562) — 2026-03-15
- [Prewarm Turn Detection Model](https://community.livekit.io/t/prewarm-turn-detection-model/560) — 2026-03-14
- [Assistance Needed:](https://community.livekit.io/t/assistance-needed/549) — 2026-03-13
- [Session Closed during running interview](https://community.livekit.io/t/session-closed-during-running-interview/548) — 2026-03-13
- [Unexpected error in triggering Server RPC functions from Frontend](https://community.livekit.io/t/unexpected-error-in-triggering-server-rpc-functions-from-frontend/547) — 2026-03-13
- [Loading Chat Context for Realtime Model](https://community.livekit.io/t/loading-chat-context-for-realtime-model/545) — 2026-03-13
- [Mismatch in usage calculations](https://community.livekit.io/t/mismatch-in-usage-calculations/542) — 2026-03-12
- [Lowest latency STT/TTS/LLM stack for German - what's your experience?](https://community.livekit.io/t/lowest-latency-stt-tts-llm-stack-for-german-whats-your-experience/541) — 2026-03-12
- [Is a parallel Observer Layer overkill for real-time guardrails in a telephony agent](https://community.livekit.io/t/is-a-parallel-observer-layer-overkill-for-real-time-guardrails-in-a-telephony-agent/536) — 2026-03-12
- [Livekit incorrectly Double and Triple-logging entries](https://community.livekit.io/t/livekit-incorrectly-double-and-triple-logging-entries/535) — 2026-03-12
- [Request for review: PR for LiveKit Agents](https://community.livekit.io/t/request-for-review-pr-for-livekit-agents/532) — 2026-03-11
- [Python livekit-agents@1.4.5 Released](https://community.livekit.io/t/python-livekit-agents-1-4-5-released/527) — 2026-03-11
- [Hard time debugging complex AgentSession flows (tools, handoffs, transcripts)](https://community.livekit.io/t/hard-time-debugging-complex-agentsession-flows-tools-handoffs-transcripts/524) — 2026-03-11
- [Agents-playground not displaying text in Speech to text agent](https://community.livekit.io/t/agents-playground-not-displaying-text-in-speech-to-text-agent/523) — 2026-03-11
- [Agent stuck - Speech not triggering any traces for user turn](https://community.livekit.io/t/agent-stuck-speech-not-triggering-any-traces-for-user-turn/517) — 2026-03-10
- [MCP tool definitions bloating prompt tokens and increasing latency in voice agents — how to handle this?](https://community.livekit.io/t/mcp-tool-definitions-bloating-prompt-tokens-and-increasing-latency-in-voice-agents-how-to-handle-this/516) — 2026-03-10
- [Agent Dispatch Errors](https://community.livekit.io/t/agent-dispatch-errors/509) — 2026-03-10
- [Migrating from self hosted livekit-agent to livekit cloud](https://community.livekit.io/t/migrating-from-self-hosted-livekit-agent-to-livekit-cloud/507) — 2026-03-10
- [anthropic.LLM() default model and other models simply not working](https://community.livekit.io/t/anthropic-llm-default-model-and-other-models-simply-not-working/496) — 2026-03-08
- [Daytona Conference (Monday, March 9th) or SF Office Visit](https://community.livekit.io/t/daytona-conference-monday-march-9th-or-sf-office-visit/495) — 2026-03-08
- [Best STT Alternative to OpenAI whisper-1 for Japanese in LiveKit](https://community.livekit.io/t/best-stt-alternative-to-openai-whisper-1-for-japanese-in-livekit/492) — 2026-03-07
- [ElevenLabs STT & TTS plugin both broken on LiveKit Agents v1.4.4](https://community.livekit.io/t/elevenlabs-stt-tts-plugin-both-broken-on-livekit-agents-v1-4-4/488) — 2026-03-06
- [Real-time translator agent: architecture feedback wanted](https://community.livekit.io/t/real-time-translator-agent-architecture-feedback-wanted/485) — 2026-03-06
- [Inbound calls ending by livekit when there is no activity from user for around 20-30 sec](https://community.livekit.io/t/inbound-calls-ending-by-livekit-when-there-is-no-activity-from-user-for-around-20-30-sec/482) — 2026-03-06
- [LLM failed and Logs are not reflected in Insight](https://community.livekit.io/t/llm-failed-and-logs-are-not-reflected-in-insight/480) — 2026-03-06
- [Session gets closed unexpectedly](https://community.livekit.io/t/session-gets-closed-unexpectedly/477) — 2026-03-06
- [Inworld error: Context 'd6b3587d74a7' not found](https://community.livekit.io/t/inworld-error-context-d6b3587d74a7-not-found/476) — 2026-03-06
- [(Plugin Contribution) PR #5014: SpatialReal Avatar Plugin - Request for Review & PyPI Transfer](https://community.livekit.io/t/plugin-contribution-pr-5014-spatialreal-avatar-plugin-request-for-review-pypi-transfer/473) — 2026-03-05
- [Livekit Cloud Agents not working](https://community.livekit.io/t/livekit-cloud-agents-not-working/472) — 2026-03-05
- [Has anyone try integrating livekit with openclaw?](https://community.livekit.io/t/has-anyone-try-integrating-livekit-with-openclaw/471) — 2026-03-05
- [UsageSummary: per-model LLM/STT/TTS usage for cost calculation](https://community.livekit.io/t/usagesummary-per-model-llm-stt-tts-usage-for-cost-calculation/469) — 2026-03-05
- [STT / voice pipeline: providers with built-in sentiment or emotion analysis?](https://community.livekit.io/t/stt-voice-pipeline-providers-with-built-in-sentiment-or-emotion-analysis/466) — 2026-03-05
- [EU data residency with Deepgram, OpenAI & ElevenLabs — how to configure regional endpoints?](https://community.livekit.io/t/eu-data-residency-with-deepgram-openai-elevenlabs-how-to-configure-regional-endpoints/465) — 2026-03-05
- [New default Elevenlabs voices](https://community.livekit.io/t/new-default-elevenlabs-voices/459) — 2026-03-04
- [Solving WebRTC NAT traversal for LiveKit agents in serverless microVMs (AgentCore Runtime)](https://community.livekit.io/t/solving-webrtc-nat-traversal-for-livekit-agents-in-serverless-microvms-agentcore-runtime/455) — 2026-03-04
- [Hello all, what is the difference between inference and plugins. Does inference will be fast compared to plugins?](https://community.livekit.io/t/hello-all-what-is-the-difference-between-inference-and-plugins-does-inference-will-be-fast-compared-to-plugins/449) — 2026-03-04
- [How to enforce agent tool call pre-speech](https://community.livekit.io/t/how-to-enforce-agent-tool-call-pre-speech/448) — 2026-03-04
- [Correct way to receive agent started talking events on the frontend](https://community.livekit.io/t/correct-way-to-receive-agent-started-talking-events-on-the-frontend/444) — 2026-03-03
- [Python livekit-agents@1.4.4 Released](https://community.livekit.io/t/python-livekit-agents-1-4-4-released/439) — 2026-03-03
- [Can I dynamically inject keyterms for STT?](https://community.livekit.io/t/can-i-dynamically-inject-keyterms-for-stt/438) — 2026-03-02
- [Feature request: Add Giskard-based evaluation integration for regression testing of LiveKit agents](https://community.livekit.io/t/feature-request-add-giskard-based-evaluation-integration-for-regression-testing-of-livekit-agents/437) — 2026-03-02
- [AgenSession Say method don't refresh AgentState](https://community.livekit.io/t/agensession-say-method-dont-refresh-agentstate/435) — 2026-03-02
- [How to commit user turn without generating a response?](https://community.livekit.io/t/how-to-commit-user-turn-without-generating-a-response/433) — 2026-03-02
- [Audio mixer hard coded sample rate causes voice warping](https://community.livekit.io/t/audio-mixer-hard-coded-sample-rate-causes-voice-warping/432) — 2026-03-02
- [Is there a node I'm supposed to use once agent turn is done?](https://community.livekit.io/t/is-there-a-node-im-supposed-to-use-once-agent-turn-is-done/431) — 2026-03-02
- [Sometimes Transcripts and audio recording not available for the session (Audio waveforms not available)](https://community.livekit.io/t/sometimes-transcripts-and-audio-recording-not-available-for-the-session-audio-waveforms-not-available/430) — 2026-03-02
- [Agent insight not loaded](https://community.livekit.io/t/agent-insight-not-loaded/428) — 2026-03-02
- [On_shutdown node](https://community.livekit.io/t/on-shutdown-node/427) — 2026-03-01
- [Request PR review and further instructions on merging Gnani.AI STT plugin into LiveKit Agents](https://community.livekit.io/t/request-pr-review-and-further-instructions-on-merging-gnani-ai-stt-plugin-into-livekit-agents/426) — 2026-03-01
- [Can’t delete stale self-hosted agent from dashboard](https://community.livekit.io/t/can-t-delete-stale-self-hosted-agent-from-dashboard/416) — 2026-02-27
- [TTS/STT Inference fails due to APIConnectionError with no clear error message](https://community.livekit.io/t/tts-stt-inference-fails-due-to-apiconnectionerror-with-no-clear-error-message/415) — 2026-02-27
- [Prewarm connections (LLM, TTS) when not using welcome message](https://community.livekit.io/t/prewarm-connections-llm-tts-when-not-using-welcome-message/414) — 2026-02-27
- [setMuted not working on first try](https://community.livekit.io/t/setmuted-not-working-on-first-try/413) — 2026-02-27
- [Agent is not joining the room](https://community.livekit.io/t/agent-is-not-joining-the-room/412) — 2026-02-27
- [Inconsistent transcripts language when using Gemini realtime model ( gemini-live-2.5-flash-native-audio )](https://community.livekit.io/t/inconsistent-transcripts-language-when-using-gemini-realtime-model-gemini-live-2-5-flash-native-audio/410) — 2026-02-27
- [Gemini 2.5 Flash Native Audio skipping letters during ”Spelling Out” tasks](https://community.livekit.io/t/gemini-2-5-flash-native-audio-skipping-letters-during-spelling-out-tasks/409) — 2026-02-27
- [Keyterms missing in Elevenlabs scribe v2 TTS plugin](https://community.livekit.io/t/keyterms-missing-in-elevenlabs-scribe-v2-tts-plugin/408) — 2026-02-27
- [Manage staging and production environment](https://community.livekit.io/t/manage-staging-and-production-environment/407) — 2026-02-27
- [Runner initialization timed out](https://community.livekit.io/t/runner-initialization-timed-out/404) — 2026-02-26
- [Tavus avatar taking time to initialize (>10s of black out)](https://community.livekit.io/t/tavus-avatar-taking-time-to-initialize-10s-of-black-out/402) — 2026-02-26
- [Sarvam TTS — update_options() missing target_language_code](https://community.livekit.io/t/sarvam-tts-update-options-missing-target-language-code/399) — 2026-02-26
- [Agent tool to make outbound calls?](https://community.livekit.io/t/agent-tool-to-make-outbound-calls/393) — 2026-02-26
- [Agent Observability set to Disabled, causing errors (inconsistent behaviour)](https://community.livekit.io/t/agent-observability-set-to-disabled-causing-errors-inconsistent-behaviour/391) — 2026-02-25
- [Using livekit.agents.llm.RealtimeModel with liteLLM](https://community.livekit.io/t/using-livekit-agents-llm-realtimemodel-with-litellm/383) — 2026-02-25
- [Agent routed to EU West B instead of EU Central](https://community.livekit.io/t/agent-routed-to-eu-west-b-instead-of-eu-central/378) — 2026-02-24
- [Achieving multi‑agent awareness and state synchronization with LiveKit Data Channels](https://community.livekit.io/t/achieving-multi-agent-awareness-and-state-synchronization-with-livekit-data-channels/377) — 2026-02-24
- [Observed STT/TTS Concurrency Behavior During Internal Testing](https://community.livekit.io/t/observed-stt-tts-concurrency-behavior-during-internal-testing/367) — 2026-02-23
- [Issue with programmatically toggle STT/TTS on off](https://community.livekit.io/t/issue-with-programmatically-toggle-stt-tts-on-off/363) — 2026-02-23
- [Configurable Backchannel Responses for Natural Conversational Flow](https://community.livekit.io/t/configurable-backchannel-responses-for-natural-conversational-flow/361) — 2026-02-23
- [Background audio mixer closed in NodeJs](https://community.livekit.io/t/background-audio-mixer-closed-in-nodejs/360) — 2026-02-23
- [Python livekit-agents 1.4.3 released](https://community.livekit.io/t/python-livekit-agents-1-4-3-released/359) — 2026-02-23
- [Voice Live API connector](https://community.livekit.io/t/voice-live-api-connector/351) — 2026-02-23
- [Add PersonaPlex (NVIDIA) realtime plugin](https://community.livekit.io/t/add-personaplex-nvidia-realtime-plugin/350) — 2026-02-22
- [Inbound call got ended by itself (Livekit sent Bye request)](https://community.livekit.io/t/inbound-call-got-ended-by-itself-livekit-sent-bye-request/349) — 2026-02-22
- [Cloud-deployed agent never connects](https://community.livekit.io/t/cloud-deployed-agent-never-connects/342) — 2026-02-21
- [Gemini live cut off issue](https://community.livekit.io/t/gemini-live-cut-off-issue/339) — 2026-02-21
- [401 error on python agent.py dev - wss://.livekit.cloud/agent](https://community.livekit.io/t/401-error-on-python-agent-py-dev-wss-livekit-cloud-agent/337) — 2026-02-20
- [How to cache an image on the LLM side](https://community.livekit.io/t/how-to-cache-an-image-on-the-llm-side/336) — 2026-02-20
- [Realtime model with Azure whisper STT](https://community.livekit.io/t/realtime-model-with-azure-whisper-stt/335) — 2026-02-20
- [Livekit failing to use an MCP server that's working on other clients](https://community.livekit.io/t/livekit-failing-to-use-an-mcp-server-thats-working-on-other-clients/334) — 2026-02-20
- [Twilio SIP TLS Certificate Rotation (ClientAuth EKU Removal) – Any Impact on LiveKit Cloud SIP?](https://community.livekit.io/t/twilio-sip-tls-certificate-rotation-clientauth-eku-removal-any-impact-on-livekit-cloud-sip/333) — 2026-02-20
- [Gemini Live issues](https://community.livekit.io/t/gemini-live-issues/331) — 2026-02-19
- [Load Testing LiveKit Agents](https://community.livekit.io/t/load-testing-livekit-agents/330) — 2026-02-19
- [Capturing Agent speech in on_enter of a Task in Tests](https://community.livekit.io/t/capturing-agent-speech-in-on-enter-of-a-task-in-tests/327) — 2026-02-19
- [Change Livekit Cloud WSS Region](https://community.livekit.io/t/change-livekit-cloud-wss-region/325) — 2026-02-19
- [Invalid agent state leads to blocked call](https://community.livekit.io/t/invalid-agent-state-leads-to-blocked-call/322) — 2026-02-18
- [Testing strategy for LiveKit voice agents (roomless vs lk.chat UI tests)](https://community.livekit.io/t/testing-strategy-for-livekit-voice-agents-roomless-vs-lk-chat-ui-tests/321) — 2026-02-18
- [Tool call started event](https://community.livekit.io/t/tool-call-started-event/320) — 2026-02-18
- [Agent tool execution w/ data messages](https://community.livekit.io/t/agent-tool-execution-w-data-messages/318) — 2026-02-17
- [Python livekit-agents 1.4.2 released](https://community.livekit.io/t/python-livekit-agents-1-4-2-released/313) — 2026-02-17
- [Recommended architecture for safe MCP tool execution in LiveKit agents](https://community.livekit.io/t/recommended-architecture-for-safe-mcp-tool-execution-in-livekit-agents/310) — 2026-02-17
- [Making tool calls dynamically and configurable](https://community.livekit.io/t/making-tool-calls-dynamically-and-configurable/308) — 2026-02-16
- [Agent Server memory usage vs. Worker](https://community.livekit.io/t/agent-server-memory-usage-vs-worker/305) — 2026-02-16
- [1.4.2 release request](https://community.livekit.io/t/1-4-2-release-request/299) — 2026-02-13
- [Websocket support for Rime](https://community.livekit.io/t/websocket-support-for-rime/290) — 2026-02-12
- [Which Eleven Labs voices are available on LiveKit Cloud?](https://community.livekit.io/t/which-eleven-labs-voices-are-available-on-livekit-cloud/288) — 2026-02-12
- [Multiple agent sessions created for the same room](https://community.livekit.io/t/multiple-agent-sessions-created-for-the-same-room/287) — 2026-02-12
- [Audio Isolation/Volume control in LiveKit SIP Rooms](https://community.livekit.io/t/audio-isolation-volume-control-in-livekit-sip-rooms/285) — 2026-02-12
- [Adding Deepdub TTS to Livekit](https://community.livekit.io/t/adding-deepdub-tts-to-livekit/284) — 2026-02-11
- [How to retain system instructions in update_chat_ctx?](https://community.livekit.io/t/how-to-retain-system-instructions-in-update-chat-ctx/282) — 2026-02-11
- [Question with LLM tool calling](https://community.livekit.io/t/question-with-llm-tool-calling/280) — 2026-02-10
- [When will AgentTask be available NodeJS](https://community.livekit.io/t/when-will-agenttask-be-available-nodejs/276) — 2026-02-10
- [Observability for voice AI agents: what's actually working for you?](https://community.livekit.io/t/observability-for-voice-ai-agents-whats-actually-working-for-you/275) — 2026-02-09
- [Using the MCP Gateway Pattern to support many MCP servers](https://community.livekit.io/t/using-the-mcp-gateway-pattern-to-support-many-mcp-servers/268) — 2026-02-07
- [Python livekit-agents 1.4.0 released](https://community.livekit.io/t/python-livekit-agents-1-4-0-released/264) — 2026-02-06
- [Failed to perform clear buffer rpc (Simli Avatar Integration)](https://community.livekit.io/t/failed-to-perform-clear-buffer-rpc-simli-avatar-integration/263) — 2026-02-06
- [Agent Join Latency Spikes at ~50 Concurrent Users on Scale Plan](https://community.livekit.io/t/agent-join-latency-spikes-at-50-concurrent-users-on-scale-plan/261) — 2026-02-06
- [RAG helpers for Node agents?](https://community.livekit.io/t/rag-helpers-for-node-agents/259) — 2026-02-06
- [Order and necessity of using session.start and ctx.connect](https://community.livekit.io/t/order-and-necessity-of-using-session-start-and-ctx-connect/246) — 2026-02-03
- [Why does my inbound agent take a long time to start after inactivity?](https://community.livekit.io/t/why-does-my-inbound-agent-take-a-long-time-to-start-after-inactivity/225) — 2026-01-21
- [How to handle long-running async tool calls with OpenAI Realtime API](https://community.livekit.io/t/how-to-handle-long-running-async-tool-calls-with-openai-realtime-api/208) — 2026-01-21
- [Why is my concurrent STT count high after sequential test calls?](https://community.livekit.io/t/why-is-my-concurrent-stt-count-high-after-sequential-test-calls/205) — 2026-01-21
- [How to run a custom speech detection model on LiveKit Cloud](https://community.livekit.io/t/how-to-run-a-custom-speech-detection-model-on-livekit-cloud/201) — 2026-01-21
- [Frontend examples for testing Python agent RPC](https://community.livekit.io/t/frontend-examples-for-testing-python-agent-rpc/200) — 2026-01-21
- [Where should I host self-hosted avatars, TTS, LLM, and STT?](https://community.livekit.io/t/where-should-i-host-self-hosted-avatars-tts-llm-and-stt/199) — 2026-01-21
- [How to add filler words to reduce perceived latency](https://community.livekit.io/t/how-to-add-filler-words-to-reduce-perceived-latency/197) — 2026-01-21
- [How to transcribe all participants in a room, not just the first one](https://community.livekit.io/t/how-to-transcribe-all-participants-in-a-room-not-just-the-first-one/196) — 2026-01-21
- [How to stop routing LLM output to TTS when sound is off](https://community.livekit.io/t/how-to-stop-routing-llm-output-to-tts-when-sound-is-off/195) — 2026-01-21
- [Turn detection support for local languages (Hindi, Punjabi, Tamil)](https://community.livekit.io/t/turn-detection-support-for-local-languages-hindi-punjabi-tamil/193) — 2026-01-21
- [How to detect user silence and timeout after agent speaks](https://community.livekit.io/t/how-to-detect-user-silence-and-timeout-after-agent-speaks/192) — 2026-01-21
- [Does LiveKit return the detected language when using STT?](https://community.livekit.io/t/does-livekit-return-the-detected-language-when-using-stt/191) — 2026-01-21
- [Is it possible to use offline STT (Whisper) and TTS with LiveKit?](https://community.livekit.io/t/is-it-possible-to-use-offline-stt-whisper-and-tts-with-livekit/188) — 2026-01-21
- [Agent speaking audio_text tokens out loud](https://community.livekit.io/t/agent-speaking-audio-text-tokens-out-loud/187) — 2026-01-21
- [How to get full chat context including initial history in custom nodes](https://community.livekit.io/t/how-to-get-full-chat-context-including-initial-history-in-custom-nodes/185) — 2026-01-21
- [Duplicate tool calls when user repeats information](https://community.livekit.io/t/duplicate-tool-calls-when-user-repeats-information/183) — 2026-01-21
- [How to handle page reloads with agent session and participant state](https://community.livekit.io/t/how-to-handle-page-reloads-with-agent-session-and-participant-state/182) — 2026-01-21
- [GitHub Action for deploying agents to LiveKit Cloud](https://community.livekit.io/t/github-action-for-deploying-agents-to-livekit-cloud/181) — 2026-01-21
- [How to add pre/post say method to MCP tools](https://community.livekit.io/t/how-to-add-pre-post-say-method-to-mcp-tools/180) — 2026-01-21
- [Best way to get conversation transcription between agent and caller](https://community.livekit.io/t/best-way-to-get-conversation-transcription-between-agent-and-caller/179) — 2026-01-21
- [How to set max call duration for inbound calls](https://community.livekit.io/t/how-to-set-max-call-duration-for-inbound-calls/177) — 2026-01-21
- [Cannot remove messages from Gemini Live chat context](https://community.livekit.io/t/cannot-remove-messages-from-gemini-live-chat-context/175) — 2026-01-21
- [How to isolate user's voice for transcription with background speakers](https://community.livekit.io/t/how-to-isolate-users-voice-for-transcription-with-background-speakers/174) — 2026-01-21
- [How to access full chat history inside a function_tool](https://community.livekit.io/t/how-to-access-full-chat-history-inside-a-function-tool/171) — 2026-01-21
- [How to update an existing agent's name via CLI](https://community.livekit.io/t/how-to-update-an-existing-agents-name-via-cli/170) — 2026-01-21
- [How to prevent interruptions during the start of a conversation](https://community.livekit.io/t/how-to-prevent-interruptions-during-the-start-of-a-conversation/169) — 2026-01-21
- [How to limit call duration and automatically disconnect](https://community.livekit.io/t/how-to-limit-call-duration-and-automatically-disconnect/168) — 2026-01-21
- [Chat context contains text that wasn't spoken due to interruption](https://community.livekit.io/t/chat-context-contains-text-that-wasnt-spoken-due-to-interruption/165) — 2026-01-21
- [How to send data from Python agent to React frontend for navigation](https://community.livekit.io/t/how-to-send-data-from-python-agent-to-react-frontend-for-navigation/164) — 2026-01-21
- [How to dynamically switch TTS language based on LLM response](https://community.livekit.io/t/how-to-dynamically-switch-tts-language-based-on-llm-response/163) — 2026-01-21
- [How to set max tokens for OpenAI Realtime model](https://community.livekit.io/t/how-to-set-max-tokens-for-openai-realtime-model/162) — 2026-01-21
- [How to build a voice agent that follows a script but handles off-script conversations](https://community.livekit.io/t/how-to-build-a-voice-agent-that-follows-a-script-but-handles-off-script-conversations/161) — 2026-01-21
- [How to change STT provider dynamically during a session](https://community.livekit.io/t/how-to-change-stt-provider-dynamically-during-a-session/158) — 2026-01-21
- [How to send text to participants and test locally](https://community.livekit.io/t/how-to-send-text-to-participants-and-test-locally/157) — 2026-01-21
- [How to instantly destroy a room when it becomes empty](https://community.livekit.io/t/how-to-instantly-destroy-a-room-when-it-becomes-empty/154) — 2026-01-21
- [Questions about AgentServer, WorkerOptions, and Prewarm in v1.3+](https://community.livekit.io/t/questions-about-agentserver-workeroptions-and-prewarm-in-v1-3/152) — 2026-01-21
- [How to have multiple participants interact with an agent in the same room](https://community.livekit.io/t/how-to-have-multiple-participants-interact-with-an-agent-in-the-same-room/151) — 2026-01-21
- [How to track logs for a specific agent worker by room name](https://community.livekit.io/t/how-to-track-logs-for-a-specific-agent-worker-by-room-name/150) — 2026-01-21
- [How to handle multiple agent configurations for different customers](https://community.livekit.io/t/how-to-handle-multiple-agent-configurations-for-different-customers/149) — 2026-01-21
- [What is context.session.agent_state and how does it work?](https://community.livekit.io/t/what-is-context-session-agent-state-and-how-does-it-work/148) — 2026-01-21
- [Session disruption when downscaling ECS Fargate tasks](https://community.livekit.io/t/session-disruption-when-downscaling-ecs-fargate-tasks/147) — 2026-01-21
- [Qwen model speaking tool call JSON out loud instead of executing](https://community.livekit.io/t/qwen-model-speaking-tool-call-json-out-loud-instead-of-executing/146) — 2026-01-21
- [Is there support for ElevenLabs pronunciation dictionaries?](https://community.livekit.io/t/is-there-support-for-elevenlabs-pronunciation-dictionaries/145) — 2026-01-21
- [LocalEntryNotFoundError when using MultilingualModel turn detector](https://community.livekit.io/t/localentrynotfounderror-when-using-multilingualmodel-turn-detector/144) — 2026-01-21
- [Agent disconnects when room becomes empty - how to prevent?](https://community.livekit.io/t/agent-disconnects-when-room-becomes-empty-how-to-prevent/142) — 2026-01-21
- [Is there multi-voice TTS support for switching voices mid-sentence?](https://community.livekit.io/t/is-there-multi-voice-tts-support-for-switching-voices-mid-sentence/141) — 2026-01-21
- [Do agent messages go to all participants in a multi-participant room?](https://community.livekit.io/t/do-agent-messages-go-to-all-participants-in-a-multi-participant-room/139) — 2026-01-21
- [How to download or export the chat transcript](https://community.livekit.io/t/how-to-download-or-export-the-chat-transcript/138) — 2026-01-21
- [How to deploy self-hosted agents without disrupting ongoing calls](https://community.livekit.io/t/how-to-deploy-self-hosted-agents-without-disrupting-ongoing-calls/137) — 2026-01-21
- [How to remove an agent from a room](https://community.livekit.io/t/how-to-remove-an-agent-from-a-room/136) — 2026-01-21
- [What's the best way to get call duration?](https://community.livekit.io/t/whats-the-best-way-to-get-call-duration/134) — 2026-01-21
- [How to contribute a new video agent plugin to LiveKit](https://community.livekit.io/t/how-to-contribute-a-new-video-agent-plugin-to-livekit/133) — 2026-01-21
- [Real-time STT with auto language detection and code-switching support](https://community.livekit.io/t/real-time-stt-with-auto-language-detection-and-code-switching-support/132) — 2026-01-21
- [How to detect if the user is idle and disconnect the room](https://community.livekit.io/t/how-to-detect-if-the-user-is-idle-and-disconnect-the-room/131) — 2026-01-21
- [MCP tool timeout error even though tool calls succeed](https://community.livekit.io/t/mcp-tool-timeout-error-even-though-tool-calls-succeed/129) — 2026-01-21
- [How to log inputs and outputs for LLM node](https://community.livekit.io/t/how-to-log-inputs-and-outputs-for-llm-node/127) — 2026-01-21
- [How to capture agent messages that were interrupted](https://community.livekit.io/t/how-to-capture-agent-messages-that-were-interrupted/126) — 2026-01-21
- [What are the concurrent call limits for LiveKit agents?](https://community.livekit.io/t/what-are-the-concurrent-call-limits-for-livekit-agents/123) — 2026-01-21
- [How to get performance metrics for the STT-LLM-TTS pipeline](https://community.livekit.io/t/how-to-get-performance-metrics-for-the-stt-llm-tts-pipeline/122) — 2026-01-21
- [Warning: resume_false_interruption is enabled but audio output does not support pause](https://community.livekit.io/t/warning-resume-false-interruption-is-enabled-but-audio-output-does-not-support-pause/120) — 2026-01-21
- [Cartesia TTS volume and emotion parameters not working](https://community.livekit.io/t/cartesia-tts-volume-and-emotion-parameters-not-working/119) — 2026-01-21
- [Is UsageSummary reliable for client billing with OpenAI Realtime?](https://community.livekit.io/t/is-usagesummary-reliable-for-client-billing-with-openai-realtime/118) — 2026-01-21
- [Why doesn't on_user_turn_completed fire in chat mode?](https://community.livekit.io/t/why-doesnt-on-user-turn-completed-fire-in-chat-mode/117) — 2026-01-21
- [Does TTS/STT latency include round-trip to external providers?](https://community.livekit.io/t/does-tts-stt-latency-include-round-trip-to-external-providers/116) — 2026-01-21
- [How to pause STT/LLM processing during a long-running function call](https://community.livekit.io/t/how-to-pause-stt-llm-processing-during-a-long-running-function-call/114) — 2026-01-21
- [How to correlate dispatched agent with participant joined event](https://community.livekit.io/t/how-to-correlate-dispatched-agent-with-participant-joined-event/110) — 2026-01-21
- [Does LLM output stream directly to TTS or wait for complete response?](https://community.livekit.io/t/does-llm-output-stream-directly-to-tts-or-wait-for-complete-response/107) — 2026-01-21
- [Can I use session.say() with a realtime model?](https://community.livekit.io/t/can-i-use-session-say-with-a-realtime-model/106) — 2026-01-21
- [How to switch TTS during agent runtime](https://community.livekit.io/t/how-to-switch-tts-during-agent-runtime/105) — 2026-01-21
- [How to trigger shutdown_hook when an error occurs](https://community.livekit.io/t/how-to-trigger-shutdown-hook-when-an-error-occurs/104) — 2026-01-21
- [Agent not joining room after deleting and recreating with same name](https://community.livekit.io/t/agent-not-joining-room-after-deleting-and-recreating-with-same-name/103) — 2026-01-21
- [Are min_interruption_words and min_interruption_duration mutually exclusive?](https://community.livekit.io/t/are-min-interruption-words-and-min-interruption-duration-mutually-exclusive/101) — 2026-01-21
- [How are workers distributed across multiple LiveKit server nodes?](https://community.livekit.io/t/how-are-workers-distributed-across-multiple-livekit-server-nodes/100) — 2026-01-21
- [How to support multiple languages in a single agent pipeline](https://community.livekit.io/t/how-to-support-multiple-languages-in-a-single-agent-pipeline/98) — 2026-01-21
- [TTS recommendations for natural conversational voice agents](https://community.livekit.io/t/tts-recommendations-for-natural-conversational-voice-agents/96) — 2026-01-21
- [Where are call recordings and transcripts stored?](https://community.livekit.io/t/where-are-call-recordings-and-transcripts-stored/94) — 2026-01-21
- [How to update environment variables for agents hosted on LiveKit Cloud](https://community.livekit.io/t/how-to-update-environment-variables-for-agents-hosted-on-livekit-cloud/93) — 2026-01-21
- [ElevenLabs TTS 'Could not synthesize' error](https://community.livekit.io/t/elevenlabs-tts-could-not-synthesize-error/91) — 2026-01-21
- [Is fallback supported for realtime models?](https://community.livekit.io/t/is-fallback-supported-for-realtime-models/90) — 2026-01-21
- [Can I change the tools available to an agent dynamically during a call?](https://community.livekit.io/t/can-i-change-the-tools-available-to-an-agent-dynamically-during-a-call/88) — 2026-01-21
- [How to run a single LiveKit worker for multiple clients (multi-tenancy)](https://community.livekit.io/t/how-to-run-a-single-livekit-worker-for-multiple-clients-multi-tenancy/87) — 2026-01-21
- [How to structure a Python project with multiple agents sharing code](https://community.livekit.io/t/how-to-structure-a-python-project-with-multiple-agents-sharing-code/86) — 2026-01-21
- [Can turn detection mode be changed during runtime?](https://community.livekit.io/t/can-turn-detection-mode-be-changed-during-runtime/80) — 2026-01-21
- [How to trigger an agent transfer programmatically](https://community.livekit.io/t/how-to-trigger-an-agent-transfer-programmatically/79) — 2026-01-21
- [How to set the agent name for a hosted agent](https://community.livekit.io/t/how-to-set-the-agent-name-for-a-hosted-agent/77) — 2026-01-21
- [How to implement message persistence and resume conversations with a LiveKit agent](https://community.livekit.io/t/how-to-implement-message-persistence-and-resume-conversations-with-a-livekit-agent/62) — 2026-01-21
- [About the Agents category](https://community.livekit.io/t/about-the-agents-category/11) — 2025-12-22

## Telephony (122)

- [Livekit Outbound Call to Exotel does not have audio on Both Sides](https://community.livekit.io/t/livekit-outbound-call-to-exotel-does-not-have-audio-on-both-sides/1539) — 2026-07-01
- [transferSIPParticipant: No event or callback when transferred call is declined by the receiving](https://community.livekit.io/t/transfersipparticipant-no-event-or-callback-when-transferred-call-is-declined-by-the-receiving/1527) — 2026-06-29
- [Inbound BYOT SIP (Twilio): calls match trunk but always return 486 Busy, no room created](https://community.livekit.io/t/inbound-byot-sip-twilio-calls-match-trunk-but-always-return-486-busy-no-room-created/1506) — 2026-06-25
- [WarmTransferTask: how to tear down a 2-party SIP room when one party hangs up after the agent has left?](https://community.livekit.io/t/warmtransfertask-how-to-tear-down-a-2-party-sip-room-when-one-party-hangs-up-after-the-agent-has-left/1499) — 2026-06-24
- [Feature Request: Allow external SIP Trunks to register with username and password](https://community.livekit.io/t/feature-request-allow-external-sip-trunks-to-register-with-username-and-password/1479) — 2026-06-22
- [Control the outbound SIP INVITE codec offer (avoid G.722) on LiveKit Cloud?](https://community.livekit.io/t/control-the-outbound-sip-invite-codec-offer-avoid-g-722-on-livekit-cloud/1476) — 2026-06-22
- [Call Lag During Warm Transfer After Agent Exit](https://community.livekit.io/t/call-lag-during-warm-transfer-after-agent-exit/1444) — 2026-06-17
- [Help with Call Errors and active calls in Livekit Cloud](https://community.livekit.io/t/help-with-call-errors-and-active-calls-in-livekit-cloud/1435) — 2026-06-15
- [BAD signal handling](https://community.livekit.io/t/bad-signal-handling/1414) — 2026-06-11
- [Error in LiveKit SIP calls](https://community.livekit.io/t/error-in-livekit-sip-calls/1408) — 2026-06-11
- [Inbound BYOT SIP (Twilio): matches trunk (100 Processing) but no room/dispatch ever created — newly-created project](https://community.livekit.io/t/inbound-byot-sip-twilio-matches-trunk-100-processing-but-no-room-dispatch-ever-created-newly-created-project/1406) — 2026-06-11
- [TLS and SIP REFER](https://community.livekit.io/t/tls-and-sip-refer/1324) — 2026-06-02
- [SIP inbound — agent receives only zero-filled audio frames despite SIP ingress upstream > 0](https://community.livekit.io/t/sip-inbound-agent-receives-only-zero-filled-audio-frames-despite-sip-ingress-upstream-0/1315) — 2026-06-02
- [SIP support for the opus codec](https://community.livekit.io/t/sip-support-for-the-opus-codec/1298) — 2026-06-01
- [Multi-tenant SIP topology: one Twilio Elastic Trunk per subaccount, or a shared setup?](https://community.livekit.io/t/multi-tenant-sip-topology-one-twilio-elastic-trunk-per-subaccount-or-a-shared-setup/1265) — 2026-05-27
- [Call forward to human agent — human's phone shows our number instead of the caller's](https://community.livekit.io/t/call-forward-to-human-agent-humans-phone-shows-our-number-instead-of-the-callers/1247) — 2026-05-26
- [How do you handle calls for businesses?](https://community.livekit.io/t/how-do-you-handle-calls-for-businesses/1219) — 2026-05-24
- [Does Telnyx support HD calling for EU-based businesses?](https://community.livekit.io/t/does-telnyx-support-hd-calling-for-eu-based-businesses/1208) — 2026-05-23
- [Allow configuring or replacing the SIP play_dialtone audio](https://community.livekit.io/t/allow-configuring-or-replacing-the-sip-play-dialtone-audio/1204) — 2026-05-21
- [Outbound dialing capabilities for LiveKit-hosted phone numbers](https://community.livekit.io/t/outbound-dialing-capabilities-for-livekit-hosted-phone-numbers/1199) — 2026-05-21
- [Yesterday, REFERs stopped matching on outbound](https://community.livekit.io/t/yesterday-refers-stopped-matching-on-outbound/1190) — 2026-05-20
- [Inbound SIP 404 ”Does not matchTrunks or Dispatch Rules” with byte-identical +E.164 on trunk, rule, and INVITE — Telnyx → LiveKit Cloud (Frankfurt)](https://community.livekit.io/t/inbound-sip-404-does-not-matchtrunks-or-dispatch-rules-with-byte-identical-e-164-on-trunk-rule-and-invite-telnyx-livekit-cloud-frankfurt/1164) — 2026-05-18
- [How can join my sip uri with api key to livekit room?](https://community.livekit.io/t/how-can-join-my-sip-uri-with-api-key-to-livekit-room/1163) — 2026-05-18
- [Real-time statuspage notification for the recent us-east media connectivity incident](https://community.livekit.io/t/real-time-statuspage-notification-for-the-recent-us-east-media-connectivity-incident/1160) — 2026-05-18
- [FQDN SIP connection](https://community.livekit.io/t/fqdn-sip-connection/1150) — 2026-05-17
- [Phone number qouta limits](https://community.livekit.io/t/phone-number-qouta-limits/1118) — 2026-05-14
- [Mitel SIP Trunk Connected to LiveKit but No Audio / Call Terminates After ~25 Seconds](https://community.livekit.io/t/mitel-sip-trunk-connected-to-livekit-but-no-audio-call-terminates-after-25-seconds/1108) — 2026-05-13
- [TwiML SIP dispatch occasionally ringing endlessly](https://community.livekit.io/t/twiml-sip-dispatch-occasionally-ringing-endlessly/1097) — 2026-05-12
- [Fixed outbound SIP trunk](https://community.livekit.io/t/fixed-outbound-sip-trunk/1094) — 2026-05-12
- [Request: Enable WhatsApp Connector for LiveKit Cloud project](https://community.livekit.io/t/request-enable-whatsapp-connector-for-livekit-cloud-project/1089) — 2026-05-12
- [(ISSUE) SIP bridge sends BYE during LLM+TTS processing gap (~14s silence), Twilio Elastic SIP Trunk reports ”caller hung up”](https://community.livekit.io/t/issue-sip-bridge-sends-bye-during-llm-tts-processing-gap-14s-silence-twilio-elastic-sip-trunk-reports-caller-hung-up/1080) — 2026-05-11
- [Outbound calls fail with: ”INVITE failed: sip status: 404: No trunk found”](https://community.livekit.io/t/outbound-calls-fail-with-invite-failed-sip-status-404-no-trunk-found/1059) — 2026-05-08
- [Getting Media Timeout Error For calls in Europe Region](https://community.livekit.io/t/getting-media-timeout-error-for-calls-in-europe-region/1048) — 2026-05-07
- [Inbound SIP returning 404 from LK Cloud edge before session creation — Telnyx FQDN, 0 dashboard sessions](https://community.livekit.io/t/inbound-sip-returning-404-from-lk-cloud-edge-before-session-creation-telnyx-fqdn-0-dashboard-sessions/1044) — 2026-05-06
- [Caller audio is often too low for reliable STT pickup unless the caller speaks loudly](https://community.livekit.io/t/caller-audio-is-often-too-low-for-reliable-stt-pickup-unless-the-caller-speaks-loudly/1035) — 2026-05-06
- [Twilio SIP INVITEs to LiveKit project p_3lrmo3qmlaz never logged in Telephony](https://community.livekit.io/t/twilio-sip-invites-to-livekit-project-p-3lrmo3qmlaz-never-logged-in-telephony/1011) — 2026-05-03
- [Bird SIP trunk → LiveKit integration for inbound PSTN calls (room routing + call control questions)](https://community.livekit.io/t/bird-sip-trunk-livekit-integration-for-inbound-pstn-calls-room-routing-call-control-questions/1001) — 2026-05-01
- [Adding Phone Lines](https://community.livekit.io/t/adding-phone-lines/987) — 2026-04-29
- [Intermittent silent calls: cross-region media routing failure between Germany and UK SIP bridge (30s BYE, agent audio in recording but no client audio)](https://community.livekit.io/t/intermittent-silent-calls-cross-region-media-routing-failure-between-germany-and-uk-sip-bridge-30s-bye-agent-audio-in-recording-but-no-client-audio/985) — 2026-04-29
- [Adding Telephony Providers](https://community.livekit.io/t/adding-telephony-providers/980) — 2026-04-28
- [SIP REFER Refer-To URI parameters stripped by LiveKit SIP bridge](https://community.livekit.io/t/sip-refer-refer-to-uri-parameters-stripped-by-livekit-sip-bridge/942) — 2026-04-23
- [Is there a way to do SIP transfers with extension using Livekit?](https://community.livekit.io/t/is-there-a-way-to-do-sip-transfers-with-extension-using-livekit/937) — 2026-04-23
- [AI Voice Agent Compromised by Injected Looping Text](https://community.livekit.io/t/ai-voice-agent-compromised-by-injected-looping-text/929) — 2026-04-22
- [SIP 200OK not sent to Telnyx after agent joins the room](https://community.livekit.io/t/sip-200ok-not-sent-to-telnyx-after-agent-joins-the-room/924) — 2026-04-21
- [Inbound SIP calls: caller hears silence; WebRTC works (project p_mu9nuejzw03)](https://community.livekit.io/t/inbound-sip-calls-caller-hears-silence-webrtc-works-project-p-mu9nuejzw03/921) — 2026-04-21
- [SIP REFER transfer via Telnyx inbound trunk — 603 Decline on all attempts](https://community.livekit.io/t/sip-refer-transfer-via-telnyx-inbound-trunk-603-decline-on-all-attempts/901) — 2026-04-17
- [Second Call Transfer Attempt Fails with TwirpError: call already being transferred elsewhere](https://community.livekit.io/t/second-call-transfer-attempt-fails-with-twirperror-call-already-being-transferred-elsewhere/899) — 2026-04-17
- [Missing PCAP file for outbound calls](https://community.livekit.io/t/missing-pcap-file-for-outbound-calls/857) — 2026-04-14
- [Getting SIP Invite request from new IP for Germany](https://community.livekit.io/t/getting-sip-invite-request-from-new-ip-for-germany/838) — 2026-04-11
- [AllowedAddresses filter on SIP Trunk](https://community.livekit.io/t/allowedaddresses-filter-on-sip-trunk/833) — 2026-04-10
- [I need urgent help!](https://community.livekit.io/t/i-need-urgent-help/830) — 2026-04-10
- [Agent dispatch silently fails and doesn't join room](https://community.livekit.io/t/agent-dispatch-silently-fails-and-doesnt-join-room/824) — 2026-04-09
- [Agent not joining sessions](https://community.livekit.io/t/agent-not-joining-sessions/823) — 2026-04-09
- [For Inbound calls user not able to hear voice agent](https://community.livekit.io/t/for-inbound-calls-user-not-able-to-hear-voice-agent/822) — 2026-04-09
- [404 ”No trunk found” — Zadarma (Bulgaria) to LiveKit Cloud — GeoDNS routing issue?](https://community.livekit.io/t/404-no-trunk-found-zadarma-bulgaria-to-livekit-cloud-geodns-routing-issue/767) — 2026-04-03
- [How to fix Latency (3 secs) of Gemini 3.1 Flash Live](https://community.livekit.io/t/how-to-fix-latency-3-secs-of-gemini-3-1-flash-live/766) — 2026-04-03
- [Does Gemini 3.1 Flash Live work well on telephony?](https://community.livekit.io/t/does-gemini-3-1-flash-live-work-well-on-telephony/756) — 2026-04-02
- [Outbound SIP — only 486 and 480 produce specific DisconnectReasons, everything else falls through to CLIENT_INITIATED](https://community.livekit.io/t/outbound-sip-only-486-and-480-produce-specific-disconnectreasons-everything-else-falls-through-to-client-initiated/728) — 2026-03-31
- [SIP 404 ”No trunk found” — DNS routing mismatch on LiveKit Cloud](https://community.livekit.io/t/sip-404-no-trunk-found-dns-routing-mismatch-on-livekit-cloud/703) — 2026-03-28
- [WhatsApp Calling SIP inbound from LiveKit not ringing (drops during dialing)](https://community.livekit.io/t/whatsapp-calling-sip-inbound-from-livekit-not-ringing-drops-during-dialing/691) — 2026-03-26
- [Call connects but no audio (agent responds in logs) and drops after ~30 seconds](https://community.livekit.io/t/call-connects-but-no-audio-agent-responds-in-logs-and-drops-after-30-seconds/679) — 2026-03-25
- [Intermittently not receiving 200 OK SDP from Livekit](https://community.livekit.io/t/intermittently-not-receiving-200-ok-sdp-from-livekit/676) — 2026-03-25
- [Twilio CPS error](https://community.livekit.io/t/twilio-cps-error/671) — 2026-03-25
- [I'm getting a media-timeout error when doing an outbound call with LK & Twilio](https://community.livekit.io/t/im-getting-a-media-timeout-error-when-doing-an-outbound-call-with-lk-twilio/669) — 2026-03-25
- [LiveKit Voice Agent + WhatsApp — mid-call & post-call integration patterns](https://community.livekit.io/t/livekit-voice-agent-whatsapp-mid-call-post-call-integration-patterns/667) — 2026-03-25
- [Carrier-specific audio issues with SIP telephony callers (G.711 via Asterisk) — seeking advice on VAD/NC configuration](https://community.livekit.io/t/carrier-specific-audio-issues-with-sip-telephony-callers-g-711-via-asterisk-seeking-advice-on-vad-nc-configuration/646) — 2026-03-23
- [Trusted CA certificates on the SIP FE/SBC's for TLS](https://community.livekit.io/t/trusted-ca-certificates-on-the-sip-fe-sbcs-for-tls/641) — 2026-03-22
- [404 No trunk found on every inbound SIP call — all configurations tried](https://community.livekit.io/t/404-no-trunk-found-on-every-inbound-sip-call-all-configurations-tried/634) — 2026-03-20
- [Trying to connect direct whatsapp with livekit sip trunk](https://community.livekit.io/t/trying-to-connect-direct-whatsapp-with-livekit-sip-trunk/628) — 2026-03-20
- [How to send BYE signal with Custom X-headers when call disconnects?](https://community.livekit.io/t/how-to-send-bye-signal-with-custom-x-headers-when-call-disconnects/605) — 2026-03-18
- [Telephone Integration issue with firewall](https://community.livekit.io/t/telephone-integration-issue-with-firewall/601) — 2026-03-17
- [Livekit - Whatsapp media relay issue](https://community.livekit.io/t/livekit-whatsapp-media-relay-issue/597) — 2026-03-17
- [SIP participants transcripts](https://community.livekit.io/t/sip-participants-transcripts/551) — 2026-03-13
- [Audio Latency Issues in SIP Trunk and Telephony Number](https://community.livekit.io/t/audio-latency-issues-in-sip-trunk-and-telephony-number/538) — 2026-03-12
- [Outbound SIP audio artifacts on carrier phones (T-Mobile) — is LiveKit Phone Numbers outbound on the roadmap soon?](https://community.livekit.io/t/outbound-sip-audio-artifacts-on-carrier-phones-t-mobile-is-livekit-phone-numbers-outbound-on-the-roadmap-soon/512) — 2026-03-10
- [Setting up SIP Provider with IP-Based Authentication and G.711/G.729 Codecs in LiveKit](https://community.livekit.io/t/setting-up-sip-provider-with-ip-based-authentication-and-g-711-g-729-codecs-in-livekit/499) — 2026-03-09
- [Gemini Realtime latency spikes and unexpected call termination](https://community.livekit.io/t/gemini-realtime-latency-spikes-and-unexpected-call-termination/497) — 2026-03-09
- [Call ending after 33 seconds automatically - URGENT](https://community.livekit.io/t/call-ending-after-33-seconds-automatically-urgent/483) — 2026-03-06
- [Inbound SIP calls from Exotel not reaching agent — 503 response](https://community.livekit.io/t/inbound-sip-calls-from-exotel-not-reaching-agent-503-response/468) — 2026-03-05
- [Japan ingress pinning?](https://community.livekit.io/t/japan-ingress-pinning/467) — 2026-03-05
- [Inbound Calls not reaching Agent](https://community.livekit.io/t/inbound-calls-not-reaching-agent/461) — 2026-03-04
- [Currently observing SIP connection timeouts on inbound calls. Currently observed over 1000 call failures](https://community.livekit.io/t/currently-observing-sip-connection-timeouts-on-inbound-calls-currently-observed-over-1000-call-failures/434) — 2026-03-02
- [Lot of latency issues](https://community.livekit.io/t/lot-of-latency-issues/421) — 2026-02-28
- [Call State Mismatch Between Twilio and LiveKit & High Volume of 1-Second Inbound Calls](https://community.livekit.io/t/call-state-mismatch-between-twilio-and-livekit-high-volume-of-1-second-inbound-calls/406) — 2026-02-27
- [Twilio to LiveKit SIP: ”invalid credentials were provided”](https://community.livekit.io/t/twilio-to-livekit-sip-invalid-credentials-were-provided/405) — 2026-02-27
- [Outbound trunk creation: wildcard * rejected in dashboard but works via CLI](https://community.livekit.io/t/outbound-trunk-creation-wildcard-rejected-in-dashboard-but-works-via-cli/376) — 2026-02-24
- [Outbound SIP call: AI agent speaks before callee's phone rings — missing 180 Ringing in PCAP](https://community.livekit.io/t/outbound-sip-call-ai-agent-speaks-before-callees-phone-rings-missing-180-ringing-in-pcap/368) — 2026-02-23
- [Krisp VIVA integration](https://community.livekit.io/t/krisp-viva-integration/324) — 2026-02-18
- [SIP gateway sends 180 Ringing ~12s after agent joins room and publishes audio](https://community.livekit.io/t/sip-gateway-sends-180-ringing-12s-after-agent-joins-room-and-publishes-audio/323) — 2026-02-18
- [One direction call only - URGENT](https://community.livekit.io/t/one-direction-call-only-urgent/312) — 2026-02-17
- [Audio gain before BVCTelephony()](https://community.livekit.io/t/audio-gain-before-bvctelephony/300) — 2026-02-13
- [LiveKit phone numbers pricing inquiry](https://community.livekit.io/t/livekit-phone-numbers-pricing-inquiry/295) — 2026-02-13
- [Subject: 325ms one-way audio latency in SIP inbound trunk — need internal pipeline diagnostics](https://community.livekit.io/t/subject-325ms-one-way-audio-latency-in-sip-inbound-trunk-need-internal-pipeline-diagnostics/270) — 2026-02-09
- [Not Getting Audio from Livekit Agent Since Morning, checked multiple telephony services](https://community.livekit.io/t/not-getting-audio-from-livekit-agent-since-morning-checked-multiple-telephony-services/257) — 2026-02-06
- [SIP Inbound Trunk - One-Way Audio Issue](https://community.livekit.io/t/sip-inbound-trunk-one-way-audio-issue/250) — 2026-02-05
- [Troubleshooting inbound SIP calls not reaching agent](https://community.livekit.io/t/troubleshooting-inbound-sip-calls-not-reaching-agent/226) — 2026-01-21
- [How to identify the source when receiving transferred calls](https://community.livekit.io/t/how-to-identify-the-source-when-receiving-transferred-calls/224) — 2026-01-21
- [Wireshark warning about truncated PCAP files from LiveKit Cloud](https://community.livekit.io/t/wireshark-warning-about-truncated-pcap-files-from-livekit-cloud/223) — 2026-01-21
- [Where to find SIP trunk IP addresses for allow-listing](https://community.livekit.io/t/where-to-find-sip-trunk-ip-addresses-for-allow-listing/222) — 2026-01-21
- [How to bind a purchased phone number to a dispatch rule via API](https://community.livekit.io/t/how-to-bind-a-purchased-phone-number-to-a-dispatch-rule-via-api/221) — 2026-01-21
- [How to attach custom headers to SIP REFER using TransferSIPParticipant](https://community.livekit.io/t/how-to-attach-custom-headers-to-sip-refer-using-transfersipparticipant/220) — 2026-01-21
- [How to change the SIP port for an outbound trunk](https://community.livekit.io/t/how-to-change-the-sip-port-for-an-outbound-trunk/219) — 2026-01-21
- [How to detect voicemail on outbound calls](https://community.livekit.io/t/how-to-detect-voicemail-on-outbound-calls/218) — 2026-01-21
- [Error: can't transfer non-established call](https://community.livekit.io/t/error-cant-transfer-non-established-call/217) — 2026-01-21
- [503 Service Unavailable for specific phone number on SIP trunk](https://community.livekit.io/t/503-service-unavailable-for-specific-phone-number-on-sip-trunk/216) — 2026-01-21
- [Can I use IP address instead of URI for SIP connections?](https://community.livekit.io/t/can-i-use-ip-address-instead-of-uri-for-sip-connections/215) — 2026-01-21
- [Two agents joining the same outbound telephony room](https://community.livekit.io/t/two-agents-joining-the-same-outbound-telephony-room/214) — 2026-01-21
- [Best practice for SIP noise cancellation in LiveKit Cloud](https://community.livekit.io/t/best-practice-for-sip-noise-cancellation-in-livekit-cloud/213) — 2026-01-21
- [403 Forbidden error when setting up outbound trunk with Twilio](https://community.livekit.io/t/403-forbidden-error-when-setting-up-outbound-trunk-with-twilio/209) — 2026-01-21
- [How to distinguish between answered calls and voicemail for outbound calls](https://community.livekit.io/t/how-to-distinguish-between-answered-calls-and-voicemail-for-outbound-calls/184) — 2026-01-21
- [Custom SIP headers not appearing in participant attributes](https://community.livekit.io/t/custom-sip-headers-not-appearing-in-participant-attributes/153) — 2026-01-21
- [Can LiveKit telephony work without SIP using WebSocket?](https://community.livekit.io/t/can-livekit-telephony-work-without-sip-using-websocket/124) — 2026-01-21
- [Call transfer error: participant does not exist (Exotel)](https://community.livekit.io/t/call-transfer-error-participant-does-not-exist-exotel/121) — 2026-01-21
- [Does LiveKit support SIP providers that use trunk registration (REGISTER)?](https://community.livekit.io/t/does-livekit-support-sip-providers-that-use-trunk-registration-register/113) — 2026-01-21
- [How to get the caller's phone number for inbound calls](https://community.livekit.io/t/how-to-get-the-callers-phone-number-for-inbound-calls/111) — 2026-01-21
- [How to make an AI agent hang up a SIP call proactively](https://community.livekit.io/t/how-to-make-an-ai-agent-hang-up-a-sip-call-proactively/108) — 2026-01-21
- [Does LiveKit support custom SIP headers for inbound and outbound calls?](https://community.livekit.io/t/does-livekit-support-custom-sip-headers-for-inbound-and-outbound-calls/102) — 2026-01-21
- [How to use LiveKit voice agents in countries where phone numbers can't be purchased](https://community.livekit.io/t/how-to-use-livekit-voice-agents-in-countries-where-phone-numbers-cant-be-purchased/99) — 2026-01-21
- [Can I use Asterisk or Issabel as a SIP provider with LiveKit?](https://community.livekit.io/t/can-i-use-asterisk-or-issabel-as-a-sip-provider-with-livekit/83) — 2026-01-21
- [How to preserve caller ID during SIP warm transfer with Twilio](https://community.livekit.io/t/how-to-preserve-caller-id-during-sip-warm-transfer-with-twilio/81) — 2026-01-21
- [How to use wildcard for inbound trunk numbers](https://community.livekit.io/t/how-to-use-wildcard-for-inbound-trunk-numbers/76) — 2026-01-21
- [About the Telephony category](https://community.livekit.io/t/about-the-telephony-category/16) — 2025-12-22

## Client SDKs (48)

- [Call initialize before setting the stream](https://community.livekit.io/t/call-initialize-before-setting-the-stream/1526) — 2026-06-29
- [Audio stops in LiveKit client after microphone cable looseness, but microphone hardware works fine and client doesn't crash](https://community.livekit.io/t/audio-stops-in-livekit-client-after-microphone-cable-looseness-but-microphone-hardware-works-fine-and-client-doesnt-crash/1432) — 2026-06-15
- [Participant are subscribing to their own mic tracks](https://community.livekit.io/t/participant-are-subscribing-to-their-own-mic-tracks/1428) — 2026-06-13
- [Does the rust SDK support wasm?](https://community.livekit.io/t/does-the-rust-sdk-support-wasm/1369) — 2026-06-08
- [rtpVideoPayloadTypes listener leak on publisher PCTransport during audio-only renegotiations](https://community.livekit.io/t/rtpvideopayloadtypes-listener-leak-on-publisher-pctransport-during-audio-only-renegotiations/1320) — 2026-06-02
- [SIGABRT crash while using client-sdk-android](https://community.livekit.io/t/sigabrt-crash-while-using-client-sdk-android/1316) — 2026-06-02
- [Speechmatics STT plugin always reports confidence=1.0 instead of provider confidence](https://community.livekit.io/t/speechmatics-stt-plugin-always-reports-confidence-1-0-instead-of-provider-confidence/1270) — 2026-05-28
- [Echo in the sound rooms between](https://community.livekit.io/t/echo-in-the-sound-rooms-between/1215) — 2026-05-24
- [Screen sharing issue on iOS 26](https://community.livekit.io/t/screen-sharing-issue-on-ios-26/1179) — 2026-05-20
- [Screen sharing on visionOS](https://community.livekit.io/t/screen-sharing-on-visionos/1159) — 2026-05-18
- [Whatsapp calls inbound](https://community.livekit.io/t/whatsapp-calls-inbound/1153) — 2026-05-18
- [Expo-callkit-telecom released: easily integrate CallKit/Core-Telecom with LiveKit on React Native](https://community.livekit.io/t/expo-callkit-telecom-released-easily-integrate-callkit-core-telecom-with-livekit-on-react-native/1145) — 2026-05-17
- [(Android SDK) Requesting PR review for reconnection stuck on server leave response](https://community.livekit.io/t/android-sdk-requesting-pr-review-for-reconnection-stuck-on-server-leave-response/1105) — 2026-05-13
- [Latest webrtc release is missing libwebrtc.jar](https://community.livekit.io/t/latest-webrtc-release-is-missing-libwebrtc-jar/1049) — 2026-05-07
- [inference.LLM crashes on openai@6.36.0+ due to empty-string apiKey at construction](https://community.livekit.io/t/inference-llm-crashes-on-openai-6-36-0-due-to-empty-string-apikey-at-construction/1029) — 2026-05-05
- [Memory leak in Google STT when no audio input exists](https://community.livekit.io/t/memory-leak-in-google-stt-when-no-audio-input-exists/1023) — 2026-05-04
- [Is there an MCP over LiveKit Cloud API?](https://community.livekit.io/t/is-there-an-mcp-over-livekit-cloud-api/990) — 2026-04-30
- [What is the reason for not allowing plugins to be imported in worker threads?](https://community.livekit.io/t/what-is-the-reason-for-not-allowing-plugins-to-be-imported-in-worker-threads/982) — 2026-04-29
- [LiveKit Tester With Realtime Debug Terminal](https://community.livekit.io/t/livekit-tester-with-realtime-debug-terminal/951) — 2026-04-25
- [React Native Voice Session Terminating in Background](https://community.livekit.io/t/react-native-voice-session-terminating-in-background/945) — 2026-04-23
- [TUN host lookup received error.,](https://community.livekit.io/t/tun-host-lookup-received-error/907) — 2026-04-20
- [Gemini 3.1 history_config](https://community.livekit.io/t/gemini-3-1-history-config/903) — 2026-04-18
- [Unreliable Connection when using VPN](https://community.livekit.io/t/unreliable-connection-when-using-vpn/900) — 2026-04-17
- [Rooms not being deleted automatically](https://community.livekit.io/t/rooms-not-being-deleted-automatically/874) — 2026-04-15
- [Memory leak in browser on screenshare](https://community.livekit.io/t/memory-leak-in-browser-on-screenshare/868) — 2026-04-15
- [5 Months since VP9 is broken in Flutter SDK](https://community.livekit.io/t/5-months-since-vp9-is-broken-in-flutter-sdk/737) — 2026-03-31
- [Noise Cancellation in Flutter causing Error for Android](https://community.livekit.io/t/noise-cancellation-in-flutter-causing-error-for-android/656) — 2026-03-24
- [Managing concurrent STT sessions](https://community.livekit.io/t/managing-concurrent-stt-sessions/652) — 2026-03-24
- [Krisp noise filter enters unrecoverable error state](https://community.livekit.io/t/krisp-noise-filter-enters-unrecoverable-error-state/645) — 2026-03-23
- [Bugfix to enable 'encryption' where currently only 'e2ee'](https://community.livekit.io/t/bugfix-to-enable-encryption-where-currently-only-e2ee/633) — 2026-03-20
- [Library not loaded: @rpath/LiveKitWebRTC.framework/LiveKitWebRTC](https://community.livekit.io/t/library-not-loaded-rpath-livekitwebrtc-framework-livekitwebrtc/611) — 2026-03-18
- [LiveKit Cloud GStreamer Pipeline Crashing with Connection Timeout (~4 min mark)](https://community.livekit.io/t/livekit-cloud-gstreamer-pipeline-crashing-with-connection-timeout-4-min-mark/533) — 2026-03-11
- [CORS error for Krisp NC](https://community.livekit.io/t/cors-error-for-krisp-nc/520) — 2026-03-10
- [401 ”token revoked” error on room rejoin after call disconnect (Android,iOS SDK)](https://community.livekit.io/t/401-token-revoked-error-on-room-rejoin-after-call-disconnect-android-ios-sdk/515) — 2026-03-10
- [How to feed custom video source (Snap Camera Kit) to LiveKit on Android?](https://community.livekit.io/t/how-to-feed-custom-video-source-snap-camera-kit-to-livekit-on-android/442) — 2026-03-03
- [Enabling Speaker After Manually Configuring Audio Session](https://community.livekit.io/t/enabling-speaker-after-manually-configuring-audio-session/441) — 2026-03-03
- [iOS SDK: Degraded audio after reconnect](https://community.livekit.io/t/ios-sdk-degraded-audio-after-reconnect/418) — 2026-02-28
- [Mirror video tracks and virtual backgrounds properly](https://community.livekit.io/t/mirror-video-tracks-and-virtual-backgrounds-properly/400) — 2026-02-26
- [Heads up: PR for better error reporting in agents-js for inference gateway connection failures](https://community.livekit.io/t/heads-up-pr-for-better-error-reporting-in-agents-js-for-inference-gateway-connection-failures/390) — 2026-02-25
- [Can the client check if the agent finished shutting down?](https://community.livekit.io/t/can-the-client-check-if-the-agent-finished-shutting-down/370) — 2026-02-23
- [Dangerous assistant turn merging with Gemini](https://community.livekit.io/t/dangerous-assistant-turn-merging-with-gemini/329) — 2026-02-19
- [ESP32 use same peer for publisher and subscriber role](https://community.livekit.io/t/esp32-use-same-peer-for-publisher-and-subscriber-role/267) — 2026-02-07
- [How to access audio buffers in the iOS SDK for VAD and volume detection](https://community.livekit.io/t/how-to-access-audio-buffers-in-the-ios-sdk-for-vad-and-volume-detection/143) — 2026-01-21
- [Can I rejoin a LiveKit room after refreshing the browser with the same token?](https://community.livekit.io/t/can-i-rejoin-a-livekit-room-after-refreshing-the-browser-with-the-same-token/140) — 2026-01-21
- [Best practices for LiveKit session management in React Native apps](https://community.livekit.io/t/best-practices-for-livekit-session-management-in-react-native-apps/130) — 2026-01-21
- [Best practice for linking appointment IDs to LiveKit room IDs](https://community.livekit.io/t/best-practice-for-linking-appointment-ids-to-livekit-room-ids/85) — 2026-01-21
- [How to send and receive text messages with the ESP32 SDK](https://community.livekit.io/t/how-to-send-and-receive-text-messages-with-the-esp32-sdk/73) — 2026-01-21
- [About the Client SDKs category](https://community.livekit.io/t/about-the-client-sdks-category/14) — 2025-12-22

## Server SDKs (21)

- [Livekit API Error](https://community.livekit.io/t/livekit-api-error/1397) — 2026-06-10
- [Server API SendData not delivered to Python on data_received using version 1.5.10](https://community.livekit.io/t/server-api-senddata-not-delivered-to-python-on-data-received-using-version-1-5-10/1175) — 2026-05-19
- [ElevenLabs pretty slow despite using inference](https://community.livekit.io/t/elevenlabs-pretty-slow-despite-using-inference/1132) — 2026-05-15
- [Native WhatsApp Calls integration with SIP](https://community.livekit.io/t/native-whatsapp-calls-integration-with-sip/1020) — 2026-05-04
- [Detect Direct Messages / Datapackets](https://community.livekit.io/t/detect-direct-messages-datapackets/912) — 2026-04-20
- [PHP Server SDK Update](https://community.livekit.io/t/php-server-sdk-update/879) — 2026-04-16
- [Include room metadata in webhook events](https://community.livekit.io/t/include-room-metadata-in-webhook-events/790) — 2026-04-07
- [Raised an PR solving issue 1179: gemini-3.1-live model support fix](https://community.livekit.io/t/raised-an-pr-solving-issue-1179-gemini-3-1-live-model-support-fix/738) — 2026-03-31
- [Request for reviewer](https://community.livekit.io/t/request-for-reviewer/581) — 2026-03-16
- [Please update Turn Detection Model](https://community.livekit.io/t/please-update-turn-detection-model/481) — 2026-03-06
- [Questions regarding forwarding participants](https://community.livekit.io/t/questions-regarding-forwarding-participants/440) — 2026-03-03
- [Agent Session Say Done Callback](https://community.livekit.io/t/agent-session-say-done-callback/364) — 2026-02-23
- [How to initialize and manage DB connections in a Livekit agent?](https://community.livekit.io/t/how-to-initialize-and-manage-db-connections-in-a-livekit-agent/340) — 2026-02-21
- [Support for multiple event callbacks to enable Node.js worker thread isolation](https://community.livekit.io/t/support-for-multiple-event-callbacks-to-enable-node-js-worker-thread-isolation/317) — 2026-02-17
- [Turn Detector Error: ”Turn detector does not support language english”](https://community.livekit.io/t/turn-detector-error-turn-detector-does-not-support-language-english/314) — 2026-02-17
- [Agent Session Say method with local audio files](https://community.livekit.io/t/agent-session-say-method-with-local-audio-files/274) — 2026-02-09
- [No logs. SENTRY_DSN configured](https://community.livekit.io/t/no-logs-sentry-dsn-configured/258) — 2026-02-06
- [Best practices for managing LiveKitAPI object in Python applications](https://community.livekit.io/t/best-practices-for-managing-livekitapi-object-in-python-applications/212) — 2026-01-21
- [How to prevent agent session from closing when participants leave](https://community.livekit.io/t/how-to-prevent-agent-session-from-closing-when-participants-leave/211) — 2026-01-21
- [Can I configure webhooks to only receive specific events?](https://community.livekit.io/t/can-i-configure-webhooks-to-only-receive-specific-events/210) — 2026-01-21
- [About the Server SDKs category](https://community.livekit.io/t/about-the-server-sdks-category/17) — 2025-12-22

## Cloud Dashboard (58)

- [How can I export Agent Insights to my own cloud provider?](https://community.livekit.io/t/how-can-i-export-agent-insights-to-my-own-cloud-provider/1535) — 2026-06-30
- [Admin unable to access Settings - Error ID 1151173688](https://community.livekit.io/t/admin-unable-to-access-settings-error-id-1151173688/1529) — 2026-06-29
- [Edit description option on existing API keys](https://community.livekit.io/t/edit-description-option-on-existing-api-keys/1496) — 2026-06-23
- [Call in progress bug](https://community.livekit.io/t/call-in-progress-bug/1493) — 2026-06-23
- [Programatic or MCP querying of data](https://community.livekit.io/t/programatic-or-mcp-querying-of-data/1338) — 2026-06-04
- [Session and room remains active even no room or agent deployment](https://community.livekit.io/t/session-and-room-remains-active-even-no-room-or-agent-deployment/1337) — 2026-06-04
- [Session inconsistency - requested room does not exist](https://community.livekit.io/t/session-inconsistency-requested-room-does-not-exist/1326) — 2026-06-03
- [Orphaned agent session remains “In progress” after deployment deletion (eu-central)](https://community.livekit.io/t/orphaned-agent-session-remains-in-progress-after-deployment-deletion-eu-central/1225) — 2026-05-25
- [Persistent emails saying limit reached on unused project](https://community.livekit.io/t/persistent-emails-saying-limit-reached-on-unused-project/1161) — 2026-05-18
- [Call transcript and audio not available](https://community.livekit.io/t/call-transcript-and-audio-not-available/1137) — 2026-05-15
- [(Billing Issue) Project stuck on ”Ship” plan after Scale upgrade](https://community.livekit.io/t/billing-issue-project-stuck-on-ship-plan-after-scale-upgrade/1130) — 2026-05-14
- [Cloud Agents: SIP inbound returns 503 before Call object created — public-side debug exhausted](https://community.livekit.io/t/cloud-agents-sip-inbound-returns-503-before-call-object-created-public-side-debug-exhausted/1065) — 2026-05-09
- [How do I disable the call analytics from storing data about the caller?](https://community.livekit.io/t/how-do-i-disable-the-call-analytics-from-storing-data-about-the-caller/1045) — 2026-05-06
- [Add --project flag to override default project context in CLI commands](https://community.livekit.io/t/add-project-flag-to-override-default-project-context-in-cli-commands/1028) — 2026-05-05
- [Unable to delete zombie session](https://community.livekit.io/t/unable-to-delete-zombie-session/1012) — 2026-05-03
- [Cloud session stuck ACTIVE while RoomService says room does not exist](https://community.livekit.io/t/cloud-session-stuck-active-while-roomservice-says-room-does-not-exist/955) — 2026-04-27
- [Noticable drop-pff in quality & stability](https://community.livekit.io/t/noticable-drop-pff-in-quality-stability/940) — 2026-04-23
- [I can't upgrade](https://community.livekit.io/t/i-cant-upgrade/935) — 2026-04-22
- [Issue with participant disconnect event](https://community.livekit.io/t/issue-with-participant-disconnect-event/880) — 2026-04-16
- [New project names cannot have spaces - Get error creating new project](https://community.livekit.io/t/new-project-names-cannot-have-spaces-get-error-creating-new-project/865) — 2026-04-14
- [Upgrade host specs on cloud livekit](https://community.livekit.io/t/upgrade-host-specs-on-cloud-livekit/813) — 2026-04-08
- [Using Custom Domain for SIP URI on LiveKit Cloud Instead of sip.livekit.cloud](https://community.livekit.io/t/using-custom-domain-for-sip-uri-on-livekit-cloud-instead-of-sip-livekit-cloud/786) — 2026-04-06
- [Billing issue my plan is not updated after payment](https://community.livekit.io/t/billing-issue-my-plan-is-not-updated-after-payment/760) — 2026-04-02
- [Enterprise IT Approval & E2EE Configuration Guidance](https://community.livekit.io/t/enterprise-it-approval-e2ee-configuration-guidance/750) — 2026-04-01
- [How long does it take to increase concurrency limit?](https://community.livekit.io/t/how-long-does-it-take-to-increase-concurrency-limit/699) — 2026-03-27
- [Agent Sessions not showing in list view](https://community.livekit.io/t/agent-sessions-not-showing-in-list-view/681) — 2026-03-25
- [LiveKit Cloud routes our users to Japan without VPN, while Germany 2 works with VPN — can we enable EU region pinning?](https://community.livekit.io/t/livekit-cloud-routes-our-users-to-japan-without-vpn-while-germany-2-works-with-vpn-can-we-enable-eu-region-pinning/665) — 2026-03-24
- [Coming from VAPI - concurrent calls question](https://community.livekit.io/t/coming-from-vapi-concurrent-calls-question/638) — 2026-03-21
- [【Billing Help】Accidentally deleted project with active Ship plan](https://community.livekit.io/t/billing-help-accidentally-deleted-project-with-active-ship-plan/627) — 2026-03-20
- [LiveKit Cloud session remains ACTIVE in Sessions dashboard even after room is gone from RoomService API](https://community.livekit.io/t/livekit-cloud-session-remains-active-in-sessions-dashboard-even-after-room-is-gone-from-roomservice-api/617) — 2026-03-19
- [LiveKit Cloud is likely classifying a normal idle SIGTERM shutdown of the cloud hosted Node agent as Error](https://community.livekit.io/t/livekit-cloud-is-likely-classifying-a-normal-idle-sigterm-shutdown-of-the-cloud-hosted-node-agent-as-error/612) — 2026-03-18
- [Unable to Access Company Workspace After Team Invitation](https://community.livekit.io/t/unable-to-access-company-workspace-after-team-invitation/514) — 2026-03-10
- [Transfer Subscription Between Projects](https://community.livekit.io/t/transfer-subscription-between-projects/500) — 2026-03-09
- [Sessions stuck in In progress](https://community.livekit.io/t/sessions-stuck-in-in-progress/489) — 2026-03-07
- [LiveKit Cloud IP Whitelisting](https://community.livekit.io/t/livekit-cloud-ip-whitelisting/484) — 2026-03-06
- [Observability EU](https://community.livekit.io/t/observability-eu/436) — 2026-03-02
- [Wrong data presented in session](https://community.livekit.io/t/wrong-data-presented-in-session/401) — 2026-02-26
- [What version of Livekit Cloud is my sever?](https://community.livekit.io/t/what-version-of-livekit-cloud-is-my-sever/392) — 2026-02-25
- [LiveKit Cloud Agent Draining](https://community.livekit.io/t/livekit-cloud-agent-draining/380) — 2026-02-24
- [Unable to login](https://community.livekit.io/t/unable-to-login/341) — 2026-02-21
- [Missing webhook requests](https://community.livekit.io/t/missing-webhook-requests/319) — 2026-02-18
- [API to access transcripts and recordings](https://community.livekit.io/t/api-to-access-transcripts-and-recordings/303) — 2026-02-15
- [Elevenlabs ”other voice ID” broken](https://community.livekit.io/t/elevenlabs-other-voice-id-broken/283) — 2026-02-11
- [Can I access 6 months of session data via the Analytics API?](https://community.livekit.io/t/can-i-access-6-months-of-session-data-via-the-analytics-api/207) — 2026-01-21
- [Do I need to upgrade each project separately on LiveKit Cloud?](https://community.livekit.io/t/do-i-need-to-upgrade-each-project-separately-on-livekit-cloud/202) — 2026-01-21
- [How frequently is the billing dashboard updated?](https://community.livekit.io/t/how-frequently-is-the-billing-dashboard-updated/198) — 2026-01-21
- [Can I whitelist IP addresses for LiveKit Cloud webhooks?](https://community.livekit.io/t/can-i-whitelist-ip-addresses-for-livekit-cloud-webhooks/190) — 2026-01-21
- [Is there downtime when deploying agent updates to LiveKit Cloud?](https://community.livekit.io/t/is-there-downtime-when-deploying-agent-updates-to-livekit-cloud/176) — 2026-01-21
- [Agent creation fails with 'failed to Lchown Dockerfile' error](https://community.livekit.io/t/agent-creation-fails-with-failed-to-lchown-dockerfile-error/167) — 2026-01-21
- [Can I change the UDP port for LiveKit Cloud?](https://community.livekit.io/t/can-i-change-the-udp-port-for-livekit-cloud/166) — 2026-01-21
- [How to specify a region for SIP ingress and room hosting](https://community.livekit.io/t/how-to-specify-a-region-for-sip-ingress-and-room-hosting/160) — 2026-01-21
- [Does the 5 concurrency limit apply to self-hosted agents?](https://community.livekit.io/t/does-the-5-concurrency-limit-apply-to-self-hosted-agents/135) — 2026-01-21
- [How to end an active LiveKit room session from the dashboard](https://community.livekit.io/t/how-to-end-an-active-livekit-room-session-from-the-dashboard/128) — 2026-01-21
- [What is the delay and reliability of LiveKit webhooks?](https://community.livekit.io/t/what-is-the-delay-and-reliability-of-livekit-webhooks/95) — 2026-01-21
- [Agent deployment error: 'package @livekit/agents not found' for Python agent](https://community.livekit.io/t/agent-deployment-error-package-livekit-agents-not-found-for-python-agent/82) — 2026-01-21
- [Is Krisp noise cancellation available with LiveKit Cloud?](https://community.livekit.io/t/is-krisp-noise-cancellation-available-with-livekit-cloud/78) — 2026-01-21
- [LiveKit Cloud billing and concurrency limits for agents](https://community.livekit.io/t/livekit-cloud-billing-and-concurrency-limits-for-agents/75) — 2026-01-21
- [About the Cloud Dashboard category](https://community.livekit.io/t/about-the-cloud-dashboard-category/18) — 2025-12-22

## Egress / Ingress (32)

- [Quotas and limits(egress + agents)](https://community.livekit.io/t/quotas-and-limits-egress-agents/1455) — 2026-06-18
- [Egress track synchronization](https://community.livekit.io/t/egress-track-synchronization/1423) — 2026-06-12
- [Black border in the right side](https://community.livekit.io/t/black-border-in-the-right-side/1352) — 2026-06-05
- [Room-composite egress completes (”Source closed”) before the room is deleted](https://community.livekit.io/t/room-composite-egress-completes-source-closed-before-the-room-is-deleted/1291) — 2026-05-31
- [Room Composite API returning 'context deadline exceed' and 404 for a room the agent is actively connected to](https://community.livekit.io/t/room-composite-api-returning-context-deadline-exceed-and-404-for-a-room-the-agent-is-actively-connected-to/1271) — 2026-05-28
- [LiveKit Egress S3 upload fails with 'ExpiredToken' after ~80 min call](https://community.livekit.io/t/livekit-egress-s3-upload-fails-with-expiredtoken-after-80-min-call/1261) — 2026-05-27
- [Egress_ended webhook not delivered for some Room Composite Egress jobs](https://community.livekit.io/t/egress-ended-webhook-not-delivered-for-some-room-composite-egress-jobs/1090) — 2026-05-12
- [Egress recording contains echo on Agent Participant audio, but echo was not present in real call](https://community.livekit.io/t/egress-recording-contains-echo-on-agent-participant-audio-but-echo-was-not-present-in-real-call/1050) — 2026-05-07
- [Egress upload retry policy](https://community.livekit.io/t/egress-upload-retry-policy/979) — 2026-04-28
- [Audio Enhancement on Room Recording - is it Raw or Processed?](https://community.livekit.io/t/audio-enhancement-on-room-recording-is-it-raw-or-processed/931) — 2026-04-22
- [Livekit video is lagging for the first 0.5 seconds](https://community.livekit.io/t/livekit-video-is-lagging-for-the-first-0-5-seconds/925) — 2026-04-21
- [Participant Id valid, but egress says not found](https://community.livekit.io/t/participant-id-valid-but-egress-says-not-found/922) — 2026-04-21
- [Multiple issues (or may be understanding related) however mainly related to engress](https://community.livekit.io/t/multiple-issues-or-may-be-understanding-related-however-mainly-related-to-engress/920) — 2026-04-21
- [Aligning individual tracks](https://community.livekit.io/t/aligning-individual-tracks/905) — 2026-04-20
- [Egress recording has ~1 min blank screen at the end](https://community.livekit.io/t/egress-recording-has-1-min-blank-screen-at-the-end/892) — 2026-04-16
- [I keep getting mp3 audio files with the wrong duration](https://community.livekit.io/t/i-keep-getting-mp3-audio-files-with-the-wrong-duration/834) — 2026-04-10
- [Unable to kill zombie session](https://community.livekit.io/t/unable-to-kill-zombie-session/761) — 2026-04-02
- [Agent turn missing in composite egress](https://community.livekit.io/t/agent-turn-missing-in-composite-egress/695) — 2026-03-27
- [Error during egress request](https://community.livekit.io/t/error-during-egress-request/650) — 2026-03-24
- [Drift between audio and video in the participant recording egress livekit cloud](https://community.livekit.io/t/drift-between-audio-and-video-in-the-participant-recording-egress-livekit-cloud/648) — 2026-03-23
- [Egress Crackling issue while recording individual participants from the room](https://community.livekit.io/t/egress-crackling-issue-while-recording-individual-participants-from-the-room/332) — 2026-02-19
- [How to estimate billing for video](https://community.livekit.io/t/how-to-estimate-billing-for-video/262) — 2026-02-06
- [Getting egress filename when recording completes automatically](https://community.livekit.io/t/getting-egress-filename-when-recording-completes-automatically/206) — 2026-01-21
- [Room composite recording shows black screen when no video tracks are active](https://community.livekit.io/t/room-composite-recording-shows-black-screen-when-no-video-tracks-are-active/204) — 2026-01-21
- [Recommended audio format for egress recordings (OGG vs MP3)](https://community.livekit.io/t/recommended-audio-format-for-egress-recordings-ogg-vs-mp3/203) — 2026-01-21
- [How to get room audio recording in a local folder after session ends](https://community.livekit.io/t/how-to-get-room-audio-recording-in-a-local-folder-after-session-ends/189) — 2026-01-21
- [Request-specific egress webhooks not working for self-hosted egress](https://community.livekit.io/t/request-specific-egress-webhooks-not-working-for-self-hosted-egress/178) — 2026-01-21
- [DUAL_CHANNEL_AGENT egress not producing dual channel output](https://community.livekit.io/t/dual-channel-agent-egress-not-producing-dual-channel-output/156) — 2026-01-21
- [How to set up Egress for call recordings in LiveKit Cloud](https://community.livekit.io/t/how-to-set-up-egress-for-call-recordings-in-livekit-cloud/125) — 2026-01-21
- [How to get the exact timestamp for when a video recording started](https://community.livekit.io/t/how-to-get-the-exact-timestamp-for-when-a-video-recording-started/112) — 2026-01-21
- [Does egress complete or fail when end reason is 'source closed'?](https://community.livekit.io/t/does-egress-complete-or-fail-when-end-reason-is-source-closed/72) — 2026-01-21
- [About the Egress / Ingress category](https://community.livekit.io/t/about-the-egress-ingress-category/13) — 2025-12-22

## Robotics (6)

- [Payload type filtering](https://community.livekit.io/t/payload-type-filtering/1521) — 2026-06-27
- [Our robotics RTC stack, before and after LiveKit C++](https://community.livekit.io/t/our-robotics-rtc-stack-before-and-after-livekit-c/1362) — 2026-06-07
- [Custom hardware support: ES8311 (DAC) + ES7243E (ADC) on shared I2C/I2S bus — init sequence incompatibility and dual-codec guidance needed](https://community.livekit.io/t/custom-hardware-support-es8311-dac-es7243e-adc-on-shared-i2c-i2s-bus-init-sequence-incompatibility-and-dual-codec-guidance-needed/498) — 2026-03-09
- [Physical AI Meetup - Robotics at Home](https://community.livekit.io/t/physical-ai-meetup-robotics-at-home/265) — 2026-02-06
- [What is LiveKit robotics used for?](https://community.livekit.io/t/what-is-livekit-robotics-used-for/159) — 2026-01-21
- [About the Robotics category](https://community.livekit.io/t/about-the-robotics-category/19) — 2025-12-22

## Self Hosting (69)

- [Guest connection dropping with DTLS timeouts after recent LiveKit Server Update](https://community.livekit.io/t/guest-connection-dropping-with-dtls-timeouts-after-recent-livekit-server-update/1503) — 2026-06-24
- [Request for Recommended LiveKit Deployment Architecture (Private Networking)](https://community.livekit.io/t/request-for-recommended-livekit-deployment-architecture-private-networking/1500) — 2026-06-24
- [LiveKit v1.13.1: Signaling WS drops mid-call with CLIENT_REQUEST_LEAVE + full reconnect (media UDP unaffected) behind Nginx ssl_preread](https://community.livekit.io/t/livekit-v1-13-1-signaling-ws-drops-mid-call-with-client-request-leave-full-reconnect-media-udp-unaffected-behind-nginx-ssl-preread/1481) — 2026-06-22
- [OpenAI TTS Plugin Returns “no audio frames were pushed” with OpenAI-Compatible Endpoints (Kokoro + DeepInfra)](https://community.livekit.io/t/openai-tts-plugin-returns-no-audio-frames-were-pushed-with-openai-compatible-endpoints-kokoro-deepinfra/1465) — 2026-06-19
- [Co-located egress uses the TURN relay instead of the SFU's advertised loopback candidate → corrupt HLS](https://community.livekit.io/t/co-located-egress-uses-the-turn-relay-instead-of-the-sfus-advertised-loopback-candidate-corrupt-hls/1437) — 2026-06-15
- [Inbound SIP — sip.h. header attributes intermittently missing when agent reads participant attributes*](https://community.livekit.io/t/inbound-sip-sip-h-header-attributes-intermittently-missing-when-agent-reads-participant-attributes/1421) — 2026-06-12
- [Slow room creation](https://community.livekit.io/t/slow-room-creation/1358) — 2026-06-06
- [Whatsapp Connectors](https://community.livekit.io/t/whatsapp-connectors/1357) — 2026-06-06
- [Is it necessary to upgrade version of self hosted livekit server when updating livekit python sdk](https://community.livekit.io/t/is-it-necessary-to-upgrade-version-of-self-hosted-livekit-server-when-updating-livekit-python-sdk/1328) — 2026-06-03
- [High end-to-end latency: ASR+LLM+TTS total <1s, but speech end to reply takes 3-4s (LiveKit Agents)](https://community.livekit.io/t/high-end-to-end-latency-asr-llm-tts-total-1s-but-speech-end-to-reply-takes-3-4s-livekit-agents/1277) — 2026-05-29
- [Is audio screen capture through screen sharing possible?](https://community.livekit.io/t/is-audio-screen-capture-through-screen-sharing-possible/1269) — 2026-05-28
- [Noise Cancelling Features with Self-Hosted Agents](https://community.livekit.io/t/noise-cancelling-features-with-self-hosted-agents/1227) — 2026-05-25
- [Livestreaming is laghing on viewers device](https://community.livekit.io/t/livestreaming-is-laghing-on-viewers-device/1218) — 2026-05-24
- [Livekit server taking longer time to cool down](https://community.livekit.io/t/livekit-server-taking-longer-time-to-cool-down/1101) — 2026-05-12
- [High Latency Between LiveKit Room and RTMP Egress Stream](https://community.livekit.io/t/high-latency-between-livekit-room-and-rtmp-egress-stream/1079) — 2026-05-11
- [Livekit Stopped Connecting today](https://community.livekit.io/t/livekit-stopped-connecting-today/1076) — 2026-05-11
- [Any plans for a self-hostable LiveKit dashboard?](https://community.livekit.io/t/any-plans-for-a-self-hostable-livekit-dashboard/1046) — 2026-05-06
- [Start/Stop recording Latency in LiveKit Rooms using Egress](https://community.livekit.io/t/start-stop-recording-latency-in-livekit-rooms-using-egress/1040) — 2026-05-06
- [Redis Settings Ignored in Livekit Server](https://community.livekit.io/t/redis-settings-ignored-in-livekit-server/1000) — 2026-05-01
- [Inbound calls AuthDrop'd as ”flood” at ~85% rate from single legitimate carrier IP — config knob?](https://community.livekit.io/t/inbound-calls-authdropd-as-flood-at-85-rate-from-single-legitimate-carrier-ip-config-knob/999) — 2026-05-01
- [Questions about screen sharing on the network](https://community.livekit.io/t/questions-about-screen-sharing-on-the-network/986) — 2026-04-29
- [Questions on Recording & Noise Cancellation in Self-Hosted LiveKit Setup](https://community.livekit.io/t/questions-on-recording-noise-cancellation-in-self-hosted-livekit-setup/968) — 2026-04-28
- [Glitches, Crickly noise, chopping, some time voice goes away](https://community.livekit.io/t/glitches-crickly-noise-chopping-some-time-voice-goes-away/957) — 2026-04-27
- [Flutter SDK chat messages incompatible with web 'useChat' protocol](https://community.livekit.io/t/flutter-sdk-chat-messages-incompatible-with-web-usechat-protocol/928) — 2026-04-22
- [Virtual avatar is displayed when using livekit cloud but not displayed in localhost](https://community.livekit.io/t/virtual-avatar-is-displayed-when-using-livekit-cloud-but-not-displayed-in-localhost/926) — 2026-04-22
- [SIP participant stuck in infinite ICE restart loop](https://community.livekit.io/t/sip-participant-stuck-in-infinite-ice-restart-loop/897) — 2026-04-17
- [Self-hosted SIP server with LiveKit Cloud – is this supported?](https://community.livekit.io/t/self-hosted-sip-server-with-livekit-cloud-is-this-supported/869) — 2026-04-15
- [Getting 701 ICE candidate error](https://community.livekit.io/t/getting-701-ice-candidate-error/793) — 2026-04-07
- [I added a new 'Ask AI' button to the community](https://community.livekit.io/t/i-added-a-new-ask-ai-button-to-the-community/763) — 2026-04-02
- [DTLS timeout after ~10s with TURN/TCP in multi-node setup (v1.9.12)](https://community.livekit.io/t/dtls-timeout-after-10s-with-turn-tcp-in-multi-node-setup-v1-9-12/748) — 2026-04-01
- [Increase maximum number of concurrency](https://community.livekit.io/t/increase-maximum-number-of-concurrency/684) — 2026-03-26
- [Outbound SIP call: agent greeting never heard by callee (self-hosted v1.9.12)](https://community.livekit.io/t/outbound-sip-call-agent-greeting-never-heard-by-callee-self-hosted-v1-9-12/642) — 2026-03-23
- [Any plan to open source the adaptive interrupt handling model?](https://community.livekit.io/t/any-plan-to-open-source-the-adaptive-interrupt-handling-model/629) — 2026-03-20
- [Best Platform To Self Host](https://community.livekit.io/t/best-platform-to-self-host/614) — 2026-03-18
- [SIP<>Airtel Outbound Calling](https://community.livekit.io/t/sip-airtel-outbound-calling/546) — 2026-03-13
- [Server crashing](https://community.livekit.io/t/server-crashing/543) — 2026-03-13
- [Warm Transfer Implementation](https://community.livekit.io/t/warm-transfer-implementation/540) — 2026-03-12
- [Best way to scale LiveKit Egress for recordings (private meetings + livestream platform)?](https://community.livekit.io/t/best-way-to-scale-livekit-egress-for-recordings-private-meetings-livestream-platform/537) — 2026-03-12
- [Autoscaling Strategy for Self-Hosted LiveKit Egress Workers in Real-Time Streaming Platform](https://community.livekit.io/t/autoscaling-strategy-for-self-hosted-livekit-egress-workers-in-real-time-streaming-platform/508) — 2026-03-10
- [Issues with integrating with Avatar locally (self-hosted)](https://community.livekit.io/t/issues-with-integrating-with-avatar-locally-self-hosted/506) — 2026-03-09
- [Egress service is ready but roomcompositeegress() from agent.py throws error (livekit.api.twirp_client.TwirpError: TwirpError(code=unavailable, message=twirp error unknown: no response from servers, status=503))](https://community.livekit.io/t/egress-service-is-ready-but-roomcompositeegress-from-agent-py-throws-error-livekit-api-twirp-client-twirperror-twirperror-code-unavailable-message-twirp-error-unknown-no-response-from-servers-status-503/487) — 2026-03-06
- [Latest Livekit server image with 'Latest' tag is having vulnerabilities. When and can we get the new image with all fixes](https://community.livekit.io/t/latest-livekit-server-image-with-latest-tag-is-having-vulnerabilities-when-and-can-we-get-the-new-image-with-all-fixes/470) — 2026-03-05
- [Introducing livekit-monitor - monitoring dashboard for self hosted instances](https://community.livekit.io/t/introducing-livekit-monitor-monitoring-dashboard-for-self-hosted-instances/445) — 2026-03-04
- [Is self-hosting LiveKit cost-effective at scale](https://community.livekit.io/t/is-self-hosting-livekit-cost-effective-at-scale/403) — 2026-02-26
- [Sip Contact Header Not being Set correctly](https://community.livekit.io/t/sip-contact-header-not-being-set-correctly/373) — 2026-02-24
- [Livekit K8s Deployment + Haproxy L4 passthrough for all TCP and UDP request via Haproxy](https://community.livekit.io/t/livekit-k8s-deployment-haproxy-l4-passthrough-for-all-tcp-and-udp-request-via-haproxy/372) — 2026-02-24
- [Self Hosted SIP Latency Issue](https://community.livekit.io/t/self-hosted-sip-latency-issue/302) — 2026-02-14
- [Egress service gets stuck](https://community.livekit.io/t/egress-service-gets-stuck/298) — 2026-02-13
- [Livekit with ECS Fargate](https://community.livekit.io/t/livekit-with-ecs-fargate/297) — 2026-02-13
- [How to integrate LiveKit self-hosted SIP with Airtel Enterprise LAN trunk?](https://community.livekit.io/t/how-to-integrate-livekit-self-hosted-sip-with-airtel-enterprise-lan-trunk/289) — 2026-02-12
- [LiveKit SIP omitting the SDP connection address in mid-dialog messages](https://community.livekit.io/t/livekit-sip-omitting-the-sdp-connection-address-in-mid-dialog-messages/281) — 2026-02-10
- [Multiple rooms reconnection fails](https://community.livekit.io/t/multiple-rooms-reconnection-fails/271) — 2026-02-09
- [Support for MoveParticipant in self-hosted LiveKit](https://community.livekit.io/t/support-for-moveparticipant-in-self-hosted-livekit/245) — 2026-02-03
- [How to advertise both private and public IPs as ICE candidates in Kubernetes](https://community.livekit.io/t/how-to-advertise-both-private-and-public-ips-as-ice-candidates-in-kubernetes/242) — 2026-01-21
- [Cannot connect to LiveKit server from another machine on the network](https://community.livekit.io/t/cannot-connect-to-livekit-server-from-another-machine-on-the-network/241) — 2026-01-21
- [Inbound SIP calls from Asterisk keep ringing - no 200 OK sent](https://community.livekit.io/t/inbound-sip-calls-from-asterisk-keep-ringing-no-200-ok-sent/240) — 2026-01-21
- [LiveKit agent fails to connect to self-hosted server - upgrade required](https://community.livekit.io/t/livekit-agent-fails-to-connect-to-self-hosted-server-upgrade-required/239) — 2026-01-21
- [High latency (5-8 seconds) with Google Gemini Realtime plugin over SIP](https://community.livekit.io/t/high-latency-5-8-seconds-with-google-gemini-realtime-plugin-over-sip/238) — 2026-01-21
- [Agent connects to LiveKit Cloud instead of local self-hosted server](https://community.livekit.io/t/agent-connects-to-livekit-cloud-instead-of-local-self-hosted-server/237) — 2026-01-21
- [Troubleshooting WebRTC connection failures with Caddy reverse proxy](https://community.livekit.io/t/troubleshooting-webrtc-connection-failures-with-caddy-reverse-proxy/236) — 2026-01-21
- [Agent fails to connect with 401 error - API key/secret mismatch](https://community.livekit.io/t/agent-fails-to-connect-with-401-error-api-key-secret-mismatch/235) — 2026-01-21
- [Running LiveKit on private Kubernetes clusters - workarounds and recommendations](https://community.livekit.io/t/running-livekit-on-private-kubernetes-clusters-workarounds-and-recommendations/234) — 2026-01-21
- [Do I need a global Redis instance for multi-region LiveKit deployments?](https://community.livekit.io/t/do-i-need-a-global-redis-instance-for-multi-region-livekit-deployments/233) — 2026-01-21
- [Where to find documentation for self-hosting LiveKit server](https://community.livekit.io/t/where-to-find-documentation-for-self-hosting-livekit-server/232) — 2026-01-21
- [SIP calls rejected with 'flood' error - how to whitelist provider IPs](https://community.livekit.io/t/sip-calls-rejected-with-flood-error-how-to-whitelist-provider-ips/231) — 2026-01-21
- [Running multiple named agents with explicit dispatching in a single container](https://community.livekit.io/t/running-multiple-named-agents-with-explicit-dispatching-in-a-single-container/230) — 2026-01-21
- [How to configure LiveKit on GCP without public IPs for VMs](https://community.livekit.io/t/how-to-configure-livekit-on-gcp-without-public-ips-for-vms/229) — 2026-01-21
- [TURN server connectivity test fails with IP-based TLS certificate](https://community.livekit.io/t/turn-server-connectivity-test-fails-with-ip-based-tls-certificate/172) — 2026-01-21
- [About the Self Hosting category](https://community.livekit.io/t/about-the-self-hosting-category/15) — 2025-12-22
</user_query>