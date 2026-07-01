# LiveKit RAG MCP Server

Serveur MCP (RAG-Voice Agent) : base de connaissances sur les agents vocaux, recherche vectorielle (Voyage 512d) + reranking Cohere, stockage Qdrant.

**Production :** https://livekit-rag-production.up.railway.app

## Rôle du repo

Ce RAG sert à **améliorer en continu le prompting** d'agents vocaux : docs LiveKit, guides OpenAI Realtime, Vapi, Hamming, Anthropic, etc. Les chunks récents portent des tags `category` et `architecture` pour filtrer les recherches selon le type de pipeline (cascadé STT→LLM→TTS vs speech-to-speech).

### Schéma de métadonnées (payload Qdrant)

| Champ | Valeurs | Description |
|-------|---------|-------------|
| `source_type` | `docs`, `forum`, `vapi_book`, `livekit_prompting`, `openai_realtime`, `vapi_prompting`, `hamming_identity`, `anthropic_tools`, … | Origine du document (extensible) |
| `category` | `prompting`, `infra` | Type de contenu (prompting = guides de prompts ; infra = déploiement, plateforme) |
| `architecture` | `cascaded`, `s2s`, `both` | Pipeline visé ; `both` matche les filtres `cascaded` et `s2s` |

Les ~10k points legacy n'ont pas encore `category`/`architecture` — seuls les nouveaux chunks ingérés sont filtrables par ces champs.

### Sources prompting curatées (5)

| URL | source_type | architecture |
|-----|-------------|--------------|
| docs.livekit.io/agents/start/prompting.md | `livekit_prompting` | both |
| developers.openai.com/.../realtime_prompting_guide | `openai_realtime` | s2s |
| docs.vapi.ai/prompting-guide.md | `vapi_prompting` | both |
| hamming.ai/.../voice-agent-caller-identity-testing-checklist | `hamming_identity` | both |
| anthropic.com/engineering/writing-tools-for-agents | `anthropic_tools` | both |

Ingestion en une commande :

```bash
curl -X POST https://livekit-rag-production.up.railway.app/ingest/prompting
```

## Setup local

```bash
cp .env.example .env
# Renseigner VOYAGE_API_KEY et COHERE_API_KEY

pip install -r requirements.txt
python mcp_server.py
```

Le serveur écoute sur `http://0.0.0.0:8080` (MCP sur `/mcp`, ingest sur `/ingest`).

## Variables d'environnement

| Variable | Requis | Description |
|----------|--------|-------------|
| `VOYAGE_API_KEY` | Oui | Embeddings Voyage AI (`voyage-3-lite`, 512 dims) |
| `COHERE_API_KEY` | Oui | Reranking (`rerank-3.5`) |
| `PORT` | Non | Port HTTP (défaut `8080`) |
| `QDRANT_URL` | Non | URL Qdrant Cloud — persistance en prod |
| `QDRANT_API_KEY` | Non | Clé API Qdrant Cloud |

Sans `QDRANT_URL`, Qdrant tourne en local dans `qdrant_storage/`. **Sur Railway, ce dossier est effacé à chaque redeploy** — il faut relancer l'ingest ou brancher Qdrant Cloud.

## Ingest (ajouter des docs)

```bash
curl -X POST https://livekit-rag-production.up.railway.app/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://docs.livekit.io"],
    "source_type": "docs"
  }'
```

`source_type` : libre (ex. `docs`, `forum`, `vapi_book`, `livekit_prompting`, …)

Champs optionnels : `category`, `architecture`, `fetch_mode` (`auto`, `markdown`, `jina`, `html`).

Corps étendu (métadonnées par URL) :

```bash
curl -X POST https://livekit-rag-production.up.railway.app/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "sources": [{
      "url": "https://docs.livekit.io/agents/start/prompting.md",
      "source_type": "livekit_prompting",
      "category": "prompting",
      "architecture": "both",
      "fetch_mode": "markdown"
    }]
  }'
```

Réponse attendue :

```json
{
  "status": "success",
  "ingested_count": 9,
  "urls_processed": 1,
  "errors": null
}
```

## Recherche (outil MCP)

La recherche ne passe **pas** par un simple REST `/sse`. Elle utilise l'outil MCP `search_livekit_kb` sur `/mcp`.

### Exemple Python

```python
import requests

base = "https://livekit-rag-production.up.railway.app/mcp"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

# 1. Initialiser la session MCP
init = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "my-client", "version": "1.0"},
    },
}
r = requests.post(base, headers=headers, json=init)
headers["mcp-session-id"] = r.headers["mcp-session-id"]

# 2. Appeler search_livekit_kb
call = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "search_livekit_kb",
        "arguments": {
            "query": "How should I write the system prompt for turn detection?",
            "top_k": 5,
            "category": "prompting",
            "architecture": "s2s",
            "source_type": "openai_realtime"
        },
    },
}
print(requests.post(base, headers=headers, json=call).text)
```

### Outils MCP disponibles

| Outil | Description |
|-------|-------------|
| `search_livekit_kb` | Recherche sémantique dans la KB |
| `livekit_kb_stats` | Stats de la collection (`points_count`, statut) |

## Qdrant Cloud (persistance Railway)

1. Créer un cluster sur [cloud.qdrant.io](https://cloud.qdrant.io)
2. Ajouter dans Railway :
   - `QDRANT_URL=https://xxxx.aws.cloud.qdrant.io:6333`
   - `QDRANT_API_KEY=...`
3. Redéployer, puis lancer l'ingest une fois

Les données survivront aux redeploys.

## Docker

```bash
docker build -t livekit-rag .
docker run -p 8080:8080 --env-file .env livekit-rag
```

## Configurer dans Cursor / Claude

Ajouter le serveur MCP en **Streamable HTTP** :

```
https://livekit-rag-production.up.railway.app/mcp
```
