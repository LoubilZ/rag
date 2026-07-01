# LiveKit RAG MCP Server

Serveur MCP qui expose une base de connaissances LiveKit (docs, forum, Vapi Playbook) via recherche vectorielle + reranking Cohere.

**Production :** https://livekit-rag-production.up.railway.app

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

`source_type` : `docs` | `forum` | `vapi_book`

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
            "query": "How do I detect turns?",
            "top_k": 5,
            "source_type": "docs"
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
