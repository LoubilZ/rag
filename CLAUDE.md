# CLAUDE.md — RAG Voice Agent

Contexte pour les sessions sur ce repo.

## Quoi

Serveur MCP FastMCP + Qdrant Cloud (`livekit_kb`, 512 dims, Voyage `voyage-3-lite`). Outils : `search_livekit_kb`, `livekit_kb_stats`. Ingest HTTP : `/ingest`, `/ingest/prompting`.

## Pourquoi

Base d'**amélioration continue du prompting** d'agents vocaux — pas seulement de la doc LiveKit infra.

## Tags

- **category** : `prompting` | `infra`
- **architecture** : `cascaded` (STT→LLM→TTS) | `s2s` (realtime) | `both` (matche cascaded et s2s en recherche)
- **source_type** : extensible ; legacy = `docs`, `forum`, `vapi_book` ; prompting = `livekit_prompting`, `openai_realtime`, `vapi_prompting`, `hamming_identity`, `anthropic_tools`

Legacy corpus (~10k points) : pas de category/architecture tant qu'on n'a pas rétro-taggé.

## Ingest prompting (5 sources)

```bash
curl -X POST https://livekit-rag-production.up.railway.app/ingest/prompting
```

HTML → Jina Reader (`https://r.jina.ai/...`). Markdown → fetch direct.

## Contraintes prod

- **Ne jamais** `delete_collection` / recréer la collection sur Qdrant Cloud.
- Même modèle embedding, 512 dims.
- IDs déterministes : `sha256(url:chunk_index)` pour upsert idempotent.

## Recherche filtrée (exemple)

```python
search_livekit_kb(
    query="caller identity testing checklist",
    category="prompting",
    architecture="s2s",
    top_k=5,
)
```
