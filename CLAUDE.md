# jarvis_v2 — Claude Code Context

## Project
LangGraph-native autonomous revenue system for FiestyGoat AI LLC.
Replaces jarvis_v1 (personal-agent-hub repo).

## Architecture
- **Tier 1**: `jarvis/workers/` — ContinuousWorker subclasses, NO LangGraph, pure Redis poll
- **Tier 2**: `jarvis/graphs/` — LangGraph StateGraph decision workflows
- **Tier 3**: `jarvis/agents/` — Specialist agent subgraphs (composable)
- **Core**: `jarvis/core/` — LLMRouter, AutoMem, Neo4j GraphMemory, ConfidenceGate

## Key Rules
- Tier 1 workers NEVER call LangGraph — raw Python + Redis only
- All LLM calls go through `LLMRouter` (Ollama first, Claude fallback)
- All memory writes go through `AutoMem` (pgvector) OR `GraphMemory` (Neo4j)
- All decisions go through `ConfidenceGate` (4-layer scoring)
- Production-quality code only — enterprise error handling

## Database
- PostgreSQL: `postgresql://localhost/personal_agent_hub` (pgvector 0.8.1)
- Neo4j: `bolt://localhost:7687` (v2026.01.4)
- Redis: `redis://localhost:6379`

## Python
- Version: 3.13
- Venv: `venv/`
- Package manager: pip with pyproject.toml
