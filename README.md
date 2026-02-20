# jarvis_v2

FiestyGoat AI LLC — LangGraph-native autonomous revenue generation system.

## Architecture

Three-tier LangGraph-native design:

```
Tier 1: Always-On Workers (supervisord → ContinuousWorker → Redis)
  ├── LearningWorker    — 30+ RSS/arXiv/Reddit sources, 24/7
  ├── ResearchWorker    — AI breakthrough detection, 24/7
  ├── HealthWorker      — Infrastructure monitoring, 24/7
  └── TradingWorker     — Market execution engine, 24/7

Tier 2: LangGraph Decision Graphs (StateGraph + checkpoints)
  ├── revenue_graph     — Opportunity → evaluate → execute/delegate
  ├── trading_graph     — Signal → risk validate → execute/skip
  ├── content_graph     — Content quality gate
  └── confidence_graph  — 4-layer confidence scoring

Tier 3: Specialist Agent Subgraphs (composable, reusable)
  ├── InstagramAgent, YouTubeAgent, NewsletterAgent
  ├── SEOBlogAgent, TikTokAgent, TwitterAgent
  └── TradingAgent
```

## Infrastructure

- **LLM**: Ollama (qwen2.5:0.5b speed / deepseek-r1:7b reasoning), $0 cost
- **Memory**: PostgreSQL + pgvector (AutoMem), Neo4j (knowledge graph)
- **Coordination**: Redis pub/sub + task queues
- **Trading**: Alpaca Paper Trading API

## Setup

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your settings
./scripts/setup.sh
```

## Running

```bash
# All workers via supervisor
supervisorctl -c config/supervisor.conf start all

# Individual worker
python -m jarvis.workers.learning

# API server
uvicorn jarvis.api.server:app --reload --port 8504
```
