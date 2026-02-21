# Deep Research: LangGraph + Neo4j + pgvector Production Architecture for BILLY

**Date**: 2026-02-20
**Methodology**: /research-deep (8-step, multi-source validation)
**Confidence Score**: 87%
**Sources Consulted**: 18 (4 academic/official, 8 industry, 6 community)

---

## Executive Summary

Production autonomous agent systems should use a **complementary dual-store architecture**: pgvector in PostgreSQL for semantic search and operational data, Neo4j (via Graphiti/Zep's temporal knowledge graph pattern) for relationship tracking, entity memory, and multi-hop reasoning. LangGraph provides native checkpointing (short-term, per-thread) and BaseStore (long-term, cross-thread) memory primitives, with a community Neo4j checkpointer available. Correlation IDs should be propagated through Redis queues using OpenTelemetry's W3C TraceContext inject/extract pattern, with custom `traceparent` headers embedded in queue message metadata.

---

## Question 1: Correlation/Lineage Tracking Through Multi-Stage LangGraph Pipelines

### Findings

**LangGraph's Native Approach: Thread IDs + Checkpointers**

LangGraph uses `thread_id` as its primary correlation mechanism. Every graph invocation receives a `config` dict with `{"configurable": {"thread_id": "xxx"}}`. The checkpointer automatically saves state at every super-step, creating an immutable chain of snapshots linked by `PREVIOUS` relationships.

However, `thread_id` is scoped to a single StateGraph execution. When an opportunity flows from Worker tier → LangGraph StateGraph → Specialist Agent (crossing Redis queue boundaries), the native `thread_id` breaks.

**Recommended Pattern: OpenTelemetry W3C TraceContext + Custom Correlation**

The production pattern for multi-service agent pipelines uses OpenTelemetry's context propagation:

```python
from opentelemetry.propagate import inject, extract
from opentelemetry.trace.propagation import TraceContextTextMapPropagator

# PRODUCER: Before enqueueing to Redis
carrier = {}
TraceContextTextMapPropagator().inject(carrier)
# carrier now contains: {"traceparent": "00-<trace_id>-<span_id>-01"}

# Add to Redis message metadata
message = {
    "task_data": {...},
    "trace_context": carrier,
    "billy_correlation_id": "opp_abc123",  # Business correlation ID
    "billy_origin": "worker/viral_detector",
    "billy_stage": "tier1_to_tier2"
}
redis_client.xadd("queue:tier2", message)

# CONSUMER: When dequeuing from Redis
ctx = extract(message["trace_context"])
with tracer.start_as_current_span("tier2_process", context=ctx):
    # Span is now linked to the original trace
    graph.invoke(state, config={"configurable": {
        "thread_id": message["billy_correlation_id"],
        "trace_context": message["trace_context"]
    }})
```

**Key Insight from Microsoft**: Microsoft is establishing standardized OpenTelemetry semantic conventions for multi-agent tracing, built on W3C Trace Context. They've integrated these into Azure AI packages for LangChain and LangGraph specifically.

**Key Insight from LangSmith**: LangSmith now offers full end-to-end OpenTelemetry support. If using LangSmith for observability, trace IDs propagate automatically through LangGraph runs. For custom queue-based architectures like BILLY, you need manual inject/extract at queue boundaries.

**Recommended Correlation Schema for BILLY**:

```python
@dataclass
class BillyCorrelation:
    # Business ID - tracks the opportunity end-to-end
    opportunity_id: str          # "opp_abc123"

    # OTel trace context - for distributed tracing
    trace_context: dict          # W3C traceparent/tracestate

    # Pipeline lineage
    origin_worker: str           # "viral_detector"
    origin_timestamp: datetime
    current_tier: int            # 1, 2, or 3
    current_stage: str           # "evaluation", "execution", etc.

    # Audit chain
    decisions: list[dict]        # Each tier's decision + confidence
```

### Sources
- [LangSmith End-to-End OpenTelemetry](https://blog.langchain.com/end-to-end-opentelemetry-langsmith/)
- [OpenTelemetry Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)
- [OpenTelemetry Python Propagation](https://opentelemetry.io/docs/languages/python/propagation/)
- [Google Cloud: Instrument LangGraph with OpenTelemetry](https://docs.cloud.google.com/stackdriver/docs/instrumentation/ai-agent-langgraph)
- [Microsoft: Trace Agents with OpenTelemetry](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/trace-agents-sdk)

---

## Question 2: Neo4j + pgvector Dual Architecture

### Findings

**The Consensus: Complementary, Not Mirrored**

Every production reference found uses the same pattern: **pgvector for content similarity, Neo4j for relationship traversal**. They serve fundamentally different retrieval needs and should NOT mirror the same data.

**Dual-Store Data Separation (from Alejandro-Candela's agentic-rag-knowledge-graph)**:

| Store | Purpose | Data Types | Query Pattern |
|-------|---------|-----------|---------------|
| **PostgreSQL + pgvector** | Semantic similarity search | Document chunks, embeddings, metadata, operational logs | "Find content similar to X" |
| **Neo4j + Graphiti** | Relationship discovery | Entities, relationships, temporal facts, communities | "How does X relate to Y?" |

The system avoids data duplication by maintaining separate concerns: "one database optimizes for similarity, the other for relationship semantics."

**Graphiti/Zep's Temporal Knowledge Graph (Academic Validation)**:

The Zep paper (arXiv:2501.13956) establishes a three-tier graph architecture that maps well to BILLY:

1. **Episode Subgraph**: Raw events (conversations, trades, content publishes) — non-lossy storage
2. **Semantic Entity Subgraph**: Extracted entities (tools, platforms, strategies) and relationships
3. **Community Subgraph**: Clustered entity groups with high-level summaries

Performance benchmarks: 94.8% accuracy on Deep Memory Retrieval (vs 93.4% MemGPT), with P95 latency of 300ms using hybrid search (semantic + BM25 + graph traversal).

**Production Lesson from Particula (12M nodes)**:

They found that **over-modeling kills performance**. Started with 40M nodes, reduced to 12M by:
- Converting low-value lookups into node properties (not separate nodes)
- Capping traversal at 3-hop depth (5 max)
- Limiting to 500 result nodes per query
- Only 7% of queries actually needed GraphRAG (71% were cache-augmented, 22% standard RAG)

**GraphRAG Pattern (Microsoft + Neo4j)**:

The `ms-graphrag-neo4j` library implements Microsoft's GraphRAG methodology:
- **Local retriever**: Vector similarity + graph traversal (for specific queries)
- **Global retriever**: Community summaries iteration (for broad queries)
- Entity extraction → relationship mapping → community detection (Neo4j GDS) → summarization

### Recommended Architecture for BILLY

```
PostgreSQL + pgvector (existing AutoMem)
├── Operational data: trades, content, revenue, bot_activity
├── Semantic search: conversation embeddings (768-dim)
├── Audit log: confidence decisions, guardrail checks
├── Time-series: performance metrics, win rates
└── Checkpointing: LangGraph state (PostgresSaver)

Neo4j + Graphiti
├── Entity nodes: Tools, Platforms, Strategies, Markets, Creators
├── Relationship edges: "generates_revenue", "competes_with", "trending_on"
├── Temporal edges: Bi-temporal (event time + ingestion time)
├── Episode nodes: Key decisions, discoveries, trade executions
├── Community clusters: Strategy groups, market segments
└── Graph queries: "What strategies work on platforms where Tool X is trending?"
```

### Sources
- [Zep: Temporal Knowledge Graph for Agent Memory (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [Graphiti GitHub - Zep](https://github.com/getzep/graphiti)
- [Agentic RAG with pgvector + Neo4j](https://github.com/Alejandro-Candela/agentic-rag-knowledge-graph)
- [GraphRAG Implementation: 12M Nodes (Particula)](https://particula.tech/blog/graphrag-implementation-enterprise-data-platform)
- [Neo4j GraphRAG Manifesto](https://neo4j.com/blog/genai/graphrag-manifesto/)
- [ms-graphrag-neo4j](https://github.com/neo4j-contrib/ms-graphrag-neo4j)

---

## Question 3: Knowledge-Worthy Filtering (What Goes in the Knowledge Graph vs Audit Log)

### Findings

**The Core Principle: Knowledge Graph = "Why" and "How", Audit Log = "What" and "When"**

From Neo4j community patterns and the Particula production case study, the filtering criteria are:

**GOES IN NEO4J (Knowledge Graph)**:
- **Entities and their relationships** — Things that have connections to other things
- **Decisions with outcomes** — Trade executed + result, content published + engagement
- **Patterns discovered** — "VWAP strategy works best on tech stocks after earnings"
- **Causal chains** — "Viral tool detected → Instagram post created → Revenue generated"
- **Entity state changes** — Strategy promoted to golden rule, bot capability upgraded
- **Temporal facts** — "Tool X was trending from Feb 1-15" (with bi-temporal tracking)

**STAYS IN POSTGRESQL (Audit Log)**:
- **Individual API calls** — Every Alpaca order submission, every LLM inference
- **Raw metrics** — CPU usage, queue depths, response times
- **Intermediate processing steps** — Content scoring iterations, confidence gate sub-scores
- **Debug/diagnostic data** — Error traces, retry attempts, timeout logs
- **High-frequency events** — 5-minute trading cycle ticks, RSS poll results
- **Bulk operational data** — Every article scraped (282+ and growing)

**Filtering Heuristic (from Particula's 40M→12M reduction)**:

Ask these questions before writing to Neo4j:
1. **Does this entity have meaningful relationships?** (If it's standalone data, use pgvector)
2. **Would an agent need to traverse TO this from another entity?** (If not, it's a property)
3. **Does this represent knowledge that improves future decisions?** (If not, it's audit)
4. **Will this be queried via graph patterns (paths, neighbors)?** (If only by ID/similarity, use pgvector)

**Template-Based Cypher Over LLM Generation**:

Particula learned that LLM-generated Cypher had 23% error rate. They switched to ~30 parameterized query templates where the LLM fills parameters. Error rate dropped to under 4%. BILLY should follow this pattern.

**Practical Filter for BILLY**:

```python
KNOWLEDGE_WORTHY_EVENTS = {
    # Trading
    "trade_executed": True,       # Entity: Trade → Strategy → Market
    "trade_signal": False,        # Too frequent, audit only
    "strategy_performance": True,  # Pattern: Strategy → Performance over time
    "market_regime_change": True,  # Temporal: Market state transition

    # Content
    "content_published": True,     # Entity: Content → Platform → Revenue
    "content_scored": False,       # Intermediate step, audit only
    "viral_tool_detected": True,   # Entity: Tool → trending_on → Platform
    "rss_article_scraped": False,  # Bulk operation, stays in ChromaDB/pgvector

    # Revenue
    "revenue_generated": True,     # Relationship: Strategy → generates → Revenue
    "affiliate_click": False,      # High frequency, audit only
    "revenue_milestone": True,     # Community/pattern level insight

    # System
    "bot_deployed": True,          # Entity: Bot → has_capability → Skill
    "bot_heartbeat": False,        # Operational, audit only
    "golden_rule_promoted": True,  # Pattern: Rule → derived_from → Outcomes
    "confidence_decision": False,  # High frequency, audit only (unless threshold)
}
```

### Sources
- [GraphRAG Implementation: 12M Nodes (Particula)](https://particula.tech/blog/graphrag-implementation-enterprise-data-platform)
- [Neo4j Knowledge Graph Use Cases](https://neo4j.com/use-cases/knowledge-graph/)
- [Context Graphs with Neo4j (Jan 2026)](https://medium.com/neo4j/hands-on-with-context-graphs-and-neo4j-8b4b8fdc16dd)

---

## Question 4: LangGraph + Neo4j Integration Patterns

### Findings

**LangGraph Has Two Memory Layers (Not One)**:

1. **Checkpointers** (Short-term, thread-scoped): Persist state within a single conversation/execution thread. Built-in options: `MemorySaver` (in-memory), `PostgresSaver`, `AsyncSqliteSaver`, and community `Neo4jSaver`.

2. **BaseStore** (Long-term, cross-thread): Persist knowledge across ALL threads. Uses namespaced key-value storage. Built-in: `InMemoryStore`. Custom implementations possible for Neo4j.

```python
# Short-term: Per-thread state (checkpointer)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver(conn_string)

# Long-term: Cross-thread knowledge (store)
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

# Compile graph with BOTH
graph = workflow.compile(checkpointer=checkpointer, store=store)

# Store namespacing for cross-thread memory
namespace = ("user_123", "trading_patterns")
store.put(namespace, "vwap_insight", {"pattern": "works best on tech earnings"})

# Any thread can retrieve:
results = store.search(("user_123", "trading_patterns"))
```

**Neo4j Checkpointer (Community)**:

The `langgraph-checkpoint-neo4j` package (from `langchain-neo4j` v0.8.0+) provides:

- **Graph Data Model**: Thread → HAS_CHECKPOINT → Checkpoint → PREVIOUS → Checkpoint chain
- **Channel State**: Per-channel versioned state nodes
- **Branching**: Non-destructive branching from any historical checkpoint
- **Time-Travel**: `ACTIVE_BRANCH` relationship switching, fork point tracking

Node Types: `Thread`, `Checkpoint`, `ChannelState`, `PendingWrite`, `Branch`
Key Relationships: `HAS_CHECKPOINT`, `PREVIOUS`, `HAS_CHANNEL`, `HAS_WRITE`, `HAS_BRANCH`, `ACTIVE_BRANCH`, `HEAD`, `ON_BRANCH`

**Official Neo4j + LangGraph Workflow Pattern**:

Neo4j published an official blog post on creating GraphRAG workflows with LangGraph where:
- The **Knowledge Graph in Neo4j** stores entities and relationships extracted from documents
- The **State Graph in LangGraph** defines the AI agent's workflow
- LangGraph nodes call Neo4j for graph queries and vector search
- The workflow routes between different retrieval strategies (vector, Cypher, hybrid)

**Recommendation for BILLY**:

Do NOT use Neo4j as the checkpointer. Use `PostgresSaver` for checkpointing (you already have PostgreSQL, and checkpoints are high-frequency operational data). Use Neo4j exclusively for the knowledge graph via Graphiti. Access Neo4j from LangGraph nodes as a tool/resource, not as the persistence backend.

```python
# BILLY's LangGraph node that reads from Neo4j
async def knowledge_lookup(state: BillyState, store: BaseStore):
    """LangGraph node that queries Neo4j knowledge graph."""
    query = state["current_query"]

    # Hybrid retrieval: pgvector for similarity, Neo4j for relationships
    similar_docs = await pgvector_search(query, k=5)
    related_entities = await neo4j_traverse(
        start_entity=state["target_entity"],
        relationship_types=["generates_revenue", "trending_on"],
        max_hops=3
    )

    state["context"] = merge_results(similar_docs, related_entities)
    return state
```

### Sources
- [LangGraph Memory Overview](https://docs.langchain.com/oss/python/langgraph/memory)
- [langgraph-checkpoint-neo4j](https://github.com/johnymontana/langgraph-checkpoint-neo4j)
- [Neo4j + LangGraph GraphRAG Workflow](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)
- [LangGraph Cross-Thread Persistence](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence-functional/)
- [LangGraph Checkpointing Best Practices 2025](https://sparkco.ai/blog/mastering-langgraph-checkpointing-best-practices-for-2025/)
- [Neo4j LangChain Integration](https://neo4j.com/labs/genai-ecosystem/langchain/)

---

## Question 5: Graph-Enhanced RAG (Neo4j + pgvector Combined Retrieval)

### Findings

**Three Production Patterns Identified**:

### Pattern A: Microsoft GraphRAG (Community-Based)
- Extract entities/relationships from documents → build knowledge graph
- Run community detection (Leiden algorithm via Neo4j GDS)
- Generate community summaries at multiple granularity levels
- **Local search**: Vector similarity + graph neighborhood traversal
- **Global search**: Iterate community summaries for broad questions
- Best for: Large document corpora where relationships between entities matter

### Pattern B: Graphiti/Zep Hybrid Retrieval (Agent Memory)
- Three-phase retrieval: φ (Search) → ρ (Rerank) → χ (Construct)
- Search combines: cosine similarity + BM25 full-text + breadth-first graph traversal
- Reranking: Reciprocal Rank Fusion, MMR, episode-mentions, node-distance, cross-encoder
- **No LLM calls during retrieval** — sub-300ms P95 latency
- Best for: Agent memory where speed matters and data is continuously updated

### Pattern C: Router-Based (Particula Production)
- Intelligent classifier routes queries to the right retrieval system:
  - **GraphRAG** for relational queries (7% of traffic): "How does Strategy X relate to Market Y?"
  - **Standard RAG** for content queries (22%): "What are best practices for VWAP?"
  - **Cache-Augmented Generation** for frequent queries (71%): "What's my current P&L?"
- Template-based Cypher (30 templates) instead of LLM-generated Cypher (4% vs 23% error rate)
- Router accuracy improved 84% → 96% over 3 months with feedback loops

**Recommended Pattern for BILLY: Hybrid (B + C)**

BILLY should combine Graphiti's retrieval approach with Particula's routing:

```python
class BillyRetriever:
    """Hybrid retriever combining pgvector + Neo4j."""

    async def retrieve(self, query: str, context: BillyContext) -> RetrievalResult:
        # Step 1: Classify query type
        query_type = self.classify_query(query)  # template-based, not LLM

        if query_type == "relationship":
            # Neo4j graph traversal (Cypher templates)
            results = await self.neo4j_template_query(query, context)
        elif query_type == "similarity":
            # pgvector semantic search
            results = await self.pgvector_search(query, k=10)
        elif query_type == "hybrid":
            # Both, then merge with RRF
            graph_results = await self.neo4j_traverse(context.entity, max_hops=3)
            vector_results = await self.pgvector_search(query, k=10)
            results = self.reciprocal_rank_fusion(graph_results, vector_results)
        else:
            # Cache-augmented (frequent operational queries)
            results = await self.cache_lookup(query)

        return results

    def neo4j_template_query(self, query: str, context: BillyContext) -> list:
        """Use parameterized Cypher templates, NOT LLM-generated Cypher."""
        template = self.match_template(query)  # From 20-30 predefined templates
        params = self.extract_params(query, template)
        return self.neo4j.run(template.cypher, params)
```

**Traversal Budget (from Particula)**:
- Max 3 hops (5 absolute maximum)
- Max 500 result nodes per query
- Timeout with partial results + explanation (never silent failure)

### Sources
- [Neo4j GraphRAG Python Library (Official)](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- [Zep Paper: Temporal Knowledge Graph (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [GraphRAG Implementation: 12M Nodes (Particula)](https://particula.tech/blog/graphrag-implementation-enterprise-data-platform)
- [ms-graphrag-neo4j](https://github.com/neo4j-contrib/ms-graphrag-neo4j)
- [Neo4j + Qdrant GraphRAG Pattern](https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/)
- [AWS Knowledge Graphs with Neo4j](https://docs.aws.amazon.com/architecture-diagrams/latest/knowledge-graphs-and-graphrag-with-neo4j/knowledge-graphs-and-graphrag-with-neo4j.html)

---

## Recommended Architecture for BILLY

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BILLY ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TIER 1: Workers (Redis Queues)                                      │
│  ├── Each worker creates BillyCorrelation with opportunity_id        │
│  ├── OTel inject() → trace_context embedded in Redis message         │
│  └── High-frequency events → PostgreSQL audit log only               │
│                                                                      │
│  ──── Redis Queue (with traceparent in message metadata) ────        │
│                                                                      │
│  TIER 2: LangGraph StateGraphs                                       │
│  ├── OTel extract() → continues trace from Tier 1                    │
│  ├── PostgresSaver for checkpointing (short-term, per-thread)        │
│  ├── BaseStore (custom Neo4j-backed) for cross-thread knowledge      │
│  ├── Knowledge-worthy events → dual-write to Neo4j via Graphiti      │
│  └── BillyRetriever: hybrid pgvector + Neo4j retrieval               │
│                                                                      │
│  ──── Redis Queue (traceparent propagated) ────                      │
│                                                                      │
│  TIER 3: Specialist Agents                                           │
│  ├── OTel extract() → continues trace from Tier 2                    │
│  ├── Graph-enhanced RAG for context (Neo4j relationships + pgvector) │
│  ├── Outcome events → Neo4j (trade result, content engagement)       │
│  └── Pattern extraction → Neo4j community detection                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                    │
│                                                                      │
│  PostgreSQL + pgvector          │  Neo4j + Graphiti                  │
│  ─────────────────────          │  ─────────────────                 │
│  • LangGraph checkpoints        │  • Entity nodes (Tools, Markets,  │
│  • Operational audit log         │    Strategies, Platforms)         │
│  • Revenue tracking              │  • Relationship edges             │
│  • Bot activity (high-freq)      │  • Temporal facts (bi-temporal)   │
│  • Embeddings (768-dim)          │  • Episode nodes (key decisions)  │
│  • Confidence audit log          │  • Community clusters             │
│  • Content queue                 │  • Golden rules + patterns        │
│  • Trade history (raw)           │  • Causal chains                  │
│                                  │                                   │
│  Query: "Find similar to X"      │  Query: "How does X relate to Y?" │
│  Query: "What happened at T?"    │  Query: "What strategies work     │
│                                  │   on platforms where Z trends?"   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                     OBSERVABILITY                                    │
│                                                                      │
│  OpenTelemetry (W3C TraceContext)                                    │
│  ├── Trace spans: Worker → Queue → StateGraph → Queue → Agent        │
│  ├── Custom attributes: opportunity_id, tier, confidence_score       │
│  ├── LangSmith integration (optional, for LangGraph-specific traces) │
│  └── Export to: Jaeger/Grafana Tempo (self-hosted, free)             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Implementation Decisions

### 1. Use PostgresSaver, NOT Neo4jSaver for Checkpointing
- Checkpoints are high-frequency, operational data
- You already have PostgreSQL — zero new infrastructure for this
- Neo4j checkpointer is community-maintained (not official LangChain)
- Reserve Neo4j exclusively for knowledge graph duties

### 2. Use Graphiti (Zep) for Neo4j Knowledge Graph
- Proven architecture (academic paper, 94.8% accuracy)
- Bi-temporal model is critical for "what did we know when?"
- Pluggable driver (supports Neo4j, FalkorDB, Kuzu, Neptune)
- Sub-300ms retrieval with no LLM calls during search
- Handles entity extraction, deduplication, temporal invalidation

### 3. Template-Based Cypher, NOT LLM-Generated
- Particula's production data: 23% error rate with LLM Cypher → 4% with templates
- Build 20-30 parameterized query templates for BILLY's common patterns
- LLM fills parameters, doesn't generate syntax

### 4. Router-Based Retrieval, NOT Always-Graph
- Only 7% of queries in production actually need graph traversal
- Classify query intent → route to cheapest effective retrieval
- Cache frequent operational queries (P&L, status, recent trades)

### 5. Traversal Budgets Are Non-Negotiable
- Max 3 hops default, 5 absolute max
- Max 500 result nodes per query
- Timeout with partial results, never silent failure

### 6. Don't Over-Model the Graph
- Start with core entities only (Tools, Strategies, Markets, Platforms, Revenue)
- Resist putting everything in Neo4j (Particula cut 40M → 12M nodes for performance)
- If an entity has no meaningful relationships, it's a pgvector row

---

## Confidence Assessment: 87%

**High Confidence (90%+)**:
- Dual-store architecture (pgvector + Neo4j) — unanimous across all sources
- OpenTelemetry for correlation tracking — industry standard
- PostgresSaver for checkpointing — obvious fit given existing infrastructure
- Template-based Cypher — validated by 12M node production system

**Medium Confidence (75-85%)**:
- Graphiti as the Neo4j framework — strong paper but relatively new (Jan 2025)
- BaseStore custom implementation for Neo4j long-term memory — pattern exists but no reference implementation
- Router-based retrieval percentages — Particula's 7/22/71 split may differ for BILLY's domain

**Lower Confidence (60-70%)**:
- Exact filtering criteria for knowledge-worthy events — domain-specific, needs empirical tuning
- Community detection ROI at BILLY's scale — may be premature until graph has 10K+ entities

**Gaps Not Fully Resolved**:
- No production reference found for LangGraph + Graphiti + Redis queues in a single system
- Custom BaseStore backed by Neo4j has no reference implementation (would need to be built)
- Grok cross-validation was not performed (Chrome MCP not available this session)

---

## Implementation Priority for BILLY

1. **Phase 1**: Add OpenTelemetry tracing with correlation IDs through Redis queues (1-2 days)
2. **Phase 2**: Deploy PostgresSaver for LangGraph checkpointing (1 day)
3. **Phase 3**: Set up Neo4j + Graphiti, define initial entity schema (2-3 days)
4. **Phase 4**: Build knowledge-worthy event filter + dual-write pipeline (2 days)
5. **Phase 5**: Implement BillyRetriever with hybrid search + query routing (3-4 days)
6. **Phase 6**: Build Cypher query templates (20-30) for common patterns (2 days)
7. **Phase 7**: Add community detection when graph exceeds 10K entities (future)

**Total estimated**: 11-14 days for Phases 1-6

---

## All Sources

### Academic/Official
1. [Zep: Temporal Knowledge Graph for Agent Memory (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
2. [Neo4j GraphRAG Python Library (Official Docs)](https://neo4j.com/docs/neo4j-graphrag-python/current/)
3. [OpenTelemetry Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)
4. [LangGraph Memory Overview](https://docs.langchain.com/oss/python/langgraph/memory)

### Industry
5. [LangSmith End-to-End OpenTelemetry](https://blog.langchain.com/end-to-end-opentelemetry-langsmith/)
6. [Neo4j GraphRAG Manifesto](https://neo4j.com/blog/genai/graphrag-manifesto/)
7. [Neo4j + LangGraph GraphRAG Workflow](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)
8. [GraphRAG Implementation: 12M Nodes (Particula)](https://particula.tech/blog/graphrag-implementation-enterprise-data-platform)
9. [AWS Knowledge Graphs with Neo4j](https://docs.aws.amazon.com/architecture-diagrams/latest/knowledge-graphs-and-graphrag-with-neo4j/knowledge-graphs-and-graphrag-with-neo4j.html)
10. [Google Cloud: Instrument LangGraph with OTel](https://docs.cloud.google.com/stackdriver/docs/instrumentation/ai-agent-langgraph)
11. [Microsoft: Multi-Agent Tracing](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/trace-agents-sdk)
12. [Neo4j Cypher AI Procedures (Dec 2025)](https://medium.com/neo4j/new-cypher-ai-procedures-6b8c3177d56d)

### Community
13. [langgraph-checkpoint-neo4j](https://github.com/johnymontana/langgraph-checkpoint-neo4j)
14. [Graphiti GitHub (Zep)](https://github.com/getzep/graphiti)
15. [ms-graphrag-neo4j](https://github.com/neo4j-contrib/ms-graphrag-neo4j)
16. [Agentic RAG with pgvector + Neo4j + Graphiti](https://github.com/Alejandro-Candela/agentic-rag-knowledge-graph)
17. [LangGraph Checkpointing Best Practices 2025](https://sparkco.ai/blog/mastering-langgraph-checkpointing-best-practices-for-2025/)
18. [Qdrant + Neo4j GraphRAG Pattern](https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/)
