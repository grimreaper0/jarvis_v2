"""GraphMemory — Neo4j knowledge graph layer (async driver).

New capability for jarvis_v2. Tracks entity relationships that vector search
cannot express: article → pattern → trade chains, multi-hop influence paths,
opportunity-to-revenue lineage.

Node types:    Article, Pattern, Opportunity, Bot, Strategy
Relationship types: MENTIONS, DERIVED_FROM, APPLIED_TO, GENERATED, SIMILAR_TO

Uses MERGE to avoid duplicates. Async neo4j driver throughout.
"""

import asyncio
from typing import Any

import structlog

log = structlog.get_logger()


class GraphMemory:
    """Neo4j knowledge graph for entity relationships.

    Stores entities and relationships that reveal patterns impossible to
    capture with vector similarity alone — e.g. which articles consistently
    influence profitable patterns, or multi-hop revenue chains.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        from config.settings import get_settings
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._driver = None
        self._driver_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Driver management
    # ------------------------------------------------------------------

    async def _get_driver(self):
        """Return (or lazily create) the async Neo4j driver."""
        if self._driver is not None:
            return self._driver
        async with self._driver_lock:
            if self._driver is not None:
                return self._driver
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            log.info("graph_memory.driver_created", uri=self.uri)
        return self._driver

    async def close(self):
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def _run(self, cypher: str, **params) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dicts."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            records = await result.data()
        return records

    async def _run_write(self, cypher: str, **params) -> None:
        """Execute a write Cypher query (no return value needed)."""
        driver = await self._get_driver()
        async with driver.session() as session:
            await session.run(cypher, **params)

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    async def store_article_node(
        self,
        article_id: str,
        title: str,
        source: str,
        tags: list[str],
    ) -> None:
        """Upsert an Article node in the knowledge graph."""
        await self._run_write(
            """
            MERGE (a:Article {article_id: $article_id})
            SET a.title = $title,
                a.source = $source,
                a.tags = $tags,
                a.updated_at = datetime()
            """,
            article_id=article_id,
            title=title,
            source=source,
            tags=tags,
        )
        log.debug("graph.article_stored", id=article_id, source=source)

    async def store_pattern_node(
        self,
        pattern_id: str,
        pattern_type: str,
        confidence: float,
    ) -> None:
        """Upsert a Pattern node."""
        await self._run_write(
            """
            MERGE (p:Pattern {pattern_id: $pattern_id})
            SET p.pattern_type = $pattern_type,
                p.confidence = $confidence,
                p.updated_at = datetime()
            """,
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            confidence=confidence,
        )
        log.debug("graph.pattern_stored", id=pattern_id, type=pattern_type)

    async def store_opportunity_node(
        self,
        opp_id: str,
        name: str,
        estimated_value: float,
    ) -> None:
        """Upsert an Opportunity node."""
        await self._run_write(
            """
            MERGE (o:Opportunity {opp_id: $opp_id})
            SET o.name = $name,
                o.estimated_value = $estimated_value,
                o.updated_at = datetime()
            """,
            opp_id=opp_id,
            name=name,
            estimated_value=estimated_value,
        )
        log.debug("graph.opportunity_stored", id=opp_id, name=name)

    async def store_bot_node(self, bot_name: str, bot_type: str) -> None:
        """Upsert a Bot node."""
        await self._run_write(
            """
            MERGE (b:Bot {bot_name: $bot_name})
            SET b.bot_type = $bot_type,
                b.updated_at = datetime()
            """,
            bot_name=bot_name,
            bot_type=bot_type,
        )
        log.debug("graph.bot_stored", name=bot_name)

    async def store_strategy_node(
        self,
        strategy_id: str,
        bot_name: str,
        status: str,
    ) -> None:
        """Upsert a Strategy node."""
        await self._run_write(
            """
            MERGE (s:Strategy {strategy_id: $strategy_id})
            SET s.bot_name = $bot_name,
                s.status = $status,
                s.updated_at = datetime()
            """,
            strategy_id=strategy_id,
            bot_name=bot_name,
            status=status,
        )
        log.debug("graph.strategy_stored", id=strategy_id, bot=bot_name)

    # ------------------------------------------------------------------
    # Relationship creation
    # ------------------------------------------------------------------

    async def link_article_to_pattern(
        self,
        article_id: str,
        pattern_id: str,
        weight: float = 1.0,
    ) -> None:
        """Create MENTIONS relationship: Article → Pattern."""
        await self._run_write(
            """
            MATCH (a:Article {article_id: $article_id})
            MATCH (p:Pattern {pattern_id: $pattern_id})
            MERGE (a)-[r:MENTIONS]->(p)
            SET r.weight = $weight,
                r.updated_at = datetime()
            """,
            article_id=article_id,
            pattern_id=pattern_id,
            weight=weight,
        )
        log.debug("graph.link_article_pattern", article=article_id, pattern=pattern_id)

    async def link_pattern_to_opportunity(
        self,
        pattern_id: str,
        opp_id: str,
    ) -> None:
        """Create DERIVED_FROM relationship: Opportunity → Pattern."""
        await self._run_write(
            """
            MATCH (p:Pattern {pattern_id: $pattern_id})
            MATCH (o:Opportunity {opp_id: $opp_id})
            MERGE (o)-[r:DERIVED_FROM]->(p)
            SET r.created_at = datetime()
            """,
            pattern_id=pattern_id,
            opp_id=opp_id,
        )
        log.debug("graph.link_pattern_opportunity", pattern=pattern_id, opp=opp_id)

    async def link_strategy_to_bot(
        self,
        strategy_id: str,
        bot_name: str,
    ) -> None:
        """Create APPLIED_TO relationship: Strategy → Bot."""
        await self._run_write(
            """
            MATCH (s:Strategy {strategy_id: $strategy_id})
            MATCH (b:Bot {bot_name: $bot_name})
            MERGE (s)-[r:APPLIED_TO]->(b)
            SET r.created_at = datetime()
            """,
            strategy_id=strategy_id,
            bot_name=bot_name,
        )
        log.debug("graph.link_strategy_bot", strategy=strategy_id, bot=bot_name)

    async def link_similar_patterns(
        self,
        pattern_id_a: str,
        pattern_id_b: str,
        similarity: float,
    ) -> None:
        """Create SIMILAR_TO relationship between two Pattern nodes."""
        await self._run_write(
            """
            MATCH (a:Pattern {pattern_id: $a})
            MATCH (b:Pattern {pattern_id: $b})
            MERGE (a)-[r:SIMILAR_TO]-(b)
            SET r.similarity = $similarity,
                r.updated_at = datetime()
            """,
            a=pattern_id_a,
            b=pattern_id_b,
            similarity=similarity,
        )

    # ------------------------------------------------------------------
    # Multi-hop queries
    # ------------------------------------------------------------------

    async def find_related_patterns(
        self,
        article_id: str,
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Find patterns reachable from an article within max_hops.

        Traverses MENTIONS and SIMILAR_TO edges. Returns patterns sorted
        by hop distance (closer = more relevant).
        """
        records = await self._run(
            """
            MATCH (a:Article {article_id: $article_id})
            MATCH path = (a)-[:MENTIONS|SIMILAR_TO*1..$max_hops]->(p:Pattern)
            RETURN p.pattern_id AS pattern_id,
                   p.pattern_type AS pattern_type,
                   p.confidence AS confidence,
                   length(path) AS hops
            ORDER BY hops ASC, p.confidence DESC
            """,
            article_id=article_id,
            max_hops=max_hops,
        )
        log.debug("graph.related_patterns", article=article_id, count=len(records))
        return records

    async def find_successful_chains(
        self,
        min_revenue: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Find Article → Pattern → Opportunity chains that generated revenue.

        Returns chains ordered by estimated opportunity value descending.
        Useful for surfacing which content types drive the most revenue.
        """
        records = await self._run(
            """
            MATCH (a:Article)-[:MENTIONS]->(p:Pattern)<-[:DERIVED_FROM]-(o:Opportunity)
            WHERE o.estimated_value >= $min_revenue
            RETURN a.article_id AS article_id,
                   a.title AS article_title,
                   a.source AS source,
                   p.pattern_id AS pattern_id,
                   p.pattern_type AS pattern_type,
                   o.opp_id AS opp_id,
                   o.name AS opp_name,
                   o.estimated_value AS estimated_value
            ORDER BY o.estimated_value DESC
            """,
            min_revenue=min_revenue,
        )
        log.debug("graph.successful_chains", count=len(records))
        return records

    async def find_high_influence_articles(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find articles that influenced the most patterns (highest out-degree via MENTIONS)."""
        records = await self._run(
            """
            MATCH (a:Article)-[:MENTIONS]->(p:Pattern)
            WITH a, COUNT(p) AS pattern_count
            ORDER BY pattern_count DESC
            LIMIT $limit
            RETURN a.article_id AS article_id,
                   a.title AS title,
                   a.source AS source,
                   a.tags AS tags,
                   pattern_count
            """,
            limit=limit,
        )
        return records

    # ------------------------------------------------------------------
    # Stats & introspection
    # ------------------------------------------------------------------

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return node and relationship counts for all types."""
        node_types = ["Article", "Pattern", "Opportunity", "Bot", "Strategy"]
        rel_types = ["MENTIONS", "DERIVED_FROM", "APPLIED_TO", "GENERATED", "SIMILAR_TO"]

        stats: dict[str, Any] = {"nodes": {}, "relationships": {}}

        for label in node_types:
            result = await self._run(
                f"MATCH (n:{label}) RETURN count(n) AS cnt"
            )
            stats["nodes"][label] = result[0]["cnt"] if result else 0

        for rel in rel_types:
            result = await self._run(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt"
            )
            stats["relationships"][rel] = result[0]["cnt"] if result else 0

        stats["total_nodes"] = sum(stats["nodes"].values())
        stats["total_relationships"] = sum(stats["relationships"].values())
        log.debug("graph.stats", **stats)
        return stats

    # ------------------------------------------------------------------
    # Generic escape hatch
    # ------------------------------------------------------------------

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run arbitrary Cypher and return results."""
        return await self._run(cypher, **(params or {}))

    # ------------------------------------------------------------------
    # Compatibility shim (sync upsert_entity from scaffold)
    # ------------------------------------------------------------------

    async def upsert_entity(
        self,
        label: str,
        name: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Generic MERGE for any node label. Properties are set via SET n += props."""
        props = properties or {}
        await self._run_write(
            f"MERGE (n:{label} {{name: $name}}) SET n += $props",
            name=name,
            props=props,
        )
        log.debug("graph.entity_upserted", label=label, name=name)

    async def upsert_relationship(
        self,
        from_label: str,
        from_name: str,
        rel_type: str,
        to_label: str,
        to_name: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Generic MERGE for a relationship between two named nodes."""
        props = properties or {}
        cypher = (
            f"MATCH (a:{from_label} {{name: $from_name}}), "
            f"(b:{to_label} {{name: $to_name}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props"
        )
        await self._run_write(cypher, from_name=from_name, to_name=to_name, props=props)
        log.debug("graph.relationship_upserted", rel=rel_type, from_=from_name, to=to_name)
