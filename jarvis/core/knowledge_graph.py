"""KnowledgeGraph — Neo4j relationship layer for BILLY's brain.

Works alongside pgvector (semantic search) to provide relationship awareness.
pgvector answers "what is similar?" — Neo4j answers "how are things connected?"

Connection: bolt://localhost:7687
Schema: see NODE_TYPES and RELATIONSHIP_TYPES below.
"""

import json
import structlog
from datetime import datetime
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

log = structlog.get_logger()


# ─────────────────────── Schema Reference ─────────────────────────────────────
#
# Node labels:
#   Bot, Activity, Conversation, Pattern, ConfidenceDecision,
#   RoadmapItem, ViralOpportunity, PersonalNote, DataSource,
#   Platform, Topic, Quest
#
# Relationship types:
#   Bot -[PERFORMED]-> Activity
#   Bot -[OPERATES_ON]-> Platform
#   Activity -[EVALUATED_BY]-> ConfidenceDecision
#   Activity -[TRIGGERED_BY]-> ViralOpportunity
#   Activity -[ABOUT]-> Topic
#   Conversation -[ABOUT]-> Topic
#   Conversation -[LED_TO]-> Pattern
#   Pattern -[APPLIED_BY]-> Bot
#   Pattern -[EVOLVED_INTO]-> Pattern
#   RoadmapItem -[BLOCKS]-> RoadmapItem
#   RoadmapItem -[RELATES_TO]-> RoadmapItem
#   ViralOpportunity -[DETECTED_ON]-> Platform
#   ViralOpportunity -[GENERATED]-> Activity
#   Quest -[BLOCKS]-> Quest
#   Quest -[RELATES_TO]-> Quest
#   DataSource -[FEEDS]-> Bot
# ──────────────────────────────────────────────────────────────────────────────


class KnowledgeGraph:
    """Neo4j async interface for BILLY's relationship memory."""

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None):
        from config.settings import get_settings
        settings = get_settings()
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._user, self._password),
            )
            await self._driver.verify_connectivity()
            log.info("knowledge_graph.connected", uri=self._uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def _ensure_connected(self) -> AsyncDriver:
        if self._driver is None:
            await self.connect()
        return self._driver

    async def ensure_indexes(self) -> None:
        """Create indexes and constraints for all node types."""
        driver = await self._ensure_connected()
        async with driver.session() as s:
            # Unique constraints (also create indexes)
            constraints = [
                ("Bot", "name"),
                ("Platform", "name"),
                ("Topic", "name"),
                ("DataSource", "name"),
            ]
            for label, prop in constraints:
                await s.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )

            # Indexes for frequently queried properties
            indexes = [
                ("Activity", "pg_id"),
                ("Activity", "bot_name"),
                ("Activity", "created_at"),
                ("Conversation", "pg_id"),
                ("ConfidenceDecision", "pg_id"),
                ("Pattern", "pg_id"),
                ("RoadmapItem", "pg_id"),
                ("ViralOpportunity", "pg_id"),
                ("PersonalNote", "pg_id"),
                ("Quest", "quest_id"),
            ]
            for label, prop in indexes:
                await s.run(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.{prop})"
                )
        log.info("knowledge_graph.indexes_ensured")

    # ─────────────────────── Node Operations ──────────────────────────────────

    async def merge_bot(self, name: str, platform: str | None = None,
                        **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (b:Bot {name: $name}) SET b += $props",
                name=name, props=props,
            )
            if platform:
                await s.run(
                    "MERGE (b:Bot {name: $name}) "
                    "MERGE (p:Platform {name: $platform}) "
                    "MERGE (b)-[:OPERATES_ON]->(p)",
                    name=name, platform=platform,
                )

    async def add_activity(self, pg_id: int | str, bot_name: str, activity_type: str,
                           status: str, created_at: str,
                           topics: list[str] | None = None,
                           **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (a:Activity {pg_id: $pg_id}) "
                "SET a.bot_name = $bot_name, a.activity_type = $activity_type, "
                "    a.status = $status, a.created_at = $created_at "
                "SET a += $props "
                "WITH a "
                "MERGE (b:Bot {name: $bot_name}) "
                "MERGE (b)-[:PERFORMED]->(a)",
                pg_id=pg_id, bot_name=bot_name, activity_type=activity_type,
                status=status, created_at=created_at, props=props,
            )
            if topics:
                for topic in topics:
                    await s.run(
                        "MATCH (a:Activity {pg_id: $pg_id}) "
                        "MERGE (t:Topic {name: $topic}) "
                        "MERGE (a)-[:ABOUT]->(t)",
                        pg_id=pg_id, topic=topic.lower().strip(),
                    )

    async def add_conversation(self, pg_id: str, title: str,
                               topics: list[str] | None = None,
                               **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (c:Conversation {pg_id: $pg_id}) "
                "SET c.title = $title SET c += $props",
                pg_id=pg_id, title=title, props=props,
            )
            if topics:
                for topic in topics:
                    await s.run(
                        "MATCH (c:Conversation {pg_id: $pg_id}) "
                        "MERGE (t:Topic {name: $topic}) "
                        "MERGE (c)-[:ABOUT]->(t)",
                        pg_id=pg_id, topic=topic.lower().strip(),
                    )

    async def add_confidence_decision(self, pg_id: int, task_type: str,
                                      decision: str, final_score: float,
                                      **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (cd:ConfidenceDecision {pg_id: $pg_id}) "
                "SET cd.task_type = $task_type, cd.decision = $decision, "
                "    cd.final_score = $final_score "
                "SET cd += $props",
                pg_id=pg_id, task_type=task_type, decision=decision,
                final_score=final_score, props=props,
            )

    async def add_viral_opportunity(self, pg_id: int, title: str,
                                    platform: str, score: float = 0.0,
                                    **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (v:ViralOpportunity {pg_id: $pg_id}) "
                "SET v.title = $title, v.platform = $platform, v.score = $score "
                "SET v += $props "
                "WITH v "
                "MERGE (p:Platform {name: $platform}) "
                "MERGE (v)-[:DETECTED_ON]->(p)",
                pg_id=pg_id, title=title, platform=platform, score=score,
                props=props,
            )

    async def add_roadmap_item(self, pg_id: int, title: str, phase: str,
                               status: str, **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (r:RoadmapItem {pg_id: $pg_id}) "
                "SET r.title = $title, r.phase = $phase, r.status = $status "
                "SET r += $props",
                pg_id=pg_id, title=title, phase=phase, status=status,
                props=props,
            )

    async def add_personal_note(self, pg_id: str, title: str,
                                topics: list[str] | None = None,
                                **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (n:PersonalNote {pg_id: $pg_id}) "
                "SET n.title = $title SET n += $props",
                pg_id=pg_id, title=title, props=props,
            )
            if topics:
                for topic in topics:
                    await s.run(
                        "MATCH (n:PersonalNote {pg_id: $pg_id}) "
                        "MERGE (t:Topic {name: $topic}) "
                        "MERGE (n)-[:ABOUT]->(t)",
                        pg_id=pg_id, topic=topic.lower().strip(),
                    )

    async def add_pattern(self, pg_id: int, pattern_type: str,
                          description: str, confidence: float = 0.0,
                          **props) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (p:Pattern {pg_id: $pg_id}) "
                "SET p.pattern_type = $pattern_type, p.description = $description, "
                "    p.confidence = $confidence "
                "SET p += $props",
                pg_id=pg_id, pattern_type=pattern_type, description=description,
                confidence=confidence, props=props,
            )

    # ─────────────────────── Relationship Operations ──────────────────────────

    async def link_activity_to_confidence(self, activity_pg_id: int,
                                          confidence_pg_id: int) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MATCH (a:Activity {pg_id: $a_id}) "
                "MATCH (cd:ConfidenceDecision {pg_id: $cd_id}) "
                "MERGE (a)-[:EVALUATED_BY]->(cd)",
                a_id=activity_pg_id, cd_id=confidence_pg_id,
            )

    async def link_activity_to_viral(self, activity_pg_id: int,
                                     viral_pg_id: int) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MATCH (a:Activity {pg_id: $a_id}) "
                "MATCH (v:ViralOpportunity {pg_id: $v_id}) "
                "MERGE (a)-[:TRIGGERED_BY]->(v)",
                a_id=activity_pg_id, v_id=viral_pg_id,
            )

    async def link_roadmap_blocks(self, blocker_pg_id: int,
                                  blocked_pg_id: int) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MATCH (a:RoadmapItem {pg_id: $a_id}) "
                "MATCH (b:RoadmapItem {pg_id: $b_id}) "
                "MERGE (a)-[:BLOCKS]->(b)",
                a_id=blocker_pg_id, b_id=blocked_pg_id,
            )

    async def link_conversation_to_pattern(self, conv_pg_id: str,
                                           pattern_pg_id: int) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MATCH (c:Conversation {pg_id: $c_id}) "
                "MATCH (p:Pattern {pg_id: $p_id}) "
                "MERGE (c)-[:LED_TO]->(p)",
                c_id=conv_pg_id, p_id=pattern_pg_id,
            )

    async def link_pattern_applied_by(self, pattern_pg_id: int,
                                      bot_name: str) -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MATCH (p:Pattern {pg_id: $p_id}) "
                "MERGE (b:Bot {name: $name}) "
                "MERGE (p)-[:APPLIED_BY]->(b)",
                p_id=pattern_pg_id, name=bot_name,
            )

    async def link_datasource_feeds_bot(self, source_name: str,
                                        bot_name: str,
                                        source_type: str = "rss") -> None:
        driver = await self._ensure_connected()
        async with driver.session() as s:
            await s.run(
                "MERGE (d:DataSource {name: $source}) "
                "SET d.type = $type "
                "MERGE (b:Bot {name: $bot}) "
                "MERGE (d)-[:FEEDS]->(b)",
                source=source_name, type=source_type, bot=bot_name,
            )

    # ─────────────────────── Query Operations ─────────────────────────────────

    async def get_bot_activity_graph(self, bot_name: str,
                                     limit: int = 50) -> list[dict]:
        """Get recent activities for a bot with all relationships."""
        driver = await self._ensure_connected()
        async with driver.session() as s:
            result = await s.run(
                "MATCH (b:Bot {name: $name})-[:PERFORMED]->(a:Activity) "
                "OPTIONAL MATCH (a)-[:ABOUT]->(t:Topic) "
                "OPTIONAL MATCH (a)-[:EVALUATED_BY]->(cd:ConfidenceDecision) "
                "RETURN a, collect(DISTINCT t.name) as topics, "
                "       cd.decision as confidence_decision, "
                "       cd.final_score as confidence_score "
                "ORDER BY a.created_at DESC LIMIT $limit",
                name=bot_name, limit=limit,
            )
            return [dict(r) async for r in result]

    async def get_topic_connections(self, topic: str) -> dict:
        """Get everything connected to a topic."""
        driver = await self._ensure_connected()
        async with driver.session() as s:
            result = await s.run(
                "MATCH (t:Topic {name: $topic})<-[:ABOUT]-(n) "
                "RETURN labels(n)[0] as type, count(n) as count",
                topic=topic.lower().strip(),
            )
            connections = {}
            async for r in result:
                connections[r["type"]] = r["count"]
            return connections

    async def get_revenue_lineage(self, activity_pg_id: int) -> list[dict]:
        """Trace revenue lineage: Activity ← ViralOpportunity ← DataSource."""
        driver = await self._ensure_connected()
        async with driver.session() as s:
            result = await s.run(
                "MATCH path = (a:Activity {pg_id: $id})"
                "-[:TRIGGERED_BY|DETECTED_ON|FEEDS*0..5]-(connected) "
                "RETURN [n IN nodes(path) | {labels: labels(n), props: properties(n)}] as chain",
                id=activity_pg_id,
            )
            return [dict(r) async for r in result]

    async def get_bot_collaboration_chain(self, limit: int = 20) -> list[dict]:
        """Find bot collaboration patterns: Bot A discovered → Bot B evaluated → Bot C published."""
        driver = await self._ensure_connected()
        async with driver.session() as s:
            result = await s.run(
                "MATCH (b1:Bot)-[:PERFORMED]->(a1:Activity)-[:TRIGGERED_BY]->"
                "(v:ViralOpportunity)<-[:TRIGGERED_BY]-(a2:Activity)<-[:PERFORMED]-(b2:Bot) "
                "WHERE b1 <> b2 "
                "RETURN b1.name as bot_1, a1.activity_type as action_1, "
                "       v.title as opportunity, "
                "       b2.name as bot_2, a2.activity_type as action_2 "
                "ORDER BY a2.created_at DESC LIMIT $limit",
                limit=limit,
            )
            return [dict(r) async for r in result]

    async def get_graph_stats(self) -> dict:
        """Return node and relationship counts by type."""
        driver = await self._ensure_connected()
        async with driver.session() as s:
            nodes_result = await s.run(
                "MATCH (n) RETURN labels(n)[0] as label, count(n) as count "
                "ORDER BY count DESC"
            )
            nodes = {r["label"]: r["count"] async for r in nodes_result}

            rels_result = await s.run(
                "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count "
                "ORDER BY count DESC"
            )
            rels = {r["type"]: r["count"] async for r in rels_result}

            return {"nodes": nodes, "relationships": rels}

    async def search_by_relationship(self, from_label: str, rel_type: str,
                                     to_label: str, limit: int = 25) -> list[dict]:
        """Generic relationship query."""
        driver = await self._ensure_connected()
        async with driver.session() as s:
            result = await s.run(
                f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label}) "
                f"RETURN properties(a) as from_node, type(r) as rel, "
                f"       properties(b) as to_node "
                f"LIMIT $limit",
                limit=limit,
            )
            return [dict(r) async for r in result]
