"""GraphMemory — Neo4j knowledge graph integration."""
import structlog
from typing import Any

log = structlog.get_logger()


class GraphMemory:
    """Neo4j-backed knowledge graph for entity and relationship storage.

    Stores entities (tools, strategies, bots, revenue streams) and their
    relationships for richer context retrieval than vector search alone.
    """

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from config.settings import get_settings
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def upsert_entity(self, label: str, name: str, properties: dict | None = None) -> None:
        driver = self._get_driver()
        props = properties or {}
        with driver.session() as session:
            session.run(
                f"MERGE (n:{label} {{name: $name}}) SET n += $props",
                name=name,
                props=props,
            )
        log.debug("entity.upserted", label=label, name=name)

    def upsert_relationship(
        self,
        from_label: str,
        from_name: str,
        rel_type: str,
        to_label: str,
        to_name: str,
        properties: dict | None = None,
    ) -> None:
        driver = self._get_driver()
        props = properties or {}
        query = (
            f"MATCH (a:{from_label} {{name: $from_name}}), (b:{to_label} {{name: $to_name}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props"
        )
        with driver.session() as session:
            session.run(query, from_name=from_name, to_name=to_name, props=props)
        log.debug("relationship.upserted", rel=rel_type, from_=from_name, to=to_name)

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [dict(record) for record in result]

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
