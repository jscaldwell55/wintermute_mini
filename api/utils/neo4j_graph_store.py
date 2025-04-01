# api/utils/neo4j_graph_store.py
from neo4j import AsyncGraphDatabase
import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class Neo4jGraphStore:
    """
    A graph storage system using Neo4j.
    Handles nodes (memories) and relationships (associations between memories).
    """

    def __init__(self):
        self.driver = None
        self.initialized = False

    async def initialize(self):
        """Initialize Neo4j connection."""
        try:
            neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

            self.driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
            async with self.driver.session() as session:
                await session.run("RETURN 1")  # Test connection

            self.initialized = True
            logger.info("✅ Connected to Neo4j successfully.")
            return True

        except Exception as e:
            logger.error(f"❌ Error initializing Neo4j: {e}")
            self.initialized = False
            return False

    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()

    async def store_memory(self, memory_id: str, content: str, memory_type: str, created_at: str):
        """Store a memory node in Neo4j."""
        query = """
        MERGE (m:Memory {id: $memory_id})
        SET m.content = $content, m.memory_type = $memory_type, m.created_at = $created_at
        """
        async with self.driver.session() as session:
            await session.run(query, memory_id=memory_id, content=content, memory_type=memory_type, created_at=created_at)
            logger.info(f"✅ Stored memory {memory_id}")

    async def store_relationship(self, source_id: str, target_id: str, rel_type: str, weight: float):
        """Store a relationship between two memories."""
        query = """
        MATCH (s:Memory {id: $source_id})
        MATCH (t:Memory {id: $target_id})
        MERGE (s)-[r:RELATED_TO {type: $rel_type}]->(t)
        SET r.weight = $weight
        """
        async with self.driver.session() as session:
            await session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                rel_type=rel_type,
                weight=weight
            )
            logger.info(f"✅ Created relationship {rel_type} between {source_id} and {target_id}")


    async def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Retrieve all relationships from Neo4j."""
        query = """
        MATCH (s)-[r]->(t)
        RETURN id(s) AS source_id, id(t) AS target_id, type(r) AS rel_type, r.weight AS weight
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            relationships = [record.data() async for record in result]
        return relationships


    async def get_graph_stats(self) -> Dict[str, Any]:
        """Retrieve basic graph statistics."""
        query = """
        MATCH (n) RETURN count(n) AS node_count
        UNION ALL
        MATCH ()-[r]->() RETURN count(r) AS relationship_count
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            counts = [record["node_count"] for record in result]
        return {"node_count": counts[0], "relationship_count": counts[1]}

    async def clear_graph(self):
        """Delete all nodes and relationships in the graph."""
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.info("✅ Cleared Neo4j graph.")
