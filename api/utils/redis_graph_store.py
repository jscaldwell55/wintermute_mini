# api/utils/redis_graph_store.py
import redis.asyncio as redis
import logging
import os
import asyncio
import json
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)

class RedisGraphStore:
    def __init__(self):
        self.redis = None
        self.initialized = False
    
    async def initialize(self):
        """Connect to Redis and initialize the client."""
        try:
            # Try both environment variables since RedisCloud uses a different name
            redis_url = os.environ.get('REDISCLOUD_URL', os.environ.get('REDIS_URL'))
            
            # Log the URL (with masked password)
            if redis_url:
                # Mask the password in logs for security
                masked_url = redis_url
                if '@' in masked_url:
                    prefix, suffix = masked_url.split('@', 1)
                    if ':' in prefix and '//' in prefix:
                        protocol_user = prefix.split(':', 1)[0]
                        masked_url = f"{protocol_user}:****@{suffix}"
                logger.info(f"Redis URL found: {masked_url}")
            else:
                logger.warning("❌ No REDISCLOUD_URL or REDIS_URL found in environment variables!")
                logger.info(f"Available environment vars: {[k for k in os.environ.keys() if 'REDIS' in k]}")
                return False
                
            logger.info(f"Connecting to Redis...")
            self.redis = redis.from_url(
                redis_url, 
                decode_responses=True,  # Return strings instead of bytes
                encoding="utf-8"
            )
            
            # Test connection with timeout
            logger.info("Testing Redis connection with PING...")
            ping_result = await asyncio.wait_for(self.redis.ping(), timeout=5.0)
            logger.info(f"✅ Redis connection test successful: {ping_result}")
            self.initialized = True
            return True
            
        except asyncio.TimeoutError:
            logger.error("❌ Redis connection timeout - no response to PING")
            self.initialized = False
            return False
        except redis.exceptions.ConnectionError as ce:
            logger.error(f"❌ Redis connection error: {ce}")
            self.initialized = False
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis connection: {type(e).__name__}: {e}")
            self.initialized = False
            return False
    
    async def store_relationship(self, source_id: str, target_id: str, rel_type: str, weight: float) -> bool:
        """Store a relationship in Redis."""
        if not self.initialized or not self.redis:
            logger.warning("Cannot store relationship: Redis not initialized")
            return False
            
        try:
            # Create a unique key for this relationship
            edge_key = f"edge:{source_id}:{target_id}"
            
            # Store the relationship details
            mapping = {
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
                "weight": str(weight)  # Convert to string for Redis
            }
            
            await self.redis.hset(edge_key, mapping=mapping)
            
            # Add to source and target indices for faster lookup
            await self.redis.sadd(f"outgoing:{source_id}", f"{target_id}:{rel_type}")
            await self.redis.sadd(f"incoming:{target_id}", f"{source_id}:{rel_type}")
            
            # Keep track of all nodes
            await self.redis.sadd("all_nodes", source_id)
            await self.redis.sadd("all_nodes", target_id)
            
            # Log the first 5 successful stores to confirm it's working
            if not hasattr(self, '_relationship_count'):
                self._relationship_count = 0
            
            self._relationship_count += 1
            if self._relationship_count <= 5:
                logger.info(f"✅ Successfully stored relationship #{self._relationship_count} in Redis: {source_id} --[{rel_type}]--> {target_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing relationship in Redis: {e}")
            return False
    
    async def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Retrieve all relationships from Redis."""
        if not self.initialized or not self.redis:
            logger.warning("Cannot get relationships: Redis not initialized")
            return []
            
        try:
            relationships = []
            cursor = 0
            pattern = "edge:*"
            
            # Use scan for better performance with large datasets
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                
                if not keys:
                    if cursor == 0:
                        break
                    continue
                    
                # Get all relationship data in a pipeline for efficiency
                pipe = self.redis.pipeline()
                for key in keys:
                    pipe.hgetall(key)
                
                results = await pipe.execute()
                
                for edge_data in results:
                    if edge_data and "source_id" in edge_data:
                        # Convert weight back to float
                        if "weight" in edge_data:
                            try:
                                edge_data["weight"] = float(edge_data["weight"])
                            except (ValueError, TypeError):
                                edge_data["weight"] = 0.5  # Default if conversion fails
                        
                        relationships.append(edge_data)
                
                if cursor == 0:
                    break
            
            logger.info(f"Retrieved {len(relationships)} relationships from Redis")
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships from Redis: {e}")
            return []
    
    async def clear_all_relationships(self) -> bool:
        """Clear all relationship data (for testing or reset)."""
        if not self.initialized or not self.redis:
            return False
            
        try:
            # Get all keys matching our patterns
            patterns = ["edge:*", "outgoing:*", "incoming:*", "all_nodes"]
            all_keys = []
            
            for pattern in patterns:
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                    all_keys.extend(keys)
                    if cursor == 0:
                        break
            
            if all_keys:
                await self.redis.delete(*all_keys)
                
            logger.info(f"Cleared {len(all_keys)} Redis keys")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing relationships from Redis: {e}")
            return False
    
    async def get_node_count(self) -> int:
        """Get the total number of unique nodes in the graph."""
        if not self.initialized or not self.redis:
            return 0
            
        try:
            return await self.redis.scard("all_nodes")
        except Exception as e:
            logger.error(f"Error getting node count: {e}")
            return 0
    
    async def get_relationship_count(self) -> int:
        """Get the total number of relationships in the graph."""
        if not self.initialized or not self.redis:
            return 0
            
        try:
            cursor = 0
            count = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match="edge:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break
                
            return count
            
        except Exception as e:
            logger.error(f"Error getting relationship count: {e}")
            return 0