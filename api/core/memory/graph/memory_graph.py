# memory_graph.py

import networkx as nx
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import uuid

from api.core.memory.models import Memory, MemoryType

logger = logging.getLogger(__name__)

class MemoryGraph:
    """
    A graph-based memory system for associative memory representation.
    Maintains an in-memory graph of memories and their relationships.
    """
    
    def __init__(self):
        # Main graph containing all memory nodes and their relationships
        self.graph = nx.DiGraph()
        
        # Counter for edge weights to track relationship strength
        self.relationship_counter = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Memory graph initialized")
    
    def add_memory_node(self, memory: Memory) -> str:
        """
        Add a memory as a node in the graph.
        
        Args:
            memory: The Memory object to add
            
        Returns:
            The node ID (memory.id)
        """
        # Use memory ID as node ID
        node_id = memory.id
        
        # Skip if node already exists
        if self.graph.has_node(node_id):
            self.logger.info(f"Node {node_id} already exists in graph, updating attributes")
            # Update node attributes just in case
            self.graph.nodes[node_id].update({
                'content': memory.content,
                'memory_type': memory.memory_type.value,
                'created_at': memory.created_at,
                'vector': memory.semantic_vector
            })
            return node_id
            
        # Node attributes
        node_attrs = {
            'id': node_id,
            'content': memory.content,
            'memory_type': memory.memory_type.value,
            'created_at': memory.created_at,
            'vector': memory.semantic_vector
        }
        
        # Add metadata if available
        if memory.metadata:
            for key, value in memory.metadata.items():
                if key not in node_attrs:  # Avoid overwriting existing attributes
                    node_attrs[key] = value
        
        # Add the node
        self.graph.add_node(node_id, **node_attrs)
        self.logger.info(f"Added memory node: {node_id} of type {memory.memory_type.value}")
        
        return node_id
    
    def add_relationship(self, 
                         source_id: str, 
                         target_id: str, 
                         rel_type: str, 
                         weight: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a relationship (edge) between two memory nodes.
        
        Args:
            source_id: Source memory node ID
            target_id: Target memory node ID
            rel_type: Relationship type (e.g., 'semantic_similarity', 'temporal_sequence', 'causality')
            weight: Relationship weight/strength (higher = stronger)
            metadata: Additional attributes for the relationship
            
        Returns:
            True if successful, False otherwise
        """
        # Validate nodes exist
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            self.logger.warning(f"Cannot create relationship: one or both nodes don't exist: {source_id}, {target_id}")
            return False
        
        # Prepare edge attributes
        edge_attrs = {
            'type': rel_type,
            'weight': weight,
            'created_at': datetime.now(timezone.utc)
        }
        
        # Add metadata if available
        if metadata:
            edge_attrs.update(metadata)
        
        # Add or update the edge
        if self.graph.has_edge(source_id, target_id):
            # Update existing edge - increase weight to strengthen relationship
            current_weight = self.graph.edges[source_id, target_id].get('weight', 0)
            edge_attrs['weight'] = current_weight + weight
            
            # Track relationship strength
            rel_key = f"{source_id}_{target_id}_{rel_type}"
            self.relationship_counter[rel_key] = self.relationship_counter.get(rel_key, 0) + 1
            
            # Update edge attributes
            self.graph.edges[source_id, target_id].update(edge_attrs)
            self.logger.info(f"Updated relationship between {source_id} and {target_id}, new weight: {edge_attrs['weight']}")
        else:
            # Create new edge
            self.graph.add_edge(source_id, target_id, **edge_attrs)
            
            # Initialize relationship counter
            rel_key = f"{source_id}_{target_id}_{rel_type}"
            self.relationship_counter[rel_key] = 1
            
            self.logger.info(f"Added new relationship ({rel_type}) from {source_id} to {target_id}")
        
        return True
    
    # In memory_graph.py or similar
    def add_temporal_relationships(self, max_time_diff_seconds: int = 3600):
        """
        Add temporal relationships between memories in chronological sequence.
        
        Args:
            max_time_diff_seconds: Maximum time difference in seconds for creating temporal relationships
            
        Returns:
            Number of relationships added
        """
        try:
            # Get all memory nodes with created_at timestamp
            memory_nodes = []
            for node_id in self.graph.nodes():
                if 'created_at' in self.graph.nodes[node_id]:
                    memory_nodes.append((node_id, self.graph.nodes[node_id]['created_at']))
                    
            # Sort nodes by creation time
            memory_nodes.sort(key=lambda x: self._parse_timestamp(x[1]))
            
            # Add relationships for sequential memories within time threshold
            relationships_added = a = 0
            for i in range(1, len(memory_nodes)):
                current_node_id, current_timestamp = memory_nodes[i]
                prev_node_id, prev_timestamp = memory_nodes[i-1]
                
                # Parse timestamps
                current_time = self._parse_timestamp(current_timestamp)
                prev_time = self._parse_timestamp(prev_timestamp)
                
                if current_time and prev_time:
                    time_diff = (current_time - prev_time).total_seconds()
                    
                    # Add relationship if within threshold
                    if time_diff <= max_time_diff_seconds:
                        # Calculate strength based on time proximity (closer = stronger)
                        strength = 1.0 - (time_diff / max_time_diff_seconds)
                        
                        # Add the temporal relationship
                        success = self.add_relationship(
                            source_id=prev_node_id,
                            target_id=current_node_id,
                            rel_type="temporal_sequence",
                            weight=strength
                        )
                        
                        if success:
                            relationships_added += 1
                        
            # Log results
            self.logger.info(f"Added {relationships_added} temporal relationships")
            return relationships_added
            
        except Exception as e:
            self.logger.error(f"Error adding temporal relationships: {e}")
            return 0
    
    def _parse_timestamp(self, timestamp):
        """Parse timestamp string or value to datetime object."""
        try:
            # Handle different timestamp formats
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                return datetime.fromtimestamp(timestamp, timezone.utc)
            elif isinstance(timestamp, str):
                # ISO format string
                return datetime.fromisoformat(timestamp.rstrip('Z')).replace(tzinfo=timezone.utc)
            elif isinstance(timestamp, datetime):
                # Already a datetime
                return timestamp
            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse timestamp {timestamp}: {e}")
            return None

    def get_memories_by_date(self, target_date):
        """
        Get memories created on a specific date.
        
        Args:
            target_date: Date to filter memories by
            
        Returns:
            List of memory node IDs from the specified date
        """
        matching_memories = []
        for node_id in self.graph.nodes():
            if 'created_at' in self.graph.nodes[node_id]:
                created_at = self._parse_timestamp(self.graph.nodes[node_id]['created_at'])
                if created_at and created_at.date() == target_date:
                    matching_memories.append(node_id)
        return matching_memories
    
    def get_connected_memories(self, memory_id: str, max_depth: int = 2) -> List[Tuple[str, float, int]]:
        """
        Get memories connected to the given memory through relationships.
        
        Args:
            memory_id: Starting memory node ID
            max_depth: Maximum traversal depth
            
        Returns:
            List of tuples (connected_memory_id, relevance_score, depth)
        """
        if not self.graph.has_node(memory_id):
            self.logger.warning(f"Memory node {memory_id} not found in graph")
            return []
        
        # BFS to find connected nodes
        connected_nodes = []
        visited = set([memory_id])
        queue = [(memory_id, 0, 1.0)]  # (node_id, depth, accumulated_relevance)
        
        while queue:
            node_id, depth, relevance = queue.pop(0)
            
            if depth > 0:  # Don't include the starting node
                connected_nodes.append((node_id, relevance, depth))
            
            if depth < max_depth:
                # Explore outgoing edges
                for neighbor in self.graph.successors(node_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        edge_weight = self.graph.edges[node_id, neighbor]['weight']
                        # Relevance decays with depth
                        new_relevance = relevance * edge_weight * (0.7 ** depth)
                        queue.append((neighbor, depth + 1, new_relevance))
        
        # Sort by relevance (descending)
        connected_nodes.sort(key=lambda x: x[1], reverse=True)
        return connected_nodes

    def find_path_between_memories(self, source_id: str, target_id: str) -> List[str]:
        """
        Find shortest path between two memory nodes.
        
        Args:
            source_id: Source memory node ID
            target_id: Target memory node ID
            
        Returns:
            List of node IDs forming the path
        """
        try:
            path = nx.shortest_path(self.graph, source=source_id, target=target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory graph."""
        return {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'memory_types': self._count_node_types(),
            'relationship_types': self._count_edge_types(),
            'is_connected': nx.is_weakly_connected(self.graph),
            'avg_clustering': nx.average_clustering(self.graph.to_undirected())
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by memory type."""
        counts = {}
        for node, attrs in self.graph.nodes(data=True):
            mem_type = attrs.get('memory_type', 'UNKNOWN')
            counts[mem_type] = counts.get(mem_type, 0) + 1
        return counts
    
    def _count_edge_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        counts = {}
        for _, _, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('type', 'UNKNOWN')
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts