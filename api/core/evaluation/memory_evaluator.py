import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import asyncio

from api.core.memory.models import Memory, MemoryType, QueryRequest, QueryResponse
from api.core.memory.graph.memory_graph import MemoryGraph
from api.utils.pinecone_service import PineconeService
from api.core.vector.vector_operations import VectorOperationsImpl

from api.utils.llm_service import LLMService

logger = logging.getLogger(__name__)

class MemoryEvaluator:
    """
    Evaluates the performance of the memory system, with specific focus
    on measuring the impact of graph-based associative memory.
    """
    
    def __init__(
        self,
        pinecone_service: PineconeService,
        vector_operations: VectorOperationsImpl,
        llm_service: LLMService,
        memory_graph: Optional[MemoryGraph] = None
    ):
        self.pinecone_service = pinecone_service
        self.vector_operations = vector_operations
        self.llm_service = llm_service
        self.memory_graph = memory_graph
        self.logger = logging.getLogger(__name__)
        
        # Tracking state
        self.evaluation_sessions = {}  # Store ongoing evaluation sessions
        self.metrics = {
            "vector_only": {
                "relevance_scores": [],
                "retrieval_times": [],
                "memory_coverage": []
            },
            "graph_enhanced": {
                "relevance_scores": [],
                "retrieval_times": [],
                "memory_coverage": []
            }
        }
    
    async def evaluate_retrieval_relevance(
        self,
        query: str,
        vector_results: QueryResponse,
        graph_results: QueryResponse
    ) -> Dict[str, float]:
        """
        Evaluate the relevance of retrieved memories to the query.
        Uses LLM to score relevance and provides comparative metrics.
        
        Args:
            query: User query
            vector_results: Results from vector-only retrieval
            graph_results: Results from graph-enhanced retrieval
            
        Returns:
            Dictionary of relevance metrics
        """
        try:
            # Generate query embedding for similarity comparison
            query_vector = await self.vector_operations.create_semantic_vector(query)
            
            # Calculate vector-based relevance (baseline)
            vector_relevance = self._calculate_vector_relevance(query_vector, vector_results)
            
            # Calculate graph-enhanced relevance
            graph_relevance = self._calculate_vector_relevance(query_vector, graph_results)
            
            # For a subset of results, use LLM to evaluate semantic relevance
            llm_vector_relevance = await self._calculate_llm_relevance(query, vector_results)
            llm_graph_relevance = await self._calculate_llm_relevance(query, graph_results)
            
            # Compare result diversity (unique concepts covered)
            vector_diversity = self._calculate_memory_diversity(vector_results)
            graph_diversity = self._calculate_memory_diversity(graph_results)
            
            # Update metrics tracking
            self.metrics["vector_only"]["relevance_scores"].append(llm_vector_relevance["average_score"])
            self.metrics["graph_enhanced"]["relevance_scores"].append(llm_graph_relevance["average_score"])
            
            self.metrics["vector_only"]["memory_coverage"].append(vector_diversity["unique_concepts"])
            self.metrics["graph_enhanced"]["memory_coverage"].append(graph_diversity["unique_concepts"])
            
            # Collect and return all metrics
            return {
                "vector_only": {
                    "vector_relevance": vector_relevance,
                    "llm_relevance": llm_vector_relevance,
                    "diversity": vector_diversity
                },
                "graph_enhanced": {
                    "vector_relevance": graph_relevance,
                    "llm_relevance": llm_graph_relevance,
                    "diversity": graph_diversity
                },
                "comparison": {
                    "relevance_improvement": llm_graph_relevance["average_score"] - llm_vector_relevance["average_score"],
                    "diversity_improvement": graph_diversity["unique_concepts"] - vector_diversity["unique_concepts"],
                    "novel_information": len(self._get_unique_memories(graph_results, vector_results))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating retrieval relevance: {e}")
            return {
                "error": str(e),
                "vector_only": {},
                "graph_enhanced": {},
                "comparison": {}
            }
    
    def _calculate_vector_relevance(
        self, 
        query_vector: List[float],
        results: QueryResponse
    ) -> Dict[str, Any]:
        """Calculate relevance based on vector similarity."""
        if not results.matches:
            return {"average_relevance": 0.0, "top_relevance": 0.0, "relevance_scores": []}
            
        # Reshape query vector for sklearn
        query_vector_np = np.array(query_vector).reshape(1, -1)
        
        # Get vectors from results
        result_vectors = []
        for memory in results.matches:
            if hasattr(memory, 'semantic_vector') and memory.semantic_vector:
                result_vectors.append(memory.semantic_vector)
            else:
                # Use zero vector if not available (will give low similarity)
                result_vectors.append([0.0] * len(query_vector))
        
        # Calculate similarity
        result_vectors_np = np.array(result_vectors)
        similarities = cosine_similarity(query_vector_np, result_vectors_np)[0]
        
        return {
            "average_relevance": float(np.mean(similarities)),
            "top_relevance": float(np.max(similarities)) if len(similarities) > 0 else 0.0,
            "relevance_scores": similarities.tolist()
        }
    
    async def _calculate_llm_relevance(
        self,
        query: str,
        results: QueryResponse,
        max_memories: int = 3  # Limit to top few for LLM evaluation
    ) -> Dict[str, Any]:
        """Use LLM to evaluate semantic relevance of memories to query."""
        if not results.matches:
            return {"average_score": 0.0, "memory_scores": []}
            
        # Limit to top memories
        top_memories = results.matches[:min(max_memories, len(results.matches))]
        
        scores = []
        memory_evaluations = []
        
        for i, memory in enumerate(top_memories):
            try:
                # Prepare LLM prompt
                prompt = f"""Evaluate how relevant this memory is to the query.
                
                Query: "{query}"
                
                Memory: "{memory.content[:300]}..."
                
                Rate the relevance from 0 to 10, where:
                - 0: Completely irrelevant
                - 5: Somewhat relevant
                - 10: Highly relevant and directly addresses the query
                
                Provide only a number (0-10)."""
                
                # Get LLM response
                response = await self.llm_service.generate_response(prompt)
                
                # Parse score
                try:
                    score = float(response.strip())
                    normalized_score = score / 10.0  # Normalize to 0-1
                    scores.append(normalized_score)
                    
                    memory_evaluations.append({
                        "memory_id": memory.id,
                        "content_snippet": memory.content[:50] + "...",
                        "relevance_score": normalized_score
                    })
                    
                except ValueError:
                    self.logger.warning(f"Failed to parse LLM relevance score: {response}")
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error evaluating memory relevance: {e}")
                continue
        
        # Calculate average score
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "average_score": average_score,
            "memory_scores": memory_evaluations
        }
    
    def _calculate_memory_diversity(self, results: QueryResponse) -> Dict[str, Any]:
        """Calculate diversity metrics for retrieved memories."""
        if not results.matches:
            return {"unique_concepts": 0, "memory_types": {}}
            
        # Analyze memory types
        memory_types = {}
        for memory in results.matches:
            mem_type = memory.memory_type.value
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        
        # Simple estimate of unique concepts
        # Use substring matching to estimate concept overlap between memories
        unique_concepts = set()
        for memory in results.matches:
            # Extract potential concepts using simple NLP-like approach
            # In a production system, you would use proper NLP here
            content = memory.content.lower()
            words = content.split()
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i+1]) > 3:  # Filter short words
                    bigram = f"{words[i]} {words[i+1]}"
                    unique_concepts.add(bigram)
        
        return {
            "unique_concepts": len(unique_concepts),
            "memory_types": memory_types
        }
    
    def _get_unique_memories(
        self,
        graph_results: QueryResponse,
        vector_results: QueryResponse
    ) -> List[str]:
        """Get memories that are unique to graph-based retrieval."""
        vector_ids = {memory.id for memory in vector_results.matches}
        unique_memories = []
        
        for memory in graph_results.matches:
            if memory.id not in vector_ids:
                unique_memories.append(memory.id)
                
        return unique_memories
    
    async def start_evaluation_session(self) -> str:
        """Start a new evaluation session."""
        session_id = str(uuid.uuid4())
        
        self.evaluation_sessions[session_id] = {
            "started_at": datetime.now(timezone.utc),
            "queries": [],
            "vector_retrievals": [],
            "graph_retrievals": [],
            "comparative_metrics": []
        }
        
        return session_id
    
    async def record_evaluation(
        self,
        session_id: str,
        query: str,
        vector_results: QueryResponse,
        graph_results: QueryResponse,
        vector_time_ms: float,
        graph_time_ms: float
    ) -> Dict[str, Any]:
        """
        Record evaluation metrics for a query.
        
        Args:
            session_id: Evaluation session ID
            query: User query
            vector_results: Results from vector-only retrieval
            graph_results: Results from graph-enhanced retrieval
            vector_time_ms: Vector retrieval time in milliseconds
            graph_time_ms: Graph retrieval time in milliseconds
            
        Returns:
            Evaluation metrics
        """
        if session_id not in self.evaluation_sessions:
            raise ValueError(f"Unknown evaluation session: {session_id}")
            
        # Update timing metrics
        self.metrics["vector_only"]["retrieval_times"].append(vector_time_ms)
        self.metrics["graph_enhanced"]["retrieval_times"].append(graph_time_ms)
        
        # Evaluate relevance
        relevance_metrics = await self.evaluate_retrieval_relevance(
            query, vector_results, graph_results
        )
        
        # Record evaluation
        self.evaluation_sessions[session_id]["queries"].append(query)
        self.evaluation_sessions[session_id]["vector_retrievals"].append({
            "time_ms": vector_time_ms,
            "result_count": len(vector_results.matches)
        })
        self.evaluation_sessions[session_id]["graph_retrievals"].append({
            "time_ms": graph_time_ms,
            "result_count": len(graph_results.matches)
        })
        self.evaluation_sessions[session_id]["comparative_metrics"].append(relevance_metrics)
        
        return relevance_metrics
    
    async def end_evaluation_session(self, session_id: str) -> Dict[str, Any]:
        """
        End an evaluation session and get summary metrics.
        
        Args:
            session_id: Evaluation session ID
            
        Returns:
            Summary metrics for the session
        """
        if session_id not in self.evaluation_sessions:
            raise ValueError(f"Unknown evaluation session: {session_id}")
            
        session = self.evaluation_sessions[session_id]
        session["ended_at"] = datetime.now(timezone.utc)
        
        # Calculate summary metrics
        vector_relevance = []
        graph_relevance = []
        relevance_improvement = []
        diversity_improvement = []
        
        for metrics in session["comparative_metrics"]:
            if "comparison" in metrics and "vector_only" in metrics and "graph_enhanced" in metrics:
                vector_relevance.append(metrics["vector_only"]["llm_relevance"]["average_score"])
                graph_relevance.append(metrics["graph_enhanced"]["llm_relevance"]["average_score"])
                relevance_improvement.append(metrics["comparison"]["relevance_improvement"])
                diversity_improvement.append(metrics["comparison"]["diversity_improvement"])
        
        # Calculate averages
        summary = {
            "session_id": session_id,
            "duration_seconds": (session["ended_at"] - session["started_at"]).total_seconds(),
            "query_count": len(session["queries"]),
            "average_metrics": {
                "vector_relevance": sum(vector_relevance) / len(vector_relevance) if vector_relevance else 0,
                "graph_relevance": sum(graph_relevance) / len(graph_relevance) if graph_relevance else 0,
                "relevance_improvement": sum(relevance_improvement) / len(relevance_improvement) if relevance_improvement else 0,
                "diversity_improvement": sum(diversity_improvement) / len(diversity_improvement) if diversity_improvement else 0,
                "vector_retrieval_ms": sum(self.metrics["vector_only"]["retrieval_times"]) / len(self.metrics["vector_only"]["retrieval_times"]) if self.metrics["vector_only"]["retrieval_times"] else 0,
                "graph_retrieval_ms": sum(self.metrics["graph_enhanced"]["retrieval_times"]) / len(self.metrics["graph_enhanced"]["retrieval_times"]) if self.metrics["graph_enhanced"]["retrieval_times"] else 0
            }
        }
        
        # Log summary
        self.logger.info(f"Evaluation session {session_id} summary: {json.dumps(summary['average_metrics'])}")
        
        return summary
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall metrics across all evaluation sessions."""
        return {
            "vector_only": {
                "avg_relevance": sum(self.metrics["vector_only"]["relevance_scores"]) / len(self.metrics["vector_only"]["relevance_scores"]) if self.metrics["vector_only"]["relevance_scores"] else 0,
                "avg_retrieval_time": sum(self.metrics["vector_only"]["retrieval_times"]) / len(self.metrics["vector_only"]["retrieval_times"]) if self.metrics["vector_only"]["retrieval_times"] else 0,
                "avg_memory_coverage": sum(self.metrics["vector_only"]["memory_coverage"]) / len(self.metrics["vector_only"]["memory_coverage"]) if self.metrics["vector_only"]["memory_coverage"] else 0
            },
            "graph_enhanced": {
                "avg_relevance": sum(self.metrics["graph_enhanced"]["relevance_scores"]) / len(self.metrics["graph_enhanced"]["relevance_scores"]) if self.metrics["graph_enhanced"]["relevance_scores"] else 0,
                "avg_retrieval_time": sum(self.metrics["graph_enhanced"]["retrieval_times"]) / len(self.metrics["graph_enhanced"]["retrieval_times"]) if self.metrics["graph_enhanced"]["retrieval_times"] else 0,
                "avg_memory_coverage": sum(self.metrics["graph_enhanced"]["memory_coverage"]) / len(self.metrics["graph_enhanced"]["memory_coverage"]) if self.metrics["graph_enhanced"]["memory_coverage"] else 0
            },
            "improvements": {
                "relevance": (sum(self.metrics["graph_enhanced"]["relevance_scores"]) / len(self.metrics["graph_enhanced"]["relevance_scores"]) if self.metrics["graph_enhanced"]["relevance_scores"] else 0) - 
                             (sum(self.metrics["vector_only"]["relevance_scores"]) / len(self.metrics["vector_only"]["relevance_scores"]) if self.metrics["vector_only"]["relevance_scores"] else 0),
                "memory_coverage": (sum(self.metrics["graph_enhanced"]["memory_coverage"]) / len(self.metrics["graph_enhanced"]["memory_coverage"]) if self.metrics["graph_enhanced"]["memory_coverage"] else 0) - 
                                   (sum(self.metrics["vector_only"]["memory_coverage"]) / len(self.metrics["vector_only"]["memory_coverage"]) if self.metrics["vector_only"]["memory_coverage"] else 0),
                "time_penalty": (sum(self.metrics["graph_enhanced"]["retrieval_times"]) / len(self.metrics["graph_enhanced"]["retrieval_times"]) if self.metrics["graph_enhanced"]["retrieval_times"] else 0) - 
                                (sum(self.metrics["vector_only"]["retrieval_times"]) / len(self.metrics["vector_only"]["retrieval_times"]) if self.metrics["vector_only"]["retrieval_times"] else 0)
            }
        }