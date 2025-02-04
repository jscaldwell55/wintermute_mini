export interface QueryResponse {
    response: string;
    metadata?: {
      memories_accessed?: number;
      processing_time?: number;
    };
  }
  
  export type TabType = 'query' | 'memories' | 'system';
  
  export interface Memory {
    id: string;
    content: string;
    type: 'episodic' | 'semantic';
    timestamp: string;
    metadata?: Record<string, unknown>;
  }
  
  export interface SystemStatus {
    status: 'online' | 'offline' | 'degraded';
    components: {
      memory_store: 'connected' | 'disconnected';
      llm_service: 'available' | 'unavailable';
      vector_operations: 'operational' | 'failed';
    };
    metrics?: {
      total_memories: number;
      response_time_ms: number;
      last_consolidation: string;
    };
  }