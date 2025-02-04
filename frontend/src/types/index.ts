// src/types/index.ts

export interface QueryRequest {
  prompt: string;
  window_id?: string;
  metadata?: Record<string, unknown>;
}

export interface QueryResponse {
  response: string;
  matches?: Memory[];
  metadata?: {
    memories_accessed?: number;
    processing_time?: number;
  };
}

export type TabType = 'query' | 'memories' | 'system';

export interface Memory {
  id: string;
  content: string;
  memory_type: 'EPISODIC' | 'SEMANTIC';  // Match backend enum
  created_at: string;
  metadata?: Record<string, unknown>;
  window_id?: string;
}

export interface SystemStatus {
  status: 'online' | 'offline' | 'degraded';
  initialization_complete: boolean;
  environment: string;
  services: {
    pinecone: {
      status: string;
      error?: string;
    };
    vector_operations: {
      status: string;
      model: string;
    };
    memory_service: {
      status: string;
      cache_enabled: boolean;
    };
  };
}