// src/types/index.ts

export type MemoryType = 'EPISODIC' | 'SEMANTIC';
export type OperationType = 'QUERY' | 'STORE' | 'UPDATE' | 'DELETE' | 'BATCH';
export type TabType = 'query' | 'memories' | 'system';

export interface RequestMetadata {
    trace_id?: string;
    timestamp?: string;
    operation_type: OperationType;
    window_id?: string;
    parent_trace_id?: string;
}

export interface QueryRequest {
    prompt: string;
    top_k?: number;
    window_id?: string;
    request_metadata?: RequestMetadata;
    metadata?: Record<string, unknown>;
}

export interface Memory {
    id: string;
    content: string;
    memory_type: MemoryType;
    created_at: string;
    metadata?: Record<string, unknown>;
    window_id?: string;
    semantic_vector?: number[];
    trace_id?: string;
    error?: ErrorDetail;
}

export interface ErrorDetail {
    code: string;
    message: string;
    trace_id?: string;
    timestamp: string;
    operation_type?: OperationType;
    details?: Record<string, any>;
}

export interface QueryResponse {
    matches: Memory[];
    similarity_scores: number[];
    response?: string;
    trace_id?: string;
    error?: ErrorDetail;
    metadata?: {
        memories_accessed?: number;
        processing_time?: number;
    };
}

  
  
  export interface TextToSpeechResponse {
    audio_url: string;
    response: string;
    error?: string;
  }
  
  export interface VoiceProcessResponse {
    status: string;
    audio_url?: string;
    session_id: string;
    webhook_enabled: boolean;
    error?: string;
  }
  
  export interface VoiceStatusResponse {
    status: 'processing' | 'completed';
    audio_url?: string;
    response?: string;
    timestamp?: string;
    message?: string;
    error?: string;
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