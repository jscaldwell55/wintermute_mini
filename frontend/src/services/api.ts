// src/services/api.ts
import {
  QueryResponse,
  QueryRequest,
  SystemStatus,
  ErrorDetail,
  RequestMetadata,
  // SpeechToTextResponse, // No longer needed
  TextToSpeechResponse,
} from '../types';

const DEFAULT_API_URL = 'https://wintermute-staging-x-49dd432d3500.herokuapp.com';
const API_URL = (import.meta.env.VITE_API_URL as string) || DEFAULT_API_URL;

// Add a trailing slash to the base URL if it's not already there,
// *and* prepend /api/v1/.  This makes the URL construction much
// more robust and less error-prone.
const BASE_URL = (API_URL.endsWith('/') ? API_URL : API_URL + '/') + 'api/v1/';

/**
 * Creates a standardized error response object
 */
const createErrorResponse = (code: string, message: string, details?: any): QueryResponse => {
  console.error(`API Error (${code}): ${message}`, details);

  const trace_id = crypto.randomUUID();

  const errorDetail: ErrorDetail = {
    code,
    message,
    trace_id,
    timestamp: new Date().toISOString(),
    details
  };

  // Create a response that matches the existing QueryResponse type
  return {
    matches: [],
    similarity_scores: [],
    error: errorDetail,
    trace_id
  };
};

/**
 * Send a query to the Wintermute API
 */
export const queryAPI = async (query: string, windowId?: string): Promise<QueryResponse> => {
  try {
    const requestMetadata: RequestMetadata = {
      operation_type: 'QUERY',
      timestamp: new Date().toISOString(),
      window_id: windowId,
      trace_id: crypto.randomUUID()
    };

    const requestData: QueryRequest = {
      prompt: query,     // For now, using same text for both
      top_k: 5,
      window_id: windowId,
      request_metadata: requestMetadata
    };

    console.log('Sending request:', {
      url: `${BASE_URL}query`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });

    const response = await fetch(`${BASE_URL}query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      let errorData: any;
      let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;

      try {
        // Try to parse error response as JSON
        errorData = await response.json();
        if (errorData.message) {
          errorMessage = errorData.message;
        }
      } catch (e) {
        // If parsing fails, get text instead
        const errorText = await response.text();
        errorData = { raw: errorText };
        if (errorText) {
          errorMessage += ` - ${errorText}`;
        }
      }

      // Return a structured error response instead of throwing
      return createErrorResponse(
        `API_ERROR_${response.status}`,
        errorMessage,
        errorData
      );
    }

    // Parse the JSON response
    const data = await response.json();
    console.log('API Response:', data);

    // Return the response as is, it should already match the QueryResponse type
    return data as QueryResponse;
  } catch (error) {
    console.error('Query API error:', error);

    // Instead of rethrowing, return a structured error response
    return createErrorResponse(
      'CLIENT_ERROR',
      error instanceof Error ? error.message : 'Unknown client error',
      { error }
    );
  }
};

/**
 * Get system health status
 */
export const getSystemHealth = async (): Promise<SystemStatus> => {
  try {
    console.log('Sending health check request:', {
      url: `${BASE_URL}health`,
      method: 'GET',
    });

    const response = await fetch(`${BASE_URL}health`, {
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Health check response:', data);

    return data as SystemStatus;
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
};

// REMOVED: speechToText - No longer needed

/**
 * Process voice input with immediate response placeholder
 */
export const processVoiceInput = async (text: string, windowId?: string, sessionId?: string): Promise<any> => {
  try {
    console.log('Sending voice process request:', text);

    const response = await fetch(`${BASE_URL}voice/process-input/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text,
        window_id: windowId,
        session_id: sessionId || crypto.randomUUID(), // Generate ID if not provided
        enable_webhook: true
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Voice process error:', errorText);
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    console.log('Voice process response:', data);

    return data;
  } catch (error) {
    console.error('Voice process error:', error);
    throw error;
  }
};

/**
 * Check voice processing status
 */
export const checkVoiceStatus = async (sessionId: string): Promise<any> => {
  try {
    const response = await fetch(`${BASE_URL}voice/check-status/${sessionId}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Status check error:', errorText);
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Status check error:', error);
    throw error;
  }
};

/**
 * Convert text to speech
 */
export const textToSpeech = async (text: string): Promise<TextToSpeechResponse> => {
  try {
    console.log('Sending text-to-speech request');

    const formData = new FormData();
    formData.append('text', text);

    const response = await fetch(`${BASE_URL}voice/text-to-speech/`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Text-to-speech error:', errorText);
      return {
        audio_url: '',
        response: text,
        error: `API Error: ${response.status} - ${errorText}`
      };
    }

    const data = await response.json();
    console.log('Text-to-speech response:', data);

    return {
      audio_url: data.audio_url || '',
      response: text,
      error: data.error
    };
  } catch (error) {
    console.error('Text-to-speech error:', error);
    return {
      audio_url: '',
      response: text,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};