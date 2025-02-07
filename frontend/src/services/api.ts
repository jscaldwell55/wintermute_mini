// src/services/api.ts
import { QueryResponse, QueryRequest, OperationType, SystemStatus } from '../types';

const DEFAULT_API_URL = 'https://wintermute-staging-x-49dd432d3500.herokuapp.com';
const API_URL = (import.meta.env.VITE_API_URL as string) || DEFAULT_API_URL;

// Add a trailing slash to the base URL if it's not already there,
// *and* prepend /api/v1/.  This makes the URL construction much
// more robust and less error-prone.
const BASE_URL = (API_URL.endsWith('/') ? API_URL : API_URL + '/') + 'api/v1/';


export const queryAPI = async (query: string, windowId?: string): Promise<QueryResponse> => {
  try {
    const requestData: QueryRequest = {
      query: query,
      prompt: query,     // For now, using same text for both
      top_k: 5,         // Add explicit default
      window_id: windowId || crypto.randomUUID(),
      request_metadata: {
        operation_type: 'QUERY' as OperationType,
        window_id: windowId,
        timestamp: new Date().toISOString()
      }
    };

    console.log('Sending request:', {
      url: `${BASE_URL}query`,  // Use BASE_URL here
      data: requestData
    });


    const response = await fetch(`${BASE_URL}query`, { // Use BASE_URL
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error:', {
        status: response.status,
        statusText: response.statusText,
        errorText
      });
      throw new Error(`API Error (${response.status}): ${errorText}`);
    }

    return response.json();
  } catch (error) {
    console.error('Query API error:', error);
    throw error;
  }
};

export const getSystemHealth = async (): Promise<SystemStatus> => {
  try {
    const response = await fetch(`${BASE_URL}health`, { // Use BASE_URL
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.statusText} - ${errorText}`);
    }

    return response.json();
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
};