// src/services/api.ts
import { QueryResponse } from '../types';

const DEFAULT_API_URL = 'https://wintermute-staging-x-49dd432d3500.herokuapp.com';
const API_URL = (import.meta.env.VITE_API_URL as string) || DEFAULT_API_URL;

interface QueryRequest {
  query: string;     // Add this
  prompt: string;
  top_k?: number;    // Optional since there's a default
  window_id?: string;
}

export const queryAPI = async (query: string, windowId?: string): Promise<QueryResponse> => {
  try {
    const requestData: QueryRequest = {
      query: query,      // Add this
      prompt: query,     // Keep this
      window_id: windowId || crypto.randomUUID()
    };

    console.log('Sending request:', {
      url: `${API_URL}/query`,
      data: requestData
    });

    const response = await fetch(`${API_URL}/query`, {
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

export const getSystemHealth = async (): Promise<{status: string}> => {
  try {
    const response = await fetch(`${API_URL}/health`, {
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