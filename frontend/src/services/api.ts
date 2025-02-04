// src/services/api.ts
import { QueryResponse } from '../types';

const DEFAULT_API_URL = 'https://wintermute-staging-x-49dd432d3500.herokuapp.com';

// Use type assertion to handle potential undefined case
const API_URL = (import.meta.env.VITE_API_URL as string) || DEFAULT_API_URL;

export const queryAPI = async (query: string): Promise<QueryResponse> => {
  try {
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: query })
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    
    return response.json();
  } catch (error) {
    console.error('Query API error:', error);
    throw error;
  }
};

export const getSystemHealth = async (): Promise<{status: string}> => {
  try {
    const response = await fetch(`${API_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    
    return response.json();
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
};