import { QueryResponse } from '../types';

const API_URL = process.env.VITE_API_URL || 'https://wintermute-staging-x-49dd432d3500.herokuapp.com';

export const queryAPI = async (query: string): Promise<QueryResponse> => {
  const response = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });
  
  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }
  
  return response.json();
};

export const getSystemStatus = async (): Promise<{status: string}> => {
  const response = await fetch(`${API_URL}/status`);
  
  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }
  
  return response.json();
};