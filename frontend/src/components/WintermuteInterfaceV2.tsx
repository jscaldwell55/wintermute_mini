// src/components/WintermuteInterfaceV2.tsx

import React, { useState, useEffect } from 'react';
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

// This console log will help us verify that the new file is being loaded
console.log("WintermuteInterfaceV2 LOADING - TIMESTAMP:", Date.now());

// Make sure we're using the imported types
interface SimpleInteraction {
  id: string;
  query: string;
  response: string | null;
  error: ErrorDetail | null; // Using the imported ErrorDetail type
  timestamp: string;
}

const WintermuteInterfaceV2: React.FC = () => {
  // Simple, clean state
  const [query, setQuery] = useState('');
  const [interactions, setInteractions] = useState<SimpleInteraction[]>([]);
  const [loading, setLoading] = useState(false);
  const [windowId] = useState(crypto.randomUUID());

  // Add a very visible indicator that V2 is loaded
  useEffect(() => {
    console.log('WintermuteInterfaceV2 mounted with windowId:', windowId);
    
    // Add a simple test interaction to verify state is working
    setInteractions([
      {
        id: 'test-interaction',
        query: 'Test query',
        response: 'Test response from initialization',
        error: null,
        timestamp: new Date().toISOString()
      }
    ]);
  }, [windowId]);

  const handleQuery = async () => {
    if (!query.trim()) return;
    
    // Log to verify the handler is being called
    console.log('Submitting query:', query);
    
    const submittedQuery = query;
    setQuery(''); // Clear input field
    setLoading(true);
    
    // Create a unique ID
    const interactionId = crypto.randomUUID();
    
    // Add a placeholder immediately - this is critical
    const newInteraction: SimpleInteraction = {
      id: interactionId,
      query: submittedQuery,
      response: '⏳ Thinking...',
      error: null,
      timestamp: new Date().toISOString()
    };
    
    // Important: Use functional update to ensure latest state
    setInteractions(prevInteractions => {
      console.log('Current interactions:', prevInteractions.length);
      return [...prevInteractions, newInteraction];
    });
    
    try {
      // Using QueryResponse type for the response
      const response: QueryResponse = await queryAPI(submittedQuery, windowId);
      console.log('Got API response:', response);
      
      // Update the interaction with real response
      setInteractions(prevInteractions => 
        prevInteractions.map(item => 
          item.id === interactionId
            ? {
                ...item, 
                response: response.response || 'No response',
                error: response.error || null
              }
            : item
        )
      );
    } catch (error) {
      console.error('Query error:', error);
      
      // Create a proper ErrorDetail object
      const errorDetail: ErrorDetail = {
        code: 'CLIENT_ERROR',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      };
      
      // Update with error
      setInteractions(prevInteractions => 
        prevInteractions.map(item => 
          item.id === interactionId
            ? {...item, response: null, error: errorDetail}
            : item
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
      {/* Very obvious indicator that V2 is loaded */}
      <div className="bg-red-500 text-white p-2 w-full text-center font-bold">
        V2 INTERFACE - {interactions.length} interactions
      </div>
      
      {/* Interactions display */}
      <div className="w-full max-w-md border-2 border-blue-500 p-4 max-h-[60vh] overflow-y-auto">
        {interactions.map(item => (
          <div key={item.id} className="mb-4 border-b pb-2">
            <p className="font-bold text-blue-400">You:</p>
            <p className="mb-2 pl-2">{item.query}</p>
            <p className="font-bold text-green-400">Wintermute:</p>
            {item.error ? (
              <div className="pl-2 text-red-500">
                <p className="font-bold">{item.error.code}</p>
                <p>{item.error.message}</p>
              </div>
            ) : (
              <p className="pl-2">{item.response}</p>
            )}
          </div>
        ))}
      </div>
      
      {/* Input area */}
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full max-w-md p-4 border rounded"
        placeholder="Enter your query here..."
        rows={4}
      />
      <button
        onClick={handleQuery}
        disabled={loading}
        className="px-6 py-2 bg-blue-500 text-white rounded"
      >
        {loading ? 'Processing...' : 'Send Query'}
      </button>
    </div>
  );
};

export default WintermuteInterfaceV2;