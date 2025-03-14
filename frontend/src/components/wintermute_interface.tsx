// src/components/wintermute_interface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

// Define interaction interface
interface Interaction {
  query: string;
  response: string | null;
  error: ErrorDetail | null;
  timestamp: string;
  isProcessing?: boolean;
  sender: 'user' | 'wintermute'; // Add a sender field
}

const WintermuteInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [windowId, setWindowId] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const interactionsEndRef = useRef<HTMLDivElement>(null);

  // Initial setup
  useEffect(() => {
    setWindowId(crypto.randomUUID());
    
    // Log configuration for debugging
    console.log('Component mounted, configuration status:', {
      environmentVars: {
        VITE_API_URL: import.meta.env.VITE_API_URL || 'not set'
      }
    });
  }, []);

  // Auto-scroll to bottom of interactions
  useEffect(() => {
    if (interactionsEndRef.current) {
      interactionsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [interactions]);

  // Limit interactions history
  useEffect(() => {
    if (interactions.length > 10) {
      setInteractions(prevInteractions => prevInteractions.slice(-10));
    }
  }, [interactions]);

  // Handle text query submission
  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    const submittedQuery = query;
    setQuery('');

    // Add the user's query to the interactions *immediately*
    const userInteraction: Interaction = {
        query: submittedQuery,
        response: null, // No response yet
        error: null,
        timestamp: new Date().toISOString(),
        sender: 'user', // Mark as sent by the user
        isProcessing: true, // Indicate that we're waiting for a response
    };
    setInteractions(prevInteractions => [...prevInteractions, userInteraction]);
    
    try {
      const data: QueryResponse = await queryAPI(submittedQuery, windowId);
      const newInteraction: Interaction = {
        query: submittedQuery, // Keep the original query for consistency
        response: data.error ? null : (data.response || "No response from the AI."),
        error: data.error || null,
        timestamp: new Date().toISOString(),
        sender: 'wintermute', // Mark as sent by Wintermute
        isProcessing: false,
      };

      // Replace user interaction with the response
      setInteractions(prevInteractions => {
          const updatedInteractions = [...prevInteractions];
          const userIndex = updatedInteractions.findIndex(i => i.timestamp === userInteraction.timestamp);
          if (userIndex > -1) {
              updatedInteractions[userIndex] = newInteraction;
          } else {
              // Fallback in case we could not find the user message
              updatedInteractions.push(newInteraction)
          }

          return updatedInteractions;
      });

    } catch (error) {
      console.error("API call failed:", error);
      setErrorMessage(error instanceof Error ? error.message : "Unknown error occurred");
      
      const errorInteraction: Interaction = {
        query: submittedQuery,
        response: null,
        error: {
          code: 'UNKNOWN_ERROR',
          message: error instanceof Error ? error.message : 'An unknown error occurred',
          timestamp: new Date().toISOString()
        },
        timestamp: new Date().toISOString(),
          sender: 'wintermute', // Mark error as from Wintermute
      };

      // Replace the user interaction with the error, if possible
      setInteractions(prevInteractions => {
        const updatedInteractions = [...prevInteractions];
        const userIndex = updatedInteractions.findIndex(i => i.timestamp === userInteraction.timestamp);

        if(userIndex > -1) {
            updatedInteractions[userIndex] = errorInteraction;
        } else {
            // Fallback if not found.
            updatedInteractions.push(errorInteraction);
        }
        return updatedInteractions;
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleQuery();
  };

  return (
    <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
      {/* Header/Logo Area */}
      <div className="w-full max-w-md flex justify-center mb-2">
        <h1 className="text-2xl font-bold text-blue-400">Wintermute</h1>
      </div>

      {/* Interactions History */}
      <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
        {interactions.length === 0 ? (
          <p className="text-gray-500 text-center italic">No interactions yet. Start by sending a query below.</p>
        ) : (
          interactions.map((interaction, index) => (
            <div key={index} className="mb-6 last:mb-2">
              {/* User query or AI response, distinguished by sender */}
              <div className="mb-2">
                <div className="font-bold text-blue-400 mb-1">
                    {interaction.sender === 'user' ? 'You:' : 'Wintermute:'}
                </div>
                <div className={`pl-3 border-l-2 ${interaction.sender === 'user' ? 'border-blue-400 text-gray-300' : 'border-green-400 text-gray-300 whitespace-pre-wrap'}`}>
                    {interaction.sender === 'user' ? interaction.query : interaction.response}
                    {interaction.isProcessing && interaction.sender === 'wintermute' && (
                      <span className="inline-flex ml-2">
                        <span className="animate-bounce mx-px">.</span>
                        <span className="animate-bounce animation-delay-200 mx-px">.</span>
                        <span className="animate-bounce animation-delay-400 mx-px">.</span>
                      </span>
                    )}
                </div>
              </div>

              {/* Error (only shown if it's an error from Wintermute) */}
              {interaction.error && interaction.sender === 'wintermute' ? (
                <div className="mt-2">
                  <div className="font-bold text-red-500 mb-1">Error:</div>
                  <div className="p-3 bg-red-900 border border-red-500 rounded-lg">
                    <p className="font-bold">{interaction.error.code}</p>
                    <p>{interaction.error.message}</p>
                    {interaction.error.details && (
                      <pre className="mt-2 text-sm overflow-x-auto">
                        {JSON.stringify(interaction.error.details, null, 2)}
                      </pre>
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          ))
        )}
        <div ref={interactionsEndRef} />
      </div>

      {/* Error message */}
      {errorMessage && (
        <div className="w-full max-w-md p-3 bg-red-900 border border-red-500 rounded text-white mb-4">
          {errorMessage}
          <button 
            className="ml-2 text-white underline"
            onClick={() => setErrorMessage(null)}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Input area */}
      <form onSubmit={handleSubmit} className="w-full max-w-md flex flex-col space-y-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full p-4 border rounded-lg text-gray-300 bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Enter your query here..."
          rows={4}
          disabled={loading}
        />

        <div className="flex items-center">
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="w-full px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Processing...' : 'Send Query'}
          </button>
        </div>
      </form>

      {/* Footer with version info */}
      <div className="w-full max-w-md text-center text-gray-500 text-xs mt-4">
        Wintermute v1.0 - Text-only Mode
      </div>
    </div>
  );
};

export default WintermuteInterface;