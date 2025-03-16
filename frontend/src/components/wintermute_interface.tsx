// src/components/wintermute_interface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

// Define interaction interface
interface Interaction {
  id: string;
  query: string;
  response: string | null;
  error: ErrorDetail | null;
  timestamp: string;
  sender: 'user' | 'wintermute';
  isProcessing?: boolean;
}

const WintermuteInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [windowId, setWindowId] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const interactionsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setWindowId(crypto.randomUUID());
    console.log('Component mounted, configuration status:', {
      environmentVars: {
        VITE_API_URL: import.meta.env.VITE_API_URL || 'not set'
      }
    });
  }, []);

  useEffect(() => {
    if (interactionsEndRef.current) {
      interactionsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [interactions]);

  useEffect(() => {
    if (interactions.length > 10) {
      setInteractions(prevInteractions => prevInteractions.slice(-10));
    }
  }, [interactions]);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    const submittedQuery = query;
    setQuery('');

    const interactionTimestamp = new Date().toISOString();
    const interactionId = crypto.randomUUID();

    const userInteraction: Interaction = {
      id: interactionId,
      query: submittedQuery,
      response: null,
      error: null,
      timestamp: interactionTimestamp,
      sender: 'user',
      isProcessing: true,
    };
    setInteractions(prevInteractions => [...prevInteractions, userInteraction]);

    try {
      const data: QueryResponse = await queryAPI(submittedQuery, windowId);
      const newInteraction: Interaction = {
        id: interactionId,
        query: submittedQuery,
        response: data.error ? null : (data.response || "No response from the AI."),
        error: data.error || null,
        timestamp: new Date().toISOString(),
        sender: 'wintermute',
        isProcessing: false,
      };

      setInteractions(prevInteractions => prevInteractions.map(i => i.id === interactionId ? newInteraction : i));

    } catch (error) {
      console.error("API call failed:", error);
      setErrorMessage(error instanceof Error ? error.message : "Unknown error occurred");

      const errorInteraction: Interaction = {
        id: interactionId,
        query: submittedQuery,
        response: null,
        error: {
          code: 'UNKNOWN_ERROR',
          message: error instanceof Error ? error.message : 'An unknown error occurred',
          timestamp: interactionTimestamp
        },
        timestamp: interactionTimestamp,
        sender: 'wintermute',
      };

      setInteractions(prevInteractions => prevInteractions.map(i => i.id === interactionId ? errorInteraction : i));
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleQuery();
  };

  return (
    <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
      <div className="w-full max-w-md flex justify-center mb-2">
        <h1 className="text-2xl font-bold text-blue-400">Wintermute</h1>
      </div>

      <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
        {interactions.length === 0 ? (
          <p className="text-gray-500 text-center italic">No interactions yet. Start by sending a query below.</p>
        ) : (
          interactions.map((interaction) => (
            <div key={interaction.id} className="mb-6 last:mb-2">
              {/* Message container with distinct styling based on sender */}
              <div className={`rounded-lg p-3 ${
                interaction.sender === 'user' 
                  ? 'bg-blue-900/30 border border-blue-500' 
                  : 'bg-emerald-900/30 border border-emerald-500'
              }`}>
                {/* Sender label */}
                <div className={`font-bold mb-2 flex items-center ${
                  interaction.sender === 'user' ? 'text-blue-400' : 'text-emerald-400'
                }`}>
                  {/* Add icon for sender */}
                  <span className={`inline-block mr-2 ${
                    interaction.sender === 'user' ? 'text-blue-400' : 'text-emerald-400'
                  }`}>
                    {interaction.sender === 'user' 
                      ? '👤' 
                      : '🤖'}
                  </span>
                  {interaction.sender === 'user' ? 'You:' : 'Wintermute:'}
                </div>
                
                {/* Message content with improved styling */}
                <div className={`ml-6 ${
                  interaction.sender === 'user' 
                    ? 'text-gray-200' 
                    : 'text-gray-200 whitespace-pre-wrap'
                }`}>
                  {interaction.sender === 'user' 
                    ? interaction.query 
                    : interaction.isProcessing 
                      ? 'Processing...' 
                      : interaction.response || 
                        (interaction.error 
                          ? `Error: ${interaction.error.message}` 
                          : 'No response')}
                </div>
              </div>
            </div>
          ))
        )}
        <div ref={interactionsEndRef} />
      </div>

      {errorMessage && (
        <div className="w-full max-w-md p-3 bg-red-900 border border-red-500 rounded text-white mb-4">
          {errorMessage}
          <button className="ml-2 text-white underline" onClick={() => setErrorMessage(null)}>
            Dismiss
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="w-full max-w-md flex flex-col space-y-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full p-4 border rounded-lg text-gray-300 bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Enter your query here..."
          rows={4}
          disabled={loading}
        />

        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="w-full px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
        >
          {loading ? 'Processing...' : 'Send Query'}
        </button>
      </form>
    </div>
  );
};

export default WintermuteInterface;