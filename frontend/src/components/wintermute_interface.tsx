// src/components/wintermute_interface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

console.log("WintermuteInterface.tsx [UPDATED VERSION] is being executed");

// Define an interface for an interaction
interface Interaction {
  id: string;
  query: string;
  response: string | null;
  error: ErrorDetail | null;
  timestamp: string;
}

const WintermuteInterface: React.FC = () => {
    const [query, setQuery] = useState<string>('');
    const [interactions, setInteractions] = useState<Interaction[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [windowId, setWindowId] = useState<string>('');
    const interactionsEndRef = useRef<HTMLDivElement>(null);

    // Generate a UUID for the session window on component mount
    useEffect(() => {
        console.log("Initializing WintermuteInterface with new windowId");
        const newWindowId = crypto.randomUUID();
        setWindowId(newWindowId);
        console.log("Generated windowId:", newWindowId);
    }, []);

    // Auto-scroll to the bottom when new interactions are added
    useEffect(() => {
        if (interactionsEndRef.current) {
            interactionsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
        console.log(`Interactions updated, total count: ${interactions.length}`);
    }, [interactions]);

    // Limit interactions to the most recent 10
    useEffect(() => {
        if (interactions.length > 10) {
            console.log(`Trimming interactions to latest 10 (current count: ${interactions.length})`);
            setInteractions(prevInteractions => prevInteractions.slice(-10));
        }
    }, [interactions]);

    const handleQuery = async () => {
        if (!query.trim()) {
            console.log("Empty query, not submitting");
            return;
        }

        setLoading(true);
        const submittedQuery = query; // Capture query
        const interactionId = crypto.randomUUID(); // Generate unique ID for this interaction
        
        console.log(`Submitting query: "${submittedQuery.substring(0, 50)}${submittedQuery.length > 50 ? '...' : ''}"`);
        console.log(`Interaction ID: ${interactionId}`);
        
        // Clear the input field after we've captured the query
        setQuery('');

        try {
            // First, add a placeholder for this interaction
            const placeholderInteraction: Interaction = {
                id: interactionId,
                query: submittedQuery,
                response: "Thinking...",
                error: null,
                timestamp: new Date().toISOString()
            };
            
            // Add the placeholder to the state
            setInteractions(prev => [...prev, placeholderInteraction]);
            
            // Call the API
            const data = await queryAPI(submittedQuery, windowId);
            console.log("Received API response", data);
            
            // Update the interaction with the real response
            setInteractions(prev => 
                prev.map(interaction => 
                    interaction.id === interactionId
                        ? {
                            ...interaction,
                            response: data.response || "No response from the AI.",
                            error: data.error || null,
                            timestamp: new Date().toISOString()
                          }
                        : interaction
                )
            );
        } catch (err) {
            console.error("API call failed", err);
            
            // Create an error object
            const errorDetail: ErrorDetail = {
                code: 'UNKNOWN_ERROR',
                message: err instanceof Error ? err.message : 'An unknown error occurred',
                timestamp: new Date().toISOString()
            };
            
            // If we already added a placeholder, update it with the error
            if (interactions.some(i => i.id === interactionId)) {
                setInteractions(prev => 
                    prev.map(interaction => 
                        interaction.id === interactionId
                            ? {
                                ...interaction,
                                response: null,
                                error: errorDetail,
                                timestamp: new Date().toISOString()
                              }
                            : interaction
                    )
                );
            } else {
                // If no placeholder was added (rare case), create a new interaction with the error
                const errorInteraction: Interaction = {
                    id: interactionId,
                    query: submittedQuery,
                    response: null,
                    error: errorDetail,
                    timestamp: new Date().toISOString()
                };
                
                setInteractions(prev => [...prev, errorInteraction]);
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
            {/* Version indicator - helps debug if the new version is loaded */}
            <div className="text-xs text-gray-600 absolute top-0 right-0 mr-2 mt-2">
                v2.0
            </div>
            
            {/* Interactions History - Scrollable container */}
            <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
                {interactions.length === 0 ? (
                    <p className="text-gray-500 text-center italic">No interactions yet. Start by sending a query below.</p>
                ) : (
                    interactions.map((interaction, index) => (
                        <div key={interaction.id} className="mb-6 last:mb-2">
                            {/* User query */}
                            <div className="mb-2">
                                <div className="font-bold text-blue-400 mb-1">You:</div>
                                <div className="pl-3 border-l-2 border-blue-400 text-gray-300">
                                    {interaction.query}
                                </div>
                            </div>
                            
                            {/* AI response or error */}
                            {interaction.error ? (
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
                            ) : (
                                <div className="mt-2">
                                    <div className="font-bold text-green-400 mb-1">Wintermute:</div>
                                    <div className="pl-3 border-l-2 border-green-400 text-gray-300 whitespace-pre-wrap">
                                        {interaction.response}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))
                )}
                <div ref={interactionsEndRef} />
            </div>

            {/* Input area */}
            <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full max-w-md p-4 border rounded-lg text-gray-300 bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your query here..."
                rows={4}
            />
            <button
                onClick={handleQuery}
                disabled={loading}
                className="px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {loading ? 'Processing...' : 'Send Query'}
            </button>
        </div>
    );
};

export default WintermuteInterface;