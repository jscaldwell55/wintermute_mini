// src/components/WintermuteInterface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

console.log("WintermuteInterface.tsx is being executed");

// Define an interface for an interaction
interface Interaction {
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
        setWindowId(crypto.randomUUID());
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
        setQuery(''); // Clear the input field
        
        console.log(`Submitting query: "${submittedQuery.substring(0, 50)}${submittedQuery.length > 50 ? '...' : ''}"`);

        try {
            const data: QueryResponse = await queryAPI(submittedQuery, windowId);
            console.log("Received API response", data);
            
            // Create a new interaction
            const newInteraction: Interaction = {
                query: submittedQuery,
                response: data.error ? null : (data.response || "No response from the AI."),
                error: data.error || null,
                timestamp: new Date().toISOString()
            };
            
            // Add the new interaction to the history
            setInteractions(prevInteractions => [...prevInteractions, newInteraction]);
        } catch (err) {
            console.error("API call failed", err);
            
            // Create an interaction with error
            const errorInteraction: Interaction = {
                query: submittedQuery,
                response: null,
                error: {
                    code: 'UNKNOWN_ERROR',
                    message: err instanceof Error ? err.message : 'An unknown error occurred',
                    timestamp: new Date().toISOString()
                },
                timestamp: new Date().toISOString()
            };
            
            setInteractions(prevInteractions => [...prevInteractions, errorInteraction]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
            {/* Interactions History - Scrollable container */}
            <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
                {interactions.length === 0 ? (
                    <p className="text-gray-500 text-center italic">No interactions yet. Start by sending a query below.</p>
                ) : (
                    interactions.map((interaction, index) => (
                        <div key={index} className="mb-6 last:mb-2">
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