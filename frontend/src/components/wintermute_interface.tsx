// src/components/WintermuteInterface.tsx
import React, { useState, useEffect } from 'react';
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

console.log("WintermuteInterface.tsx is being executed");

const WintermuteInterface: React.FC = () => {
    const [query, setQuery] = useState<string>('');
    const [response, setResponse] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<ErrorDetail | null>(null);
    const [windowId, setWindowId] = useState<string>('');

    useEffect(() => {
        setWindowId(crypto.randomUUID());
    }, []);

    const handleQuery = async () => {
        if (!query.trim()) return;

        setLoading(true);
        setError(null);
        setResponse(null);
        const submittedQuery = query; //capture query
        setQuery(''); // Clear the input field *before* the API call


        try {
            //const data: QueryResponse = await queryAPI(query, windowId); // Use captured query
            const data: QueryResponse = await queryAPI(submittedQuery, windowId); // Use captured query
            if (data.error) {
                setError(data.error);
            } else if (data.response) {
                setResponse(data.response);
            } else {
                setResponse("No response from the AI.");
            }
        } catch (err) {
            setError({
                code: 'UNKNOWN_ERROR',
                message: err instanceof Error ? err.message : 'An unknown error occurred',
                timestamp: new Date().toISOString()
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
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

            {error && (
                <div className="w-full max-w-md p-4 text-red-500 bg-red-900 border border-red-500 rounded-lg">
                    <p className="font-bold">{error.code}</p>
                    <p>{error.message}</p>
                    {error.details && (
                        <pre className="mt-2 text-sm">
                            {JSON.stringify(error.details, null, 2)}
                        </pre>
                    )}
                </div>
            )}

            {response && (
                <div className="w-full max-w-md p-4 bg-gray-800 rounded-lg border border-gray-700 text-gray-300">
                    <p className="whitespace-pre-wrap">{response}</p>
                </div>
            )}
        </div>
    );
};

export default WintermuteInterface;