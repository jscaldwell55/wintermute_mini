// src/components/WintermuteInterface.tsx
import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// Types
interface QueryResponse {
  response: string;
  metadata?: {
    memories_accessed?: number;
    processing_time?: number;
  };
}

// Constants
const API_URL = process.env.VITE_API_URL || 'https://wintermute-staging-x-49dd432d3500.herokuapp.com';

const WintermuteInterface: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const result = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!result.ok) {
        throw new Error(`Failed to process query: ${result.statusText}`);
      }

      const data = await result.json();
      setResponse(data); // Save the response to state
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Wintermute</h1>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Query Interface</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Query Text Input */}
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full p-3 border rounded-lg h-32 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your query here..."
              />

              {/* Send Query Button */}
              <button
                onClick={handleQuery}
                disabled={loading || !query.trim()}
                className="flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  'Send Query'
                )}
              </button>

              {/* Display Error Message */}
              {error && (
                <div className="mt-4 text-red-600 bg-red-100 p-3 rounded-lg">
                  <strong>Error:</strong> {error}
                </div>
              )}

              {/* Display Query Response */}
              {response && (
                <div className="mt-6 p-4 bg-white rounded-lg border">
                  <h3 className="font-semibold mb-2">Response:</h3>
                  <div className="whitespace-pre-wrap">{response.response}</div>
                  {response.metadata && (
                    <div className="mt-4 text-sm text-gray-500">
                      <p>Memories accessed: {response.metadata.memories_accessed}</p>
                      <p>Processing time: {response.metadata.processing_time}ms</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default WintermuteInterface;
