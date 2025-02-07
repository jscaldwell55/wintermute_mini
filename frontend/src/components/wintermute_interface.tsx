import React, { useState, useEffect } from 'react'; // useEffect is already imported
import { Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { queryAPI } from '../services/api';
import { QueryResponse } from '../types';

const WintermuteInterface: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<string | null>(null); // Store only the response string
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [windowId, setWindowId] = useState<string>('');

  useEffect(() => {
    setWindowId(crypto.randomUUID());
  }, []);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null); // Clear previous response

    try {
      const data: QueryResponse = await queryAPI(query, windowId);
      if (data.error) {
        // Handle API errors
        setError(data.error.message || 'An API error occurred.');
      } else if (data.response) {
        // Handle successful response
        setResponse(data.response);
      } else {
        // Handle no response case
        setResponse("No response from the AI."); // Or any other suitable message
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
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
                {/*  Display the response string */}
              {response && (
                <div className="mt-6 p-4 bg-white rounded-lg border">
                  <h3 className="font-semibold mb-2">Response:</h3>
                  <div className="whitespace-pre-wrap">{response}</div>
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