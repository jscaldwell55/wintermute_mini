// src/components/WintermuteInterface.tsx
import React, { useState } from 'react';
import { MessageSquare, Brain, Database, AlertCircle, Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';

// Types
interface QueryResponse {
  response: string;
  metadata?: {
    memories_accessed?: number;
    processing_time?: number;
  };
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}

// Constants
const API_URL = process.env.VITE_API_URL || 'https://wintermute-staging-x-49dd432d3500.herokuapp.com';

// Components
const TabButton: React.FC<TabButtonProps> = ({ active, onClick, icon, children }) => (
  <button
    onClick={onClick}
    className={cn(
      'flex items-center px-4 py-2 rounded-lg transition-colors',
      active ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'
    )}
  >
    {icon}
    <span className="ml-2">{children}</span>
  </button>
);

const WintermuteInterface: React.FC = () => {
  // State
  const [activeTab, setActiveTab] = useState<'query' | 'memories' | 'system'>('query');
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Handlers
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
      setResponse(data);
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
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Project Wintermute</h1>
          <p className="text-gray-600">AI Assistant with Semantic Memory</p>
        </header>

        <div className="flex space-x-4 mb-6">
          <TabButton
            active={activeTab === 'query'}
            onClick={() => setActiveTab('query')}
            icon={<MessageSquare className="w-5 h-5" />}
          >
            Query
          </TabButton>
          <TabButton
            active={activeTab === 'memories'}
            onClick={() => setActiveTab('memories')}
            icon={<Brain className="w-5 h-5" />}
          >
            Memories
          </TabButton>
          <TabButton
            active={activeTab === 'system'}
            onClick={() => setActiveTab('system')}
            icon={<Database className="w-5 h-5" />}
          >
            System
          </TabButton>
        </div>

        {activeTab === 'query' && (
          <Card>
            <CardHeader>
              <CardTitle>Query Interface</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full p-3 border rounded-lg h-32 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your query here..."
                />
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
              </div>

              {error && (
                <Alert variant="destructive" className="mt-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

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
            </CardContent>
          </Card>
        )}

        {activeTab === 'memories' && (
          <Card>
            <CardHeader>
              <CardTitle>Memory Management</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">Memory management interface coming soon...</p>
            </CardContent>
          </Card>
        )}

        {activeTab === 'system' && (
          <Card>
            <CardHeader>
              <CardTitle>System Status</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">System status interface coming soon...</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default WintermuteInterface;