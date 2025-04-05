// src/components/wintermute_interface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid'; // <<< ADDED: Import uuid
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

// Define interaction interface
interface Interaction {
  id: string; // Keep using crypto.randomUUID for interaction display key
  query: string;
  response: string | null;
  error: ErrorDetail | null;
  timestamp: string;
  sender: 'user' | 'wintermute';
  isProcessing?: boolean;
}

// <<< CHANGED: Define localStorage key (moved from inside component)
const WINDOW_ID_KEY = 'wintermuteWindowId';

const WintermuteInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [loading, setLoading] = useState(false);
  // <<< CHANGED: Initialize windowId state to null initially
  const [windowId, setWindowId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const interactionsEndRef = useRef<HTMLDivElement>(null);

  // <<< CHANGED: Modified useEffect to handle persistent windowId >>>
  useEffect(() => {
    let effectiveWindowId: string;
    const storedId = localStorage.getItem(WINDOW_ID_KEY);

    if (storedId) {
      effectiveWindowId = storedId;
      console.log('Loaded existing windowId from localStorage:', effectiveWindowId);
    } else {
      effectiveWindowId = uuidv4(); // Use uuidv4 for generating the persistent ID
      localStorage.setItem(WINDOW_ID_KEY, effectiveWindowId);
      console.log('Generated and stored new windowId:', effectiveWindowId);
    }

    setWindowId(effectiveWindowId); // Set the determined ID in state

    // Keep this log for config status if needed
    console.log('Component mounted, configuration status:', {
      environmentVars: {
        VITE_API_URL: import.meta.env.VITE_API_URL || 'not set'
      }
    });
  }, []); // Empty dependency array ensures this runs only once on mount

  useEffect(() => {
    if (interactionsEndRef.current) {
      interactionsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [interactions]);

  // Removed interaction trimming logic - add back if needed
  // useEffect(() => {
  //   if (interactions.length > 10) {
  //     setInteractions(prevInteractions => prevInteractions.slice(-10));
  //   }
  // }, [interactions]);

  const handleQuery = async () => {
    // <<< CHANGED: Check for windowId being ready >>>
    if (!query.trim() || !windowId) {
        console.warn("Query attempt blocked: Query is empty or windowId is not ready.");
        return;
    }


    setLoading(true);
    const submittedQuery = query;
    setQuery(''); // Clear input after grabbing query

    const interactionTimestamp = new Date().toISOString();
    // Use crypto.randomUUID for the temporary ID for this specific interaction instance
    const interactionDisplayId = crypto.randomUUID();

    const userInteraction: Interaction = {
      id: interactionDisplayId, // Use temporary ID for UI key
      query: submittedQuery,
      response: null,
      error: null,
      timestamp: interactionTimestamp,
      sender: 'user',
      isProcessing: true, // Mark as processing initially for AI response slot
    };
    // Add BOTH user message and AI placeholder immediately
    const aiPlaceholderInteraction: Interaction = {
        id: crypto.randomUUID(), // Give AI placeholder its own temporary key
        query: submittedQuery, // Carry over query for context if needed later
        response: null,
        error: null,
        timestamp: interactionTimestamp, // Use same initial timestamp
        sender: 'wintermute',
        isProcessing: true, // Indicate loading state
    };

    setInteractions(prevInteractions => [...prevInteractions, userInteraction, aiPlaceholderInteraction]);


    try {
      // <<< CHANGED: Pass the windowId from state >>>
      const data: QueryResponse = await queryAPI(submittedQuery, windowId);

      // Prepare update data for the AI placeholder interaction
      const newInteractionData = {
        response: data.error ? null : (data.response || "No response from the AI."),
        error: data.error || null,
        timestamp: new Date().toISOString(), // Update timestamp to response time
        isProcessing: false, // Mark as not processing
      };

      // Update the AI placeholder interaction in the array
      setInteractions(prevInteractions =>
        prevInteractions.map(interaction =>
            interaction.id === aiPlaceholderInteraction.id // Find the placeholder by its unique ID
            ? { ...interaction, ...newInteractionData }
            : interaction
        )
      );


    } catch (error) {
      console.error("API call failed:", error);
      const errorMsg = error instanceof Error ? error.message : "Unknown error occurred";
      setErrorMessage(errorMsg); // Display error banner to user

      // Prepare error update data for the AI placeholder interaction
      const errorInteractionData = {
         response: null,
         error: {
           code: 'CLIENT_SIDE_ERROR', // Indicate error happened in client
           message: errorMsg,
           timestamp: new Date().toISOString() // Update timestamp
         },
         isProcessing: false, // Mark as not processing
      };

      // Update the AI placeholder interaction in the array with error info
       setInteractions(prevInteractions =>
         prevInteractions.map(interaction =>
             interaction.id === aiPlaceholderInteraction.id // Find the placeholder
             ? { ...interaction, ...errorInteractionData }
             : interaction
         )
       );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleQuery();
  };

  // <<< ADDED: New Chat handler >>>
  const handleNewChat = () => {
    console.log('Starting new chat session.');
    localStorage.removeItem(WINDOW_ID_KEY); // Clear the stored ID
    const newId = uuidv4(); // Generate a new one using uuid
    localStorage.setItem(WINDOW_ID_KEY, newId); // Store the new one
    setWindowId(newId); // Update the state
    setInteractions([]); // Clear the chat history display
    setErrorMessage(null); // Clear any previous errors
    setQuery(''); // Clear input field
    console.log('Generated new windowId for new chat:', newId);
    // Optionally focus the input field
    document.getElementById('chat-input')?.focus(); // Example ID, adjust if needed
  };


  return (
    <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
      {/* <<< CHANGED: Header layout for button >>> */}
      <h2 className="text-yellow-400 text-xl font-bold my-4">DEPLOYMENT TEST v3 - Build Time: {new Date().toLocaleTimeString()}</h2>
      <div className="w-full max-w-md flex justify-between items-center mb-2">
        <h1 className="text-2xl font-bold text-blue-400">Wintermute</h1>
        {/* <<< ADDED: New Chat button >>> */}
        <button
           onClick={handleNewChat}
           className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 transition-colors disabled:opacity-50"
           title="Clear history and start a new session context"
           disabled={loading} // Disable while processing a query
        >
            New Chat
        </button>
      </div>

      <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
        {interactions.length === 0 ? (
          <p className="text-gray-500 text-center italic">No interactions yet. Start by sending a query below.</p>
        ) : (
          // <<< REMOVED unused 'index' parameter >>>
          interactions.map((interaction) => (
            // Use unique interaction ID for key
            <div key={interaction.id} className="mb-6 last:mb-2">
              {/* Message container */}
              <div className={`rounded-lg p-3 ${
                interaction.sender === 'user'
                  ? 'bg-blue-900/30 border border-blue-500'
                  : 'bg-emerald-900/30 border border-emerald-500'
              }`}>
                {/* Sender label */}
                <div className={`font-bold mb-2 flex items-center ${
                  interaction.sender === 'user' ? 'text-blue-400' : 'text-emerald-400'
                }`}>
                  <span className={`inline-block mr-2 ${
                    interaction.sender === 'user' ? 'text-blue-400' : 'text-emerald-400'
                  }`}>
                    {interaction.sender === 'user' ? '👤' : '🤖'}
                  </span>
                  {interaction.sender === 'user' ? 'You:' : 'Wintermute:'}
                </div>

                {/* Message content */}
                <div className={`ml-6 ${
                  interaction.sender === 'user'
                    ? 'text-gray-200' // Normal text for user
                    : 'text-gray-200 whitespace-pre-wrap' // Keep pre-wrap for AI responses
                }`}>
                   {/* Show query for user, handle AI response/processing/error */}
                   {interaction.sender === 'user'
                     ? interaction.query // Display user query
                     : interaction.isProcessing // Check if AI response is processing
                       ? <span className="italic text-gray-400">Processing...</span> // Indicate processing
                       : interaction.error // Check for error next
                         ? <span className="text-red-400 italic">Error: {interaction.error.message}</span> // Display error message
                         : interaction.response || <span className="italic text-gray-500">No response</span> // Display AI response or fallback
                   }
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
          id="chat-input" // Added ID for focus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full p-4 border rounded-lg text-gray-300 bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-70"
          placeholder="Enter your query here..."
          rows={4}
          // <<< CHANGED: Disable input while loading OR if windowId isn't ready >>>
          disabled={loading || !windowId}
        />

        <button
          type="submit"
          // <<< CHANGED: Disable button while loading OR if windowId isn't ready OR if query is empty >>>
          disabled={loading || !windowId || !query.trim()}
          className="w-full px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
        >
          {loading ? 'Processing...' : 'Send Query'}
        </button>
      </form>
    </div>
  );
};

export default WintermuteInterface;
