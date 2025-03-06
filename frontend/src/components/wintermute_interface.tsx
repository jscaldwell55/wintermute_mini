// src/components/wintermute_interface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI, processVoiceInput, textToSpeech, checkVoiceStatus } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';
import Vapi from "@vapi-ai/web";

// Declare global window interface extension
declare global {
  interface Window {
    VAPI_CONFIG?: {
      vapi_public_key: string;
      vapi_voice_id: string;
      api_url: string;
    };
  }
}

// Define interaction interface
interface Interaction {
  query: string;
  response: string | null;
  error: ErrorDetail | null;
  timestamp: string;
  audioUrl?: string | null;
  isProcessing?: boolean;
}

const WintermuteInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [windowId, setWindowId] = useState('');
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [vapi, setVapi] = useState<Vapi | null>(null);

  const interactionsEndRef = useRef<HTMLDivElement>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const statusPollingRef = useRef<NodeJS.Timeout | null>(null);
  const currentSessionId = useRef<string | null>(null);

  // Initial setup
  useEffect(() => {
    setWindowId(crypto.randomUUID());

    if (!audioPlayerRef.current) {
      const player = new Audio();
      player.addEventListener('ended', () => setAudioPlaying(false));
      audioPlayerRef.current = player;
    }

    // Log configuration for debugging
    console.log('Component mounted, configuration status:', {
      windowVapiConfig: window.VAPI_CONFIG ? 'defined' : 'undefined', 
      environmentVars: {
        VITE_API_URL: import.meta.env.VITE_API_URL || 'not set',
        VITE_vapi_public_key: import.meta.env.VITE_vapi_public_key ? 'set' : 'not set'
      }
    });

    return () => {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause();
        audioPlayerRef.current.removeEventListener('ended', () => setAudioPlaying(false));
      }
      if (statusPollingRef.current) {
        clearInterval(statusPollingRef.current);
      }
      if (vapi) {
        vapi.stop();
      }
    };
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

  // Initialize Vapi when voice is enabled
  useEffect(() => {
    if (voiceEnabled) {
      const getApiKey = async (): Promise<string> => {
        try {
          // Try API endpoint first
          const response = await fetch(`${window.location.origin}/api/v1/config`);
          if (response.ok) {
            const config = await response.json();
            if (config.vapi_public_key) {
              console.log('Using API key from config endpoint');
              return config.vapi_public_key;
            }
          }
        } catch (error) {
          console.error('Failed to fetch config from API:', error);
        }
        
        // Try window.VAPI_CONFIG second
        if (window.VAPI_CONFIG?.vapi_public_key) {
          console.log('Using API key from window.VAPI_CONFIG');
          return window.VAPI_CONFIG.vapi_public_key;
        }
        
        // Try environment variables last
        if (import.meta.env.VITE_vapi_public_key) {
          console.log('Using API key from environment variables');
          return import.meta.env.VITE_vapi_public_key;
        }
        
        throw new Error('No Vapi API key found');
      };
      
      getApiKey()
        .then(apiKey => initializeVapi(apiKey))
        .catch(error => {
          console.error('Failed to get Vapi API key:', error);
          setErrorMessage(`Could not enable voice mode: ${error.message}`);
          setVoiceEnabled(false);
        });
    } else {
      // Clean up when voice is disabled
      if (vapi) {
        vapi.stop();
        setVapi(null);
      }
    }
  }, [voiceEnabled]);

  // Initialize Vapi with API key
  const initializeVapi = (apiKey: string) => {
    return navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        try {
          // Create new Vapi instance with just the API key
          const newVapi = new Vapi(apiKey as any);
          
          // Set up event handlers
          newVapi.on('message', (data) => {
            console.log('Vapi message:', data);
            if (data.type === 'transcript' && data.transcript) {
              handleVoiceInput(data.transcript);
            }
          });

          newVapi.on('error', (error) => {
            console.error('Vapi error:', error);
            setErrorMessage(`Voice error: ${error.message}`);
          });

          newVapi.on('call-start', () => {
            console.log('Vapi call started');
          });

          newVapi.on('call-end', () => {
            console.log('Vapi call ended');
            if (currentSessionId.current) {
              currentSessionId.current = null;
            }
            setIsRecording(false);
          });
          
          setVapi(newVapi);
          console.log('Vapi initialized successfully');
          
          // Clean up microphone access
          stream.getTracks().forEach(track => track.stop());
          
          return newVapi;
        } catch (error) {
          console.error('Error initializing Vapi:', error);
          throw error;
        }
      });
  };

  // Handle text query submission
  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    const submittedQuery = query;
    setQuery('');

    try {
      const data: QueryResponse = await queryAPI(submittedQuery, windowId);
      const newInteraction: Interaction = {
        query: submittedQuery,
        response: data.error ? null : (data.response || "No response from the AI."),
        error: data.error || null,
        timestamp: new Date().toISOString()
      };

      if (voiceEnabled && !data.error && data.response) {
        try {
          const ttsResponse = await textToSpeech(data.response);
          if (ttsResponse.audio_url) {
            newInteraction.audioUrl = ttsResponse.audio_url;
            if (audioPlayerRef.current) {
              audioPlayerRef.current.src = ttsResponse.audio_url;
              audioPlayerRef.current.play();
              setAudioPlaying(true);
            }
          }
        } catch (error) {
          console.error("Text-to-speech error:", error);
        }
      }

      setInteractions(prevInteractions => [...prevInteractions, newInteraction]);
    } catch (error) {
      console.error("API call failed:", error);
      const errorInteraction: Interaction = {
        query: submittedQuery,
        response: null,
        error: {
          code: 'UNKNOWN_ERROR',
          message: error instanceof Error ? error.message : 'An unknown error occurred',
          timestamp: new Date().toISOString()
        },
        timestamp: new Date().toISOString()
      };
      setInteractions(prevInteractions => [...prevInteractions, errorInteraction]);
    } finally {
      setLoading(false);
    }
  };

  // Start voice recording
  const startRecording = async () => {
    setErrorMessage(null);
    if (vapi) {
      console.log("Starting Vapi call");
      setIsRecording(true);
      currentSessionId.current = crypto.randomUUID();
      
      try {
        // Create assistant options
        const assistantOptions = {
          transcriber: {
            provider: "deepgram",
            model: "nova-2",
            language: "en-US"
          },
          voice: {
            provider: "playht",
            voiceId: "jennifer"
          },
          model: {
            provider: "openai",
            model: "gpt-4",
            messages: [
              {
                role: "system",
                content: "You are Wintermute, a helpful AI assistant. Keep responses concise and friendly."
              }
            ]
          }
        };
        
        // Use type assertion to work around TypeScript issues
        await (vapi as any).start(assistantOptions);
        console.log("Vapi call started successfully");
      } catch (error) {
        console.error("Error starting Vapi call:", error);
        setErrorMessage(`Could not start voice: ${error instanceof Error ? error.message : 'Unknown error'}`);
        setIsRecording(false);
      }
    } else {
      console.error("Vapi not initialized");
      setErrorMessage("Voice mode is not properly initialized. Try toggling voice mode off and on again.");
    }
  };

  // Stop voice recording
  const stopRecording = () => {
    if (vapi) {
      console.log("Stopping Vapi call");
      vapi.stop();
      setIsRecording(false);
      setLoading(true);
    }
  };

  // Handle transcribed voice input
  const handleVoiceInput = async (transcribedText: string) => {
    setLoading(true);
    setQuery(transcribedText);
    const sessionId = currentSessionId.current;

    if (!sessionId) {
      console.error("Session ID is missing!");
      setErrorMessage("Session ID is missing. Please try again.");
      setLoading(false);
      return;
    }

    // Add interaction with "Thinking..." placeholder
    const newInteractionIndex = interactions.length;
    setInteractions(prev => [...prev, {
      query: transcribedText,
      response: "Thinking...",
      error: null,
      timestamp: new Date().toISOString(),
      isProcessing: true
    }]);

    try {
      // Process voice input through backend
      const processResponse = await processVoiceInput(transcribedText, windowId, sessionId);

      if (!processResponse || processResponse.error) {
        throw new Error(processResponse?.error || "Processing failed");
      }
      
      // Play "thinking" audio if available
      if (processResponse.audio_url && audioPlayerRef.current) {
        audioPlayerRef.current.src = processResponse.audio_url;
        audioPlayerRef.current.play();
        setAudioPlaying(true);
      }

      // Start polling for final response
      if (statusPollingRef.current) {
        clearInterval(statusPollingRef.current);
      }

      statusPollingRef.current = setInterval(async () => {
        try {
          const statusData = await checkVoiceStatus(sessionId);
          if (statusData.status === 'completed' && statusData.audio_url) {
            if (statusPollingRef.current) {
              clearInterval(statusPollingRef.current);
              statusPollingRef.current = null;
            }
            
            // Update interaction with final response
            setInteractions(prev => prev.map((interaction, idx) =>
              idx === newInteractionIndex ? {
                ...interaction,
                response: statusData.response || interaction.response,
                audioUrl: statusData.audio_url,
                isProcessing: false
              } : interaction
            ));

            // Play audio
            if (statusData.audio_url && audioPlayerRef.current && !audioPlaying) {
              audioPlayerRef.current.src = statusData.audio_url;
              audioPlayerRef.current.play();
              setAudioPlaying(true);
            }
          }
        } catch (error) {
          console.error("Error checking status:", error);
        }
      }, 2000);

      // Set timeout to prevent indefinite waiting
      setTimeout(() => {
        if (statusPollingRef.current) {
          clearInterval(statusPollingRef.current);
          statusPollingRef.current = null;

          setInteractions(prev => prev.map((interaction, idx) =>
            idx === newInteractionIndex && interaction.isProcessing ? {
              ...interaction,
              response: "Sorry, it's taking longer than expected. Please try again.",
              isProcessing: false
            } : interaction
          ));
        }
      }, 30000);

    } catch (error) {
      console.error("Error processing voice input:", error);
      setErrorMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      
      // Remove thinking interaction on error
      setInteractions(prev => {
        if (prev.length > 0 && prev[prev.length - 1].isProcessing) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    } finally {
      setLoading(false);
    }
  };

  // Play audio from URL
  const playAudio = (url: string) => {
    if (audioPlayerRef.current && url) {
      if (audioPlaying) {
        audioPlayerRef.current.pause();
      }

      audioPlayerRef.current.src = url;
      audioPlayerRef.current.play();
      setAudioPlaying(true);
    }
  };

  return (
    <div className="flex flex-col items-center justify-start w-full h-full p-8 space-y-6">
      {/* Voice toggle switch */}
      <div className="w-full max-w-md flex justify-end mb-2">
        <label className="inline-flex items-center cursor-pointer">
          <span className="mr-3 text-sm font-medium text-gray-400">Voice Mode</span>
          <div className="relative">
            <input
              type="checkbox"
              className="sr-only peer"
              checked={voiceEnabled}
              onChange={() => setVoiceEnabled(!voiceEnabled)}
            />
            <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
          </div>
        </label>
      </div>

      {/* Interactions History */}
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
                  <div className="font-bold text-green-400 mb-1 flex items-center justify-between">
                    <span>Wintermute:</span>
                    {voiceEnabled && interaction.audioUrl && (
                      <button
                        onClick={() => playAudio(interaction.audioUrl as string)}
                        className="text-sm bg-blue-600 hover:bg-blue-700 text-white py-1 px-2 rounded flex items-center"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Play
                      </button>
                    )}
                  </div>
                  <div className={`pl-3 border-l-2 border-green-400 text-gray-300 whitespace-pre-wrap ${interaction.isProcessing ? "italic text-gray-500" : ""}`}>
                    {interaction.response}
                    {interaction.isProcessing && (
                      <span className="inline-flex ml-2">
                        <span className="animate-bounce mx-px">.</span>
                        <span className="animate-bounce animation-delay-200 mx-px">.</span>
                        <span className="animate-bounce animation-delay-400 mx-px">.</span>
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))
        )}
        <div ref={interactionsEndRef} />
      </div>

      {/* Error message */}
      {errorMessage && (
        <div className="w-full max-w-md p-3 bg-red-900 border border-red-500 rounded text-white mb-4">
          {errorMessage}
        </div>
      )}

      {/* Input area */}
      <div className="w-full max-w-md flex flex-col space-y-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full p-4 border rounded-lg text-gray-300 bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Enter your query here..."
          rows={4}
          disabled={isRecording || loading}
        />

        <div className="flex items-center space-x-2">
          <button
            onClick={handleQuery}
            disabled={loading || isRecording || !query.trim()}
            className="flex-1 px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Processing...' : 'Send Query'}
          </button>

          {voiceEnabled && (
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={loading && !isRecording}
              className={`w-12 h-12 rounded-full flex items-center justify-center transition duration-300 ${
                isRecording
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-blue-600 hover:bg-blue-700'
              } ${loading && !isRecording ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {loading && !isRecording ? (
                // Loading spinner
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : isRecording ? (
                // Stop icon
                <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <rect x="6" y="6" width="12" height="12" strokeWidth="2" />
                </svg>
              ) : (
                // Microphone icon
                <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              )}
            </button>
          )}
        </div>

        {/* Voice mode instructions */}
        {voiceEnabled && (
          <div className="text-center text-gray-400 text-sm">
            {isRecording
              ? "Click the button when you're done speaking"
              : loading
                ? "Processing your question..."
                : "Click the microphone button to speak your question"}
          </div>
        )}
      </div>
    </div>
  );
};

export default WintermuteInterface;