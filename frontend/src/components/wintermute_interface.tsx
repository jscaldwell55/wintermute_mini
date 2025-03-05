// src/components/wintermute_interface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI, processVoiceInput, textToSpeech, checkVoiceStatus } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';
import Vapi from "@vapi-ai/web"; // Corrected import (default import)

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
    const [errorMessage, setErrorMessage] = useState<string | null>(null); // Explicitly type
    const [vapi, setVapi] = useState<Vapi | null>(null); // Store the Vapi instance

    const interactionsEndRef = useRef<HTMLDivElement>(null);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
    const statusPollingRef = useRef<NodeJS.Timeout | null>(null);
    const currentSessionId = useRef<string | null>(null); // Store current session ID

    useEffect(() => {
        setWindowId(crypto.randomUUID());

        if (!audioPlayerRef.current) {
            const player = new Audio();
            player.addEventListener('ended', () => setAudioPlaying(false));
            audioPlayerRef.current = player;
        }

        return () => {
            if (audioPlayerRef.current) {
                audioPlayerRef.current.pause();
                audioPlayerRef.current.removeEventListener('ended', () => setAudioPlaying(false));
            }
            if (statusPollingRef.current) {
                clearInterval(statusPollingRef.current);
            }
            if (vapi) {
                vapi.stop(); // Stop the Vapi call on unmount
            }
        };
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

    // Initialize Vapi when voice is enabled
    useEffect(() => {
        if (voiceEnabled) {
            // Fetch configuration from the backend
            fetch(`${import.meta.env.VITE_API_URL || ''}/api/v1/config`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(config => {
                    // Check if we have the required configuration
                    if (!config.vapi_api_key) {
                        console.error("Vapi API key is missing");
                        setErrorMessage("Vapi API key is not configured. Please check your server configuration.");
                        setVoiceEnabled(false);
                        return;
                    }
    
                    if (!config.vapi_voice_id) {
                        console.error("Vapi voice ID is missing");
                        setErrorMessage("Vapi voice ID is not configured. Please check your server configuration.");
                        setVoiceEnabled(false);
                        return;
                    }
    
                    console.log("Config received from server:", {
                        apiKeyDefined: !!config.vapi_api_key,
                        voiceIdDefined: !!config.vapi_voice_id,
                        apiUrl: config.api_url
                    });
    
                    // Explicitly request microphone access *before* initializing Vapi
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(stream => {
                            // We have microphone access, now initialize Vapi
                            console.log("Microphone access granted, initializing Vapi");
    
                            try {
                                const newVapi = new Vapi({
                                    apiKey: config.vapi_api_key,
                                    voiceId: config.vapi_voice_id,
                                    webhookUrl: `${config.api_url}/api/v1/voice/vapi-webhook/`,
                                });
    
                                newVapi.on('assistant-response', async (data) => {
                                    if (data.type === 'final-transcript') {
                                        if (data.message !== undefined) { // Add this check
                                            handleVoiceInput(data.message);
                                        } else {
                                            // Handle the case where data.message is undefined
                                            console.error("Received 'final-transcript' event with undefined message.");
                                            // Optionally, set an error message for the user:
                                            setErrorMessage("Received incomplete transcription data from Vapi.");
                                        }
                                    } else if (data.type === 'audio') {
                                        // ...
                                    }
                                });
    
                                newVapi.on('error', (error) => {
                                    console.error("Vapi error:", error);
                                    setErrorMessage(`Vapi Error: ${error.message}`);
                                });
    
                                newVapi.on('ready', () => {
                                    console.log("Vapi is ready");
                                });
    
                                newVapi.on('started', () => {
                                    console.log("VAPI STARTED");
                                });
    
                                newVapi.on('ended', () => {
                                    console.log("VAPI END");
                                    if (currentSessionId.current) {
                                        currentSessionId.current = null; // Clear session ID on call end
                                    }
                                    setIsRecording(false); // Ensure recording is set to false
                                });
    
                                setVapi(newVapi);
                                console.log("Vapi instance created successfully");
                            } catch (error) {
                                console.error("Error creating Vapi instance:", error);
                                setErrorMessage(`Failed to initialize Vapi: ${error instanceof Error ? error.message : 'Unknown error'}`);
                                setVoiceEnabled(false);
                            }
    
                            // Stop the initial microphone stream (Vapi will request again)
                            stream.getTracks().forEach(track => track.stop());
                        })
                        .catch(err => {
                            console.error("Microphone access denied:", err);
                            setErrorMessage("Microphone access is required for voice mode. Please enable it in your browser settings.");
                            setVoiceEnabled(false); // Disable voice mode if no access
                        });
                })
                .catch(error => {
                    console.error("Failed to fetch configuration:", error);
                    setErrorMessage(`Failed to fetch configuration from server: ${error.message}`);
                    setVoiceEnabled(false);
                });
        } else {
            if (vapi) {
                console.log("Stopping and cleaning up Vapi instance");
                vapi.stop();
                setVapi(null);
            }
        }
    }, [voiceEnabled]); // Only re-initialize when voiceEnabled changes


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
                } catch (err) {
                    console.error("Text-to-speech error", err);
                }
            }

            setInteractions(prevInteractions => [...prevInteractions, newInteraction]);
        } catch (err) {
            console.error("API call failed", err);
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


    const startRecording = async () => {
        setErrorMessage(null);
        if (vapi) {
            console.log("Starting Vapi call");
            setIsRecording(true);
            currentSessionId.current = crypto.randomUUID(); // Generate a session ID
            try {
                // Add detailed logging before starting
                console.log("Vapi configuration:", {
                    apiKeyDefined: !!import.meta.env.VITE_VAPI_API_KEY,
                    voiceIdDefined: !!import.meta.env.VITE_VAPI_VOICE_ID,
                    webhookUrlDefined: !!import.meta.env.VITE_API_URL
                });

                await vapi.start(); // Start the Vapi call
                console.log("Vapi call started successfully");
            } catch (error) {
                console.error("Error starting Vapi call:", error);
                // More detailed error message
                const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                setErrorMessage(`Could not start voice call: ${errorMsg}. Please check your microphone and Vapi configuration.`);  // Corrected type
                setIsRecording(false);
            }
        } else {
            console.error("Vapi not initialized");
            setErrorMessage("Voice mode is not properly initialized. Try toggling voice mode off and on again."); // Corrected type
        }
    };

    const stopRecording = () => {
        if (vapi) {
            console.log("Stopping Vapi call");
            vapi.stop(); // Stop the Vapi call
            setIsRecording(false);
            setLoading(true); // Indicate loading while waiting for final response
        }
    };

    // This function handles the transcribed text from Vapi
    const handleVoiceInput = async (transcribedText: string) => {
        setLoading(true);
        setQuery(transcribedText); // Update the input field
        const sessionId = currentSessionId.current; // Use the stored session ID

        if (!sessionId) {
            console.error("Session ID is missing!");
            setErrorMessage("Session ID is missing. Please try again."); // Corrected type
            setLoading(false);
            return;
        }

        // Add user's question immediately, marked as processing
        const newInteractionIndex = interactions.length;
        setInteractions(prev => [...prev, {
            query: transcribedText,
            response: "Thinking...",
            error: null,
            timestamp: new Date().toISOString(),
            isProcessing: true
        }]);

        try {
            // Get the "Thinking..." placeholder audio (from your backend)
            const processResponse = await processVoiceInput(transcribedText, windowId, sessionId);

            if (!processResponse || processResponse.error) {
                throw new Error(processResponse?.error || "Processing failed");
            }
            if (processResponse.audio_url && audioPlayerRef.current) {
                audioPlayerRef.current.src = processResponse.audio_url;
                audioPlayerRef.current.play();
                setAudioPlaying(true);
            }

            // Start polling for the final response (from your backend)
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
                        setInteractions(prev => prev.map((interaction, idx) =>
                            idx === newInteractionIndex ? {
                                ...interaction,
                                response: statusData.response || interaction.response,  // Use backend response
                                audioUrl: statusData.audio_url, // Use backend audio URL
                                isProcessing: false
                            } : interaction
                        ));

                        // Play final audio if not already playing
                        if (statusData.audio_url && audioPlayerRef.current && !audioPlaying) {
                            audioPlayerRef.current.src = statusData.audio_url;
                            audioPlayerRef.current.play();
                            setAudioPlaying(true);
                        }
                    }
                } catch (error) {
                    console.error("Error checking status:", error);
                }
            }, 2000); // Check every 2 seconds

            // Timeout after 30 seconds
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
            setErrorMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`); // Corrected type
            // Remove "thinking" interaction on error:
            setInteractions(prev => {
                if (prev.length > 0 && prev[prev.length - 1].response === "Thinking...") {
                    return prev.slice(0, -1); // Remove last element
                }
                return prev;
            });

        } finally {
            setLoading(false);
        }
    };


    const playAudio = (url: string) => {
        if (audioPlayerRef.current && url) {
            if (audioPlaying) {
                audioPlayerRef.current.pause(); // Pause if already playing
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

            {/* Interactions History - Scrollable container */}
            <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
                {interactions.length === 0 ? (
                    <p className="text-gray-500 text-center italic">No interactions yet.  Start by sending a query below.</p>
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

            {/* Input area with voice button when enabled */}
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