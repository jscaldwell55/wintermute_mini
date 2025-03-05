// src/components/WintermuteInterface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI, speechToText, processVoiceInput, textToSpeech, checkVoiceStatus } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

console.log("WintermuteInterface.tsx is being executed");

// Define an interface for an interaction
interface Interaction {
  query: string;
  response: string | null;
  error: ErrorDetail | null;
  timestamp: string;
  audioUrl?: string | null;
  isProcessing?: boolean;
}

const WintermuteInterface: React.FC = () => {
    const [query, setQuery] = useState<string>('');
    const [interactions, setInteractions] = useState<Interaction[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [windowId, setWindowId] = useState<string>('');
    const [voiceEnabled, setVoiceEnabled] = useState<boolean>(false);
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [audioPlaying, setAudioPlaying] = useState<boolean>(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    
    const interactionsEndRef = useRef<HTMLDivElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
    const statusPollingRef = useRef<NodeJS.Timeout | null>(null);

    // Generate a UUID for the session window on component mount
    useEffect(() => {
        console.log("Initializing WintermuteInterface with new windowId");
        setWindowId(crypto.randomUUID());
        
        // Set up audio player ref for voice mode
        if (!audioPlayerRef.current) {
          const player = new Audio();
          player.addEventListener('ended', () => setAudioPlaying(false));
          audioPlayerRef.current = player;
        }
        
        return () => {
          // Cleanup
          if (audioPlayerRef.current) {
            audioPlayerRef.current.pause();
            audioPlayerRef.current.removeEventListener('ended', () => setAudioPlaying(false));
          }
          if (statusPollingRef.current) {
            clearInterval(statusPollingRef.current);
          }
        };
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

    // Handle text query submission
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
            
            // If voice is enabled, convert response to speech
            if (voiceEnabled && !data.error && data.response) {
                try {
                    const ttsResponse = await textToSpeech(data.response);
                    if (ttsResponse.audio_url) {
                        newInteraction.audioUrl = ttsResponse.audio_url;
                        
                        // Play audio
                        if (audioPlayerRef.current) {
                            audioPlayerRef.current.src = ttsResponse.audio_url;
                            audioPlayerRef.current.play();
                            setAudioPlaying(true);
                        }
                    }
                } catch (err) {
                    console.error("Text-to-speech error", err);
                    // Continue without audio if TTS fails
                }
            }
            
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

    // Start voice recording
    const startRecording = async () => {
        setErrorMessage(null);
        try {
            console.log("Starting voice recording");
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };
            
            mediaRecorder.onstop = processVoiceRecording;
            
            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error("Error starting recording:", error);
            setErrorMessage("Could not access microphone. Please check permissions.");
        }
    };

    // Stop voice recording
    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            console.log("Stopping voice recording");
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            setLoading(true);
        }
    };

    // Process the voice recording
    const processVoiceRecording = async () => {
        try {
            console.log("Processing recording");
            const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
            
            // Step 1: Speech to Text
            console.log("Sending audio for speech-to-text");
            const sttResponse = await speechToText(audioBlob);
            
            if (sttResponse.error) {
                throw new Error(sttResponse.error);
            }
            
            const transcribedText = sttResponse.transcribed_text;
            console.log("Transcribed text:", transcribedText);
            
            if (!transcribedText.trim()) {
                throw new Error("Could not understand speech. Please try again.");
            }
            
            // Set the transcribed text in the input field
            setQuery(transcribedText);
            
            // Create a unique session ID for this interaction
            const sessionId = crypto.randomUUID();
            
            // Add the user's question to interactions immediately
            const newInteractionIndex = interactions.length;
            setInteractions(prev => [...prev, {
                query: transcribedText,
                response: "Thinking...",
                error: null,
                timestamp: new Date().toISOString(),
                isProcessing: true
            }]);
            
            // Process with Wintermute using the webhook approach
            console.log("Sending transcribed text to Wintermute");
            const processResponse = await processVoiceInput(transcribedText, windowId, sessionId);
            
            if (!processResponse || processResponse.error) {
                throw new Error(processResponse?.error || "Processing failed");
            }
            
            // Play the initial "thinking" audio if available
            if (processResponse.audio_url && audioPlayerRef.current) {
                audioPlayerRef.current.src = processResponse.audio_url;
                audioPlayerRef.current.play();
                setAudioPlaying(true);
            }
            
            // Start polling for the final response
            if (statusPollingRef.current) {
                clearInterval(statusPollingRef.current);
            }
            
            statusPollingRef.current = setInterval(async () => {
                try {
                    console.log(`Checking status for session ${sessionId}`);
                    const statusData = await checkVoiceStatus(sessionId);
                    
                    if (statusData.status === 'completed' && statusData.audio_url) {
                        if (statusPollingRef.current) {
                            clearInterval(statusPollingRef.current);
                            statusPollingRef.current = null;
                        }
                        
                        console.log("Received final voice response:", statusData);
                        
                        // Update the interaction with the final response
                        setInteractions(prev => prev.map((interaction, idx) => 
                            idx === newInteractionIndex ? {
                                ...interaction,
                                response: statusData.response || interaction.response,
                                audioUrl: statusData.audio_url,
                                isProcessing: false
                            } : interaction
                        ));
                        
                        // Play the final audio if we're not already playing something
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
            
            // Set up cleanup after 30 seconds (timeout)
            setTimeout(() => {
                if (statusPollingRef.current) {
                    clearInterval(statusPollingRef.current);
                    statusPollingRef.current = null;
                    
                    // Update the interaction if it's still processing
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
            console.error("Error processing recording:", error);
            setErrorMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
            
            // Remove the "thinking" interaction if there was an error
            setInteractions(prev => {
                if (prev.length > 0 && prev[prev.length - 1].response === "Thinking...") {
                    return prev.slice(0, -1);
                }
                return prev;
            });
            
        } finally {
            setLoading(false);
        }
    };
    
    // Play audio for an interaction
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
                        <input type="checkbox" 
                               className="sr-only peer" 
                               checked={voiceEnabled} 
                               onChange={() => setVoiceEnabled(!voiceEnabled)} />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </div>
                </label>
            </div>
            
            {/* Interactions History - Scrollable container */}
            <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
                {interactions.length === 0 ? (
                    <p className="text-gray-500 text-center italic">No interactions yet. Start by sending a query below.</p>
                ) : (
                    interactions.map((interaction, _index) => (
                        <div key={_index} className="mb-6 last:mb-2">
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