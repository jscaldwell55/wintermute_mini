// src/components/wintermute_interface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { queryAPI } from '../services/api';
import { QueryResponse, ErrorDetail } from '../types';

console.log("WintermuteInterface.tsx [ENHANCED WITH VOICE] is being executed");

// Define an interface for an interaction
interface Interaction {
  id: string;
  query: string;
  response: string | null;
  error: ErrorDetail | null;
  audioUrl: string | null; // New field for voice response URL
  timestamp: string;
}

const WintermuteInterface: React.FC = () => {
    const [query, setQuery] = useState<string>('');
    const [interactions, setInteractions] = useState<Interaction[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [windowId, setWindowId] = useState<string>('');
    const [voiceEnabled, setVoiceEnabled] = useState<boolean>(false); // Toggle for voice responses
    const [isRecording, setIsRecording] = useState<boolean>(false);
    const [audioPlaying, setAudioPlaying] = useState<boolean>(false);
    
    // Refs
    const interactionsEndRef = useRef<HTMLDivElement>(null);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);

    // Generate a UUID for the session window on component mount
    useEffect(() => {
        console.log("Initializing WintermuteInterface with new windowId");
        const newWindowId = crypto.randomUUID();
        setWindowId(newWindowId);
        console.log("Generated windowId:", newWindowId);
        
        // Set up audio player ref
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

    // Handle voice recording
    const startRecording = async () => {
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
            
            mediaRecorder.onstop = processVoiceInput;
            
            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error("Error starting recording:", error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            console.log("Stopping voice recording");
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const processVoiceInput = async () => {
        try {
            console.log("Processing voice recording");
            setLoading(true);
            
            const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
            
            // Create form data for API request
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            
            // Send audio for speech-to-text
            console.log("Sending audio for speech-to-text");
            const sttResponse = await fetch('/api/v1/voice/speech-to-text/', {
                method: 'POST',
                body: formData
            });
            
            const sttData = await sttResponse.json();
            
            if (sttData.error) {
                throw new Error(sttData.error);
            }
            
            const transcribedText = sttData.transcribed_text;
            console.log("Transcribed text:", transcribedText);
            
            // Now that we have text, process it like a normal text query
            if (transcribedText) {
                setQuery(transcribedText);
                await handleQueryWithText(transcribedText);
            }
            
        } catch (error) {
            console.error("Error processing voice input:", error);
        } finally {
            setLoading(false);
        }
    };

    // Process a query with given text (used by both text input and voice input)
    const handleQueryWithText = async (inputText: string) => {
        if (!inputText.trim()) {
            console.log("Empty query, not submitting");
            return;
        }

        setLoading(true);
        const submittedQuery = inputText;
        const interactionId = crypto.randomUUID();
        
        console.log(`Submitting query: "${submittedQuery.substring(0, 50)}${submittedQuery.length > 50 ? '...' : ''}"`);
        console.log(`Interaction ID: ${interactionId}`);
        
        // Clear input field if it matches what we're submitting (don't clear if voice input)
        if (query === submittedQuery) {
            setQuery('');
        }

        try {
            // First, add a placeholder for this interaction
            const placeholderInteraction: Interaction = {
                id: interactionId,
                query: submittedQuery,
                response: "Thinking...",
                error: null,
                audioUrl: null,
                timestamp: new Date().toISOString()
            };
            
            // Add the placeholder to the state
            setInteractions(prev => [...prev, placeholderInteraction]);
            
            // Call the API for text response
            const data = await queryAPI(submittedQuery, windowId);
            console.log("Received API response", data);
            
            let audioUrl = null;
            
            // If voice is enabled, convert response to speech
            if (voiceEnabled && data.response) {
                try {
                    console.log("Converting response to speech");
                    const ttsResponse = await fetch('/api/v1/voice/text-to-speech/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            response: data.response
                        })
                    });
                    
                    const ttsData = await ttsResponse.json();
                    
                    if (!ttsData.error && ttsData.audio_url) {
                        audioUrl = ttsData.audio_url;
                        console.log("Received audio URL:", audioUrl);
                    }
                } catch (ttsError) {
                    console.error("Error converting to speech:", ttsError);
                    // Continue without audio if TTS fails
                }
            }
            
            // Update the interaction with the real response
            setInteractions(prev => 
                prev.map(interaction => 
                    interaction.id === interactionId
                        ? {
                            ...interaction,
                            response: data.response || "No response from the AI.",
                            error: data.error || null,
                            audioUrl: audioUrl,
                            timestamp: new Date().toISOString()
                          }
                        : interaction
                )
            );
            
            // Automatically play audio if available
            if (audioUrl && audioPlayerRef.current) {
                audioPlayerRef.current.src = audioUrl;
                audioPlayerRef.current.play();
                setAudioPlaying(true);
            }
            
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
                                audioUrl: null,
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
                    audioUrl: null,
                    timestamp: new Date().toISOString()
                };
                
                setInteractions(prev => [...prev, errorInteraction]);
            }
        } finally {
            setLoading(false);
        }
    };

    // Wrapper for text input handling
    const handleQuery = () => handleQueryWithText(query);
    
    // Play audio response
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
            {/* Version indicator - helps debug if the new version is loaded */}
            <div className="text-xs text-gray-600 absolute top-0 right-0 mr-2 mt-2">
                v2.0 with Voice
            </div>
            
            {/* Voice toggle switch */}
            <div className="w-full max-w-md flex justify-end mb-2">
                <div className="flex items-center">
                    <span className="mr-2 text-sm text-gray-400">Voice Responses</span>
                    <label className="relative inline-block w-12 h-6">
                        <input
                            type="checkbox"
                            className="opacity-0 w-0 h-0"
                            checked={voiceEnabled}
                            onChange={() => setVoiceEnabled(prev => !prev)}
                        />
                        <span className={`absolute cursor-pointer inset-0 rounded-full transition-colors ${voiceEnabled ? 'bg-blue-600' : 'bg-gray-600'}`}>
                            <span 
                                className={`absolute transition-transform w-5 h-5 bg-white rounded-full top-0.5 ${voiceEnabled ? 'translate-x-6' : 'translate-x-0.5'}`}
                            />
                        </span>
                    </label>
                </div>
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
                                    <div className="font-bold text-green-400 mb-1 flex items-center justify-between">
                                        <span>Wintermute:</span>
                                        {interaction.audioUrl && (
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

            {/* Input area with voice button */}
            <div className="w-full max-w-md flex flex-col">
                <div className="flex mb-2">
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="flex-grow p-4 border rounded-lg text-gray-300 bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="Enter your query here..."
                        rows={4}
                    />
                    
                    {/* Voice input button */}
                    <button
                        onClick={isRecording ? stopRecording : startRecording}
                        disabled={loading}
                        className={`ml-2 w-12 self-stretch rounded-lg flex items-center justify-center ${
                            isRecording 
                                ? 'bg-red-600 hover:bg-red-700' 
                                : 'bg-blue-600 hover:bg-blue-700'
                        } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                        {isRecording ? (
                            <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <rect x="6" y="6" width="12" height="12" strokeWidth="2" />
                            </svg>
                        ) : (
                            <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                            </svg>
                        )}
                    </button>
                </div>
                
                <button
                    onClick={handleQuery}
                    disabled={loading || !query.trim()}
                    className="w-full px-6 py-2 bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading ? 'Processing...' : 'Send Query'}
                </button>
                
                {/* Voice indicator */}
                {isRecording && (
                    <div className="mt-2 text-center text-red-400 animate-pulse">
                        Recording... Click the microphone button when done.
                    </div>
                )}
            </div>
        </div>
    );
};

export default WintermuteInterface;