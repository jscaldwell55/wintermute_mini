// src/components/WintermuteVoiceInterface.tsx
import React, { useState, useRef, useEffect } from 'react';
import { queryAPI } from '../services/api';

console.log("WintermuteVoiceInterface.tsx is being executed");

// Interface for our conversation history
interface Interaction {
  id: string;
  query: string;
  response: string | null;
  audioUrl: string | null;
  timestamp: string;
}

const WintermuteVoiceInterface: React.FC = () => {
  // State
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [windowId, setWindowId] = useState<string>('');
  const [audioPlaying, setAudioPlaying] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const interactionsEndRef = useRef<HTMLDivElement>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Initialize
  useEffect(() => {
    console.log("Initializing WintermuteVoiceInterface");
    setWindowId(crypto.randomUUID());
    
    // Add a welcome message
    setInteractions([
      {
        id: crypto.randomUUID(),
        query: "",
        response: "Hi there! I'm Wintermute, your AI tutor. Click the microphone button and ask me a question!",
        audioUrl: null,
        timestamp: new Date().toISOString()
      }
    ]);
    
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

  // Start recording
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
      
      mediaRecorder.onstop = processRecording;
      
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
      setErrorMessage("Could not access microphone. Please check permissions.");
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      console.log("Stopping voice recording");
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  };

  // Process the recording
  const processRecording = async () => {
    try {
      console.log("Processing recording");
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      
      // Create unique ID for this interaction
      const interactionId = crypto.randomUUID();
      
      // Create form data for API request
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');
      
      // Step 1: Speech to Text
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
      
      // Add the user's question to interactions immediately
      setInteractions(prev => [
        ...prev,
        {
          id: interactionId,
          query: transcribedText,
          response: "Thinking...",
          audioUrl: null,
          timestamp: new Date().toISOString()
        }
      ]);
      
      // Step 2: Process with Wintermute
      console.log("Sending transcribed text to Wintermute");
      const processResponse = await fetch('/api/v1/voice/process-input/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: transcribedText,
          window_id: windowId
        })
      });
      
      const processData = await processResponse.json();
      
      if (processData.error) {
        throw new Error(processData.error);
      }
      
      const aiResponse = processData.response;
      console.log("Wintermute response:", aiResponse?.substring(0, 50) + "...");
      
      // Step 3: Text to Speech
      console.log("Converting Wintermute response to speech");
      const ttsResponse = await fetch('/api/v1/voice/text-to-speech/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          response: aiResponse
        })
      });
      
      const ttsData = await ttsResponse.json();
      
      if (ttsData.error) {
        throw new Error(ttsData.error);
      }
      
      const audioUrl = ttsData.audio_url;
      console.log("Received audio URL:", audioUrl);
      
      // Update interaction with final response and audio URL
      setInteractions(prev => 
        prev.map(interaction => 
          interaction.id === interactionId
            ? {
                ...interaction,
                response: aiResponse,
                audioUrl: audioUrl
              }
            : interaction
        )
      );
      
      // Play the response automatically
      if (audioUrl && audioPlayerRef.current) {
        audioPlayerRef.current.src = audioUrl;
        audioPlayerRef.current.play();
        setAudioPlaying(true);
      }
      
    } catch (error) {
      console.error("Error processing recording:", error);
      setErrorMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };
  
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
      {/* Optional: Version indicator */}
      <div className="text-xs text-gray-600 absolute top-0 right-0 mr-2 mt-2">
        Voice v1.0
      </div>
      
      {/* Conversation history */}
      <div className="w-full max-w-md flex-grow overflow-y-auto mb-4 max-h-[60vh] border border-gray-700 rounded-lg p-4 bg-gray-900">
        {interactions.map((interaction, index) => (
          <div key={interaction.id} className="mb-6 last:mb-2">
            {/* Only show user's query if it exists */}
            {interaction.query && (
              <div className="mb-2">
                <div className="font-bold text-blue-400 mb-1">You:</div>
                <div className="pl-3 border-l-2 border-blue-400 text-gray-300">
                  {interaction.query}
                </div>
              </div>
            )}
            
            {/* AI response */}
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
          </div>
        ))}
        <div ref={interactionsEndRef} />
      </div>

      {/* Error message */}
      {errorMessage && (
        <div className="w-full max-w-md p-3 bg-red-900 border border-red-500 rounded text-white mb-4">
          {errorMessage}
        </div>
      )}

      {/* Voice control button */}
      <div className="w-full max-w-md flex justify-center">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
          className={`w-16 h-16 rounded-full flex items-center justify-center transition duration-300 ${
            isRecording 
              ? 'bg-red-600 hover:bg-red-700' 
              : 'bg-blue-600 hover:bg-blue-700'
          } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isProcessing ? (
            // Loading spinner
            <svg className="animate-spin h-8 w-8 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : isRecording ? (
            // Stop icon
            <svg className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <rect x="6" y="6" width="12" height="12" strokeWidth="2" />
            </svg>
          ) : (
            // Microphone icon
            <svg className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          )}
        </button>
      </div>
      
      {/* Instructions */}
      <div className="text-center text-gray-400 text-sm">
        {isRecording 
          ? "Click the button when you're done speaking" 
          : isProcessing 
            ? "Processing your question..." 
            : "Click the microphone and ask a question"}
      </div>
    </div>
  );
};

export default WintermuteVoiceInterface;