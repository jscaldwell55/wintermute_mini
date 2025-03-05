import os
import time
import logging
import requests
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from api.core.memory.memory import MemorySystem
from api.utils.llm_service import LLMService
from api.utils.prompt_templates import CaseResponseTemplate
from api.dependencies import get_memory_system, get_llm_service, get_case_response_template

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

# Get Vapi API key from environment
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_STT_URL = "https://api.vapi.ai/stt"
VAPI_TTS_URL = "https://api.vapi.ai/tts"
VAPI_VOICE_ID = os.getenv("VAPI_VOICE_ID", "default")  # Default voice or specify in env

# Define models
class ProcessInputRequest(BaseModel):
    text: str
    window_id: Optional[str] = None

class TextToSpeechRequest(BaseModel):
    response: str
    voice: Optional[str] = VAPI_VOICE_ID

class VoiceResponse(BaseModel):
    audio_url: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None

@router.post("/speech-to-text/")
async def speech_to_text(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Convert speech audio to text using Vapi's STT API
    """
    if not VAPI_API_KEY:
        raise HTTPException(status_code=500, detail="VAPI_API_KEY not configured")

    try:
        trace_id = f"stt_{int(time.time())}_{os.urandom(3).hex()}"
        logger.info(f"[{trace_id}] Processing speech-to-text request")
        
        # Prepare file for upload to Vapi
        files = {"file": (file.filename, await file.read(), file.content_type)}
        headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}
        
        # Send to Vapi STT API
        response = requests.post(VAPI_STT_URL, headers=headers, files=files)
        
        if response.status_code == 200:
            text = response.json().get("text", "")
            logger.info(f"[{trace_id}] Speech transcribed: {text[:50]}...")
            return {"transcribed_text": text}
        else:
            error_msg = f"Vapi STT API error: {response.status_code}"
            logger.error(f"[{trace_id}] {error_msg}")
            return {"error": error_msg}
            
    except Exception as e:
        logger.error(f"Error in speech-to-text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

@router.post("/process-input/")
async def process_input(
    data: ProcessInputRequest,
    memory_system: MemorySystem = Depends(get_memory_system),
    llm_service: LLMService = Depends(get_llm_service),
    case_response_template: CaseResponseTemplate = Depends(get_case_response_template)
):
    """
    Process user text input through Wintermute's memory system and generate a response
    """
    try:
        trace_id = f"voice_{int(time.time())}_{os.urandom(3).hex()}"
        window_id = data.window_id or f"voice_session_{os.urandom(4).hex()}"
        
        logger.info(f"[{trace_id}] Processing voice input: {data.text[:50]}...")
        
        # Create semantic vector from user input
        query_vector = memory_system.vector_operations.create_semantic_vector(data.text)
        
        # Get relevant semantic memories (coaching knowledge)
        logger.info(f"[{trace_id}] Retrieving semantic memories")
        semantic_memories = await memory_system.query_memories(
            query_vector=query_vector,
            memory_type="SEMANTIC",
            top_k=5
        )
        
        # Get relevant episodic memories (conversation history)
        logger.info(f"[{trace_id}] Retrieving episodic memories")
        time_weighted_query = data.text
        episodic_memories = await memory_system.query_memories(
            query_vector=memory_system.vector_operations.create_semantic_vector(time_weighted_query),
            memory_type="EPISODIC",
            window_id=window_id,
            top_k=5
        )
        
        # Format prompt using Wintermute's template
        prompt = case_response_template.format(
            user_query=data.text,
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories
        )
        
        # Use Wintermute's LLM to generate response
        logger.info(f"[{trace_id}] Generating response with LLM")
        # Use a slightly lower temperature for more consistent voice responses
        temperature = 0.7
        
        logger.info(f"[{trace_id}] Using temperature: {temperature}")
        response_text = await llm_service.generate_gpt_response_async(
            prompt=prompt,
            temperature=temperature
        )
        
        # Store interaction in memory system
        logger.info(f"[{trace_id}] Storing interaction in memory")
        await memory_system.store_interaction(
            query=data.text,
            response=response_text,
            window_id=window_id
        )
        
        logger.info(f"[{trace_id}] Response generated successfully")
        return {"response": response_text, "window_id": window_id}
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")

@router.post("/text-to-speech/")
async def text_to_speech(data: TextToSpeechRequest):
    """
    Convert text to speech using Vapi's TTS API
    """
    if not VAPI_API_KEY:
        raise HTTPException(status_code=500, detail="VAPI_API_KEY not configured")
        
    try:
        trace_id = f"tts_{int(time.time())}_{os.urandom(3).hex()}"
        logger.info(f"[{trace_id}] Converting text to speech")
        
        # Optimize text for speech (optional)
        # You could add processing here to make the text more speech-friendly
        # For example, adding pauses, emphasis, etc.
        
        # Prepare request to Vapi
        headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}
        payload = {
            "text": data.response,
            "voice": data.voice or VAPI_VOICE_ID
        }
        
        # Send to Vapi TTS API
        response = requests.post(VAPI_TTS_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            audio_url = response.json().get("audio_url")
            logger.info(f"[{trace_id}] Text-to-speech successful, audio URL generated")
            return {"audio_url": audio_url, "response": data.response}
        else:
            error_msg = f"Vapi TTS API error: {response.status_code}"
            logger.error(f"[{trace_id}] {error_msg}")
            return {"error": error_msg, "response": data.response}
            
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")