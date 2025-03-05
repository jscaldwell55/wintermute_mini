# In /app/api/voice_api.py
import logging
import os
import time
import asyncio
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
import requests
from typing import Optional, Dict, Any
import json

from api.dependencies import get_memory_system, get_llm_service, get_case_response_template
from api.core.memory.models import QueryRequest, QueryResponse
from api.utils.responses import create_response

# Set up enhanced logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
)

# Environment variables with validation
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_VOICE_ID = os.getenv("VAPI_VOICE_ID", "11labs_voice")  # Default voice if not specified
VAPI_WEBHOOK_URL = os.getenv("VAPI_WEBHOOK_URL", None)  # Your webhook URL

# Models for request/response
class ProcessInputRequest(BaseModel):
    text: str
    window_id: Optional[str] = None
    enable_webhook: bool = True

class WebhookRequest(BaseModel):
    audio_url: str
    session_id: str
    window_id: Optional[str] = None

# Log voice configuration at startup
def log_voice_config():
    """Log voice API configuration status"""
    if VAPI_API_KEY:
        logger.info(f"Vapi API integration configured with voice ID: {VAPI_VOICE_ID}")
        if VAPI_WEBHOOK_URL:
            logger.info(f"Vapi webhook URL configured: {VAPI_WEBHOOK_URL}")
        else:
            logger.warning("Vapi webhook URL not configured - using synchronous processing")
    else:
        logger.warning("Vapi API key not configured - voice features will be disabled")

@router.post("/speech-to-text/")
async def speech_to_text(
    audio_file: UploadFile = File(...),
):
    """
    Convert speech audio to text using Vapi API
    """
    if not VAPI_API_KEY:
        logger.error("Vapi API key not configured")
        raise HTTPException(status_code=500, detail="Voice services not configured")
        
    logger.info(f"Processing speech-to-text request: {audio_file.filename}, content_type: {audio_file.content_type}")
    
    try:
        # Read audio file content
        audio_content = await audio_file.read()
        file_size = len(audio_content)
        logger.debug(f"Audio file size: {file_size} bytes")
        
        # Prepare multipart form data for Vapi
        files = {
            'audio': (audio_file.filename, audio_content, audio_file.content_type)
        }
        
        headers = {
            'Authorization': f'Bearer {VAPI_API_KEY}'
        }
        
        # Make request to Vapi speech-to-text API
        logger.info("Sending request to Vapi STT API")
        response = requests.post(
            'https://api.vapi.ai/speech-to-text',
            headers=headers,
            files=files
        )
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Vapi STT API error: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Speech to text conversion failed: {response.text}")
        
        # Parse response
        result = response.json()
        logger.info(f"STT successful: {result.get('text', '')[:30]}...")
        
        return {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0)
        }
        
    except Exception as e:
        logger.error(f"Error in speech-to-text conversion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech to text conversion failed: {str(e)}")

@router.post("/process-input/")
async def process_input(
    data: ProcessInputRequest,
    background_tasks: BackgroundTasks,
    memory_system = Depends(get_memory_system),
    llm_service = Depends(get_llm_service),
    case_response_template = Depends(get_case_response_template)
):
    """
    Process text input through Wintermute's memory system with streaming response
    """
    if not VAPI_API_KEY:
        logger.error("Vapi API key not configured")
        raise HTTPException(status_code=500, detail="Voice services not configured")
    
    logger.info(f"Processing voice input: {data.text[:50]}...")
    session_id = f"voice_{int(time.time())}_{os.urandom(3).hex()}"
    window_id = data.window_id or f"window_{os.urandom(4).hex()}"
    
    try:
        # 1. Send an immediate placeholder response
        placeholder_text = "Thinking... let me get that answer for you."
        placeholder_payload = {
            "text": placeholder_text,
            "voice_id": VAPI_VOICE_ID
        }
        
        headers = {
            'Authorization': f'Bearer {VAPI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Sending placeholder response with voice ID: {VAPI_VOICE_ID}")
        placeholder_response = requests.post(
            'https://api.vapi.ai/text-to-speech',
            headers=headers,
            json=placeholder_payload
        )
        
        if placeholder_response.status_code != 200:
            logger.error(f"Placeholder TTS failed: {placeholder_response.status_code}, {placeholder_response.text}")
            raise HTTPException(status_code=placeholder_response.status_code, 
                               detail=f"Placeholder speech generation failed: {placeholder_response.text}")
        
        placeholder_result = placeholder_response.json()
        audio_url = placeholder_result.get("audio_url")
        
        # 2. Process Wintermute AI response asynchronously
        if data.enable_webhook and VAPI_WEBHOOK_URL:
            logger.info(f"Adding background task for webhook-based response generation, session: {session_id}")
            background_tasks.add_task(
                generate_final_speech_webhook, 
                data.text, 
                window_id, 
                session_id,
                memory_system, 
                llm_service, 
                case_response_template
            )
            
            return {
                "status": "processing", 
                "audio_url": audio_url,
                "session_id": session_id,
                "webhook_enabled": True
            }
        else:
            # Fallback to synchronous processing if webhook not configured
            logger.info("Using synchronous processing (webhook disabled or not configured)")
            background_tasks.add_task(
                generate_final_speech_sync, 
                data.text, 
                window_id,
                memory_system, 
                llm_service, 
                case_response_template
            )
            
            return {
                "status": "processing", 
                "audio_url": audio_url,
                "session_id": session_id,
                "webhook_enabled": False
            }
            
    except Exception as e:
        logger.error(f"Error initiating voice processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice input processing failed: {str(e)}")

@router.post("/vapi-webhook/")
async def vapi_webhook(data: WebhookRequest):
    """
    Handle Vapi webhook - receives final voice response
    """
    logger.info(f"Received webhook from Vapi: session {data.session_id}, audio URL: {data.audio_url[:30]}...")
    
    # You could store the final audio URL in a database or cache for the frontend to retrieve
    # Here we're just acknowledging receipt
    return {
        "status": "success", 
        "session_id": data.session_id,
        "final_audio_url": data.audio_url
    }

@router.post("/text-to-speech/")
async def text_to_speech(
    text: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Convert text to speech using Vapi API (standard synchronous version)
    """
    if not VAPI_API_KEY:
        logger.error("Vapi API key not configured")
        raise HTTPException(status_code=500, detail="Voice services not configured")
        
    logger.info(f"Processing text-to-speech request: {text[:50]}...")
    
    try:
        # Prepare request data for Vapi
        payload = {
            "text": text,
            "voice_id": VAPI_VOICE_ID,
            "audio_format": "mp3",
            "speed": 1.0,  # Normal speed
            "quality": "standard"
        }
        
        headers = {
            'Authorization': f'Bearer {VAPI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Make request to Vapi text-to-speech API
        logger.info(f"Sending TTS request to Vapi API with voice ID: {VAPI_VOICE_ID}")
        response = requests.post(
            'https://api.vapi.ai/text-to-speech',
            headers=headers,
            json=payload
        )
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Vapi TTS API error: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Text to speech conversion failed: {response.text}")
        
        # Parse response to get audio URL
        result = response.json()
        audio_url = result.get("audio_url")
        
        if not audio_url:
            logger.error("No audio URL in Vapi response")
            raise HTTPException(status_code=500, detail="No audio URL returned from TTS service")
            
        logger.info(f"TTS successful, audio URL generated: {audio_url[:30]}...")
        
        return {
            "audio_url": audio_url,
            "format": "mp3",
            "duration": result.get("duration", 0)
        }
        
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text to speech conversion failed: {str(e)}")

# Helper functions for background processing
async def generate_final_speech_webhook(
    user_text: str,
    window_id: str,
    session_id: str,
    memory_system,
    llm_service,
    case_response_template
):
    """
    Generate final AI response, convert to speech, and call Vapi webhook
    """
    try:
        logger.info(f"Background task: Generating response for session {session_id}")
        
        # Create query request
        query_request = QueryRequest(query=user_text)
        
        # Use memory system to process the query
        logger.info(f"Processing query through Wintermute memory system: {user_text[:30]}...")
        response = await memory_system.process_query(
            query_request, 
            llm_service, 
            case_response_template
        )
        
        logger.info(f"Response generated successfully for session {session_id}")
        response_text = response.response
        
        # Convert AI response to speech with webhook
        payload = {
            "text": response_text,
            "voice_id": VAPI_VOICE_ID,
            "webhook_url": VAPI_WEBHOOK_URL,
            "webhook_data": {
                "session_id": session_id,
                "window_id": window_id
            }
        }
        
        headers = {
            'Authorization': f'Bearer {VAPI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Sending final TTS request with webhook for session {session_id}")
        response = requests.post(
            'https://api.vapi.ai/text-to-speech',
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Final TTS webhook request failed: {response.status_code}, {response.text}")
        else:
            logger.info(f"Final TTS webhook request sent successfully for session {session_id}")
            
    except Exception as e:
        logger.error(f"Error generating final speech with webhook: {str(e)}", exc_info=True)

async def generate_final_speech_sync(
    user_text: str,
    window_id: str,
    memory_system,
    llm_service,
    case_response_template
):
    """
    Generate final AI response and convert to speech (synchronous version)
    """
    try:
        logger.info(f"Background task: Generating response for window {window_id}")
        
        # Create query request
        query_request = QueryRequest(query=user_text)
        
        # Use memory system to process the query
        logger.info(f"Processing query through Wintermute memory system: {user_text[:30]}...")
        response = await memory_system.process_query(
            query_request, 
            llm_service, 
            case_response_template
        )
        
        logger.info(f"Response generated successfully for window {window_id}")
        response_text = response.response
        
        # Convert AI response to speech (synchronous)
        payload = {
            "text": response_text,
            "voice_id": VAPI_VOICE_ID
        }
        
        headers = {
            'Authorization': f'Bearer {VAPI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Sending final TTS request for window {window_id}")
        response = requests.post(
            'https://api.vapi.ai/text-to-speech',
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Final TTS request failed: {response.status_code}, {response.text}")
        else:
            logger.info(f"Final TTS request sent successfully for window {window_id}")
            
    except Exception as e:
        logger.error(f"Error generating final speech: {str(e)}", exc_info=True)