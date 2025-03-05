# In /app/api/voice_api.py
import logging
import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
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

# Log voice configuration at startup
def log_voice_config():
    """Log voice API configuration status"""
    if VAPI_API_KEY:
        logger.info(f"Vapi API integration configured with voice ID: {VAPI_VOICE_ID}")
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
    text: str = Form(...),
    memory_system = Depends(get_memory_system),
    llm_service = Depends(get_llm_service),
    case_response_template = Depends(get_case_response_template)
):
    """
    Process text input through Wintermute's memory system
    """
    logger.info(f"Processing voice input: {text[:50]}...")
    
    try:
        # Create query request
        query_request = QueryRequest(query=text)
        
        # Use the same processing flow as the main query endpoint
        response = await memory_system.process_query(
            query_request, 
            llm_service, 
            case_response_template
        )
        
        logger.info(f"Response generated successfully: {response.response[:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice input processing failed: {str(e)}")

@router.post("/text-to-speech/")
async def text_to_speech(
    text: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Convert text to speech using Vapi API
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
        
        # Schedule cleanup task if needed (optional)
        if background_tasks:
            # Could add task to clean up temporary files, etc.
            pass
        
        return {
            "audio_url": audio_url,
            "format": "mp3",
            "duration": result.get("duration", 0)
        }
        
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text to speech conversion failed: {str(e)}")

@router.get("/health")
async def voice_health():
    """Check voice API health"""
    if not VAPI_API_KEY:
        return {"status": "disabled", "message": "Vapi API key not configured"}
    
    try:
        # Make a lightweight call to Vapi API to verify connectivity
        headers = {
            'Authorization': f'Bearer {VAPI_API_KEY}'
        }
        
        # A simple health check call to Vapi
        response = requests.get(
            'https://api.vapi.ai/health',  # Adjust endpoint as needed
            headers=headers
        )
        
        if response.status_code == 200:
            return {
                "status": "healthy", 
                "message": f"Vapi integration configured with voice ID: {VAPI_VOICE_ID}"
            }
        else:
            return {
                "status": "unhealthy", 
                "message": f"Vapi API returned status code: {response.status_code}"
            }
            
    except Exception as e:
        logger.error(f"Voice health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}