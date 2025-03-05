# /app/api/voice_api.py
import logging
import os
import time
import random
from datetime import datetime, timezone, timedelta
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
    session_id: Optional[str] = None  # Add session_id
    enable_webhook: bool = True

class WebhookRequest(BaseModel):
    audio_url: str
    session_id: str
    response: Optional[str] = None  # Include the 'response' field
    window_id: Optional[str] = None

# In-memory store for voice processing status
voice_responses = {}

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

        # Consistent naming with "transcribed_text" in WintermuteInterface
        return {
            "transcribed_text": result.get("text", ""),
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
    
    logger.info(f"Processing voice input: {data.text[:50]}... Session ID: {data.session_id}")
    # Use provided session_id or generate one if it's missing
    session_id = data.session_id or f"voice_{int(time.time())}_{os.urandom(3).hex()}"
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
                session_id,  # Pass session_id here as well
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
    session_id = data.session_id
    logger.info(f"Received webhook from Vapi: session {session_id}, audio URL: {data.audio_url[:30]}...")
    
    # Store the final response data
    voice_responses[session_id] = {
        "status": "completed",
        "audio_url": data.audio_url,
        "response": data.response,  # Directly use data.response
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Clean up old responses after a while (optional)
    asyncio.create_task(cleanup_old_responses())
    
    return {
        "status": "success", 
        "session_id": session_id,
        "final_audio_url": data.audio_url
    }

@router.get("/check-status/{session_id}")
async def check_status(session_id: str):
    """
    Check the status of a voice processing request
    """
    logger.info(f"Checking status for session {session_id}")
    
    if session_id in voice_responses:
        return voice_responses[session_id]
    else:
        return {
            "status": "processing",
            "message": "Still processing or session not found"
        }

async def cleanup_old_responses():
    """Remove responses older than 5 minutes to prevent memory leaks"""
    try:
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, data in voice_responses.items():
            timestamp = datetime.fromisoformat(data["timestamp"])
            if (now - timestamp).total_seconds() > 300:  # 5 minutes
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del voice_responses[session_id]
            
        logger.info(f"Cleaned up {len(to_remove)} old voice responses")
    except Exception as e:
        logger.error(f"Error cleaning up old responses: {str(e)}")

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
        
        # Create query request with the correct field name
        query_request = QueryRequest(prompt=user_text, window_id=window_id)
        
        # Process query similar to how it's done in main.py
        logger.info(f"Processing query through Wintermute memory system: {user_text[:30]}...")
        
        # Get embeddings for the user query
        user_query_embedding = await memory_system.vector_operations.create_semantic_vector(
            query_request.prompt
        )
        
        # --- Semantic Query ---
        semantic_results = await memory_system.pinecone_service.query_memories(
            query_vector=user_query_embedding,
            top_k=memory_system.settings.semantic_top_k,
            filter={"memory_type": "SEMANTIC"},
            include_metadata=True,
        )
        
        # Filter semantic memories
        semantic_memories = []
        for match, _ in semantic_results:
            content = match["metadata"]["content"]
            if len(content.split()) >= 5:  # Keep only memories with 5+ words
                semantic_memories.append(content)
        
        # --- Episodic Query ---
        episodic_results = await memory_system.pinecone_service.query_memories(
            query_vector=user_query_embedding,
            top_k=memory_system.settings.episodic_top_k,
            filter={"memory_type": "EPISODIC"},
            include_metadata=True,
        )
        
        # Process episodic memories
        episodic_memories = []
        for match in episodic_results:
            memory_data, _ = match
            created_at = memory_data["metadata"].get("created_at")
            
            try:
                time_ago = (datetime.now(timezone.utc) - created_at).total_seconds()
                if time_ago < 60:
                    time_str = f"{int(time_ago)} seconds ago"
                elif time_ago < 3600:
                    time_str = f"{int(time_ago / 60)} minutes ago"
                else:
                    time_str = f"{int(time_ago / 3600)} hours ago"
                
                episodic_memories.append(f"{time_str}: {memory_data['metadata']['content'][:200]}")
            except Exception as e:
                logger.error(f"Error processing episodic memory: {e}")
                continue
        
        # Construct the prompt
        prompt = case_response_template.format(
            query=query_request.prompt,
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories,
        )
        
        # Generate response with random temperature
        temperature = round(random.uniform(0.6, 0.9), 2)
        logger.info(f"Using temperature: {temperature} for session {session_id}")
        
        response_text = await llm_service.generate_response_async(
            prompt,
            max_tokens=500,
            temperature=temperature
        )
        
        logger.info(f"Response generated successfully for session {session_id}")
        
        # Store the interaction
        await memory_system.store_interaction_enhanced(
            query=query_request.prompt,
            response=response_text,
            window_id=window_id,
        )
        
        # Convert AI response to speech with webhook
        payload = {
            "text": response_text,
            "voice_id": VAPI_VOICE_ID,
            "webhook_url": VAPI_WEBHOOK_URL,
            "webhook_data": {
                "session_id": session_id,
                "window_id": window_id,
                "response": response_text  # Include response in webhook data
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
            # Store error in voice_responses
            voice_responses[session_id] = {
                "status": "error",
                "error": f"TTS request failed: {response.status_code}",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            logger.info(f"Final TTS webhook request sent successfully for session {session_id}")
            # In case webhook fails, at least store the text response
            voice_responses[session_id] = {
                "status": "processing_webhook",
                "response": response_text,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error generating final speech with webhook: {str(e)}", exc_info=True)
        # Store error in voice_responses
        voice_responses[session_id] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def generate_final_speech_sync(
    user_text: str,
    window_id: str,
    session_id: str,
    memory_system,
    llm_service,
    case_response_template
):
    """
    Generate final AI response and convert to speech (synchronous version)
    """
    try:
        logger.info(f"Background task: Generating response for window {window_id}, session {session_id}")
        
        # Create query request with the correct field name
        query_request = QueryRequest(prompt=user_text, window_id=window_id)
        
        # Process query similar to how it's done in main.py
        logger.info(f"Processing query through Wintermute memory system: {user_text[:30]}...")
        
        # Get embeddings for the user query
        user_query_embedding = await memory_system.vector_operations.create_semantic_vector(
            query_request.prompt
        )
        
        # --- Semantic Query ---
        semantic_results = await memory_system.pinecone_service.query_memories(
            query_vector=user_query_embedding,
            top_k=memory_system.settings.semantic_top_k,
            filter={"memory_type": "SEMANTIC"},
            include_metadata=True,
        )
        
        # Filter semantic memories
        semantic_memories = []
        for match, _ in semantic_results:
            content = match["metadata"]["content"]
            if len(content.split()) >= 5:  # Keep only memories with 5+ words
                semantic_memories.append(content)
        
        # --- Episodic Query ---
        episodic_results = await memory_system.pinecone_service.query_memories(
            query_vector=user_query_embedding,
            top_k=memory_system.settings.episodic_top_k,
            filter={"memory_type": "EPISODIC"},
            include_metadata=True,
        )
        
        # Process episodic memories
        episodic_memories = []
        for match in episodic_results:
            memory_data, _ = match
            created_at = memory_data["metadata"].get("created_at")
            
            try:
                time_ago = (datetime.now(timezone.utc) - created_at).total_seconds()
                if time_ago < 60:
                    time_str = f"{int(time_ago)} seconds ago"
                elif time_ago < 3600:
                    time_str = f"{int(time_ago / 60)} minutes ago"
                else:
                    time_str = f"{int(time_ago / 3600)} hours ago"
                
                episodic_memories.append(f"{time_str}: {memory_data['metadata']['content'][:200]}")
            except Exception as e:
                logger.error(f"Error processing episodic memory: {e}")
                continue
        
        # Construct the prompt
        prompt = case_response_template.format(
            query=query_request.prompt,
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories,
        )
        
        # Generate response with random temperature
        temperature = round(random.uniform(0.6, 0.9), 2)
        logger.info(f"Using temperature: {temperature} for session {session_id}")
        
        response_text = await llm_service.generate_response_async(
            prompt,
            max_tokens=500,
            temperature=temperature
        )
        
        logger.info(f"Response generated successfully for window {window_id}, session: {session_id}")
        
        # Store the interaction
        await memory_system.store_interaction_enhanced(
            query=query_request.prompt,
            response=response_text,
            window_id=window_id,
        )

        # Convert the response to speech using the text-to-speech endpoint
        logger.info(f"Sending TTS request for window {window_id}, session {session_id}")
        tts_response = await text_to_speech(text=response_text)

        if "audio_url" in tts_response:
            # Store the result directly
            voice_responses[session_id] = {
                "status": "completed",
                "audio_url": tts_response["audio_url"],
                "response": response_text,
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"Final TTS completed and stored for session {session_id}")
        else:
            logger.error(f"Final TTS failed for session {session_id}")
            voice_responses[session_id] = {
                "status": "error",
                "error": "Failed to generate audio",
                "response": response_text,  # At least include the text response
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error generating final speech: {str(e)}", exc_info=True)
        # Store error in voice_responses
        voice_responses[session_id] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(f"Final TTS completed and stored for session {session_id}")
    else:
        logger.error(f"Final TTS failed for session {session_id}")
            
