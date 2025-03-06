# /app/api/voice_api.py
import logging
import os
import time
import random
from datetime import datetime, timezone, timedelta
import asyncio
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Response
from pydantic import BaseModel
import requests
from typing import Optional, Dict, Any
import json

from api.dependencies import get_memory_system, get_llm_service, get_case_response_template
from api.core.memory.models import QueryRequest, QueryResponse  # Assuming this exists
from api.utils.responses import create_response # and this
from api.utils.config import get_settings

# Set up enhanced logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
)

# Environment variables with validation
settings = get_settings()
VAPI_PUBLIC_KEY = os.getenv("VAPI_PUBLIC_KEY")
VAPI_VOICE_ID = os.getenv("VAPI_VOICE_ID")  # Default voice if not specified
VAPI_WEBHOOK_URL = os.getenv("VAPI_WEBHOOK_URL")  # Your webhook URL

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

class SpeechToTextRequest(BaseModel): # Model for the new speech-to-text endpoint
    text: str
    session_id: Optional[str] = None
    window_id: Optional[str] = None


# In-memory store for voice processing status
voice_responses = {}

# Log voice configuration at startup
def log_voice_config():
    if VAPI_PUBLIC_KEY:
        logger.info(f"Vapi API integration configured with voice ID: {VAPI_VOICE_ID}")
        if VAPI_WEBHOOK_URL:
            logger.info(f"Vapi webhook URL configured: {VAPI_WEBHOOK_URL}")
        else:
            logger.warning("Vapi webhook URL not configured - using synchronous processing")
    else:
        logger.warning("Vapi API key not configured - voice features will be disabled")

@router.post("/text-to-speech/")
async def text_to_speech(text: str = Form(...)):
    """
    Convert text to speech using Vapi API.  This is your *synchronous* TTS.
    """
    if not VAPI_PUBLIC_KEY:
        raise HTTPException(status_code=500, detail="Voice services not configured")

    logger.info(f"Processing text-to-speech request: {text[:50]}...")
    try:
        VAPI_TTS_URL = "https://api.vapi.ai/v1/audio/tts"  # Correct TTS endpoint
        payload = {
            "text": text,
            "voice": VAPI_VOICE_ID,  # Use 'voice' consistently
        }
        headers = {
            'Authorization': f'Bearer {VAPI_PUBLIC_KEY}',
            'Content-Type': 'application/json'
        }
        response = requests.post(VAPI_TTS_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        audio_url = result.get("audio_url")

        if not audio_url:
            raise HTTPException(status_code=500, detail="No audio URL returned")

        return {"audio_url": audio_url, "format": "mp3"}

    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Vapi failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vapi request failed: {e}")
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-input/")
async def process_input(
    data: ProcessInputRequest,
    background_tasks: BackgroundTasks,
    memory_system = Depends(get_memory_system),
    llm_service = Depends(get_llm_service),
    case_response_template = Depends(get_case_response_template)
):
    """Process text input (initial placeholder)"""

    if not VAPI_PUBLIC_KEY:
        raise HTTPException(status_code=500, detail="Vapi API key not set")

    session_id = data.session_id or f"voice_{int(time.time())}_{os.urandom(3).hex()}"
    window_id = data.window_id or f"window_{os.urandom(4).hex()}"
    logger.info(f"Processing input: {data.text[:50]}..., session_id: {session_id}, window_id: {window_id}")

    try:
        # Send placeholder TTS *immediately*
        placeholder_response = await text_to_speech(text="Thinking... let me get that answer for you.")
        audio_url = placeholder_response["audio_url"]

        # Choose webhook or synchronous processing
        if data.enable_webhook and VAPI_WEBHOOK_URL:
            background_tasks.add_task(
                generate_final_speech_webhook,
                data.text, window_id, session_id, memory_system, llm_service, case_response_template
            )
            webhook_enabled = True
        else:
            background_tasks.add_task(
                generate_final_speech_sync,
                data.text, window_id, session_id, memory_system, llm_service, case_response_template
            )
            webhook_enabled = False

        return {
            "status": "processing",
            "audio_url": audio_url,
            "session_id": session_id,
            "webhook_enabled": webhook_enabled,
        }

    except HTTPException as e:  # Catch HTTP exceptions from text_to_speech
        raise e
    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing input: {e}")


@router.post("/vapi-webhook/")
async def vapi_webhook(data: WebhookRequest):
    """Handle Vapi webhook"""
    session_id = data.session_id
    logger.info(f"Received Vapi webhook: session {session_id}, audio URL: {data.audio_url[:30]}...")
    voice_responses[session_id] = {
        "status": "completed",
        "audio_url": data.audio_url,
        "response": data.response,
        "timestamp": datetime.utcnow().isoformat()
    }
    asyncio.create_task(cleanup_old_responses())
    return {"status": "success", "session_id": session_id}


@router.get("/check-status/{session_id}")
async def check_status(session_id: str):
    """Check status of a voice request"""
    logger.info(f"Checking status for session: {session_id}")
    if session_id in voice_responses:
        return voice_responses[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")  # Return 404


async def cleanup_old_responses():
    """Remove old responses (5+ minutes)"""
    now = datetime.utcnow()
    to_remove = []
    for session_id, data in voice_responses.items():
        if "timestamp" in data:  # Check if timestamp exists
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
                if (now - timestamp).total_seconds() > 300:
                    to_remove.append(session_id)
            except ValueError:
                logger.warning(f"Invalid timestamp for session: {session_id}")
                to_remove.append(session_id)  # Remove malformed entries

    for session_id in to_remove:
        del voice_responses[session_id]
        logger.info(f"Cleaned up session: {session_id}")


# NEW ENDPOINT: /speech-to-text/
@router.post("/speech-to-text/")
async def speech_to_text(
    data: SpeechToTextRequest,
    background_tasks: BackgroundTasks,
    memory_system = Depends(get_memory_system),
    llm_service = Depends(get_llm_service),
    case_response_template = Depends(get_case_response_template)

):
    """
    Handles incoming transcribed text from Vapi (likely via your frontend).
    This endpoint processes the text and initiates the response generation.
    """
    if not VAPI_PUBLIC_KEY:
        raise HTTPException(status_code=500, detail="Vapi API key not set")

    session_id = data.session_id or f"voice_{int(time.time())}_{os.urandom(3).hex()}"
    window_id = data.window_id or f"window_{os.urandom(4).hex()}"

    logger.info(f"Received speech-to-text: {data.text[:50]}..., session_id: {session_id}, window_id: {window_id}")

    # Immediately return a 202 Accepted response
    background_tasks.add_task(
        process_transcribed_text,
        data.text, window_id, session_id, memory_system, llm_service, case_response_template
    )
    return Response(status_code=202)

async def process_transcribed_text(
    user_text: str,
    window_id: str,
    session_id: str,
    memory_system,
    llm_service,
    case_response_template
):
    """
    This function handles the actual processing of the transcribed text.
    It's run as a background task.  It's essentially your original
    generate_final_speech_sync/webhook logic, but now triggered by the
    speech-to-text endpoint.
    """
    try:
        logger.info(f"Background task: Processing transcribed text for session {session_id}")

        # --- (Rest of your LLM interaction and response generation logic here) ---
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
        # --- (End of your LLM interaction logic) ---

        # Now, convert the AI response to speech and send to Vapi (either webhook or sync)
        if VAPI_WEBHOOK_URL:
            # Webhook version
            payload = {
                "text": response_text,
                "voice": VAPI_VOICE_ID,
                "webhook_url": VAPI_WEBHOOK_URL,
                "webhook_data": {
                    "session_id": session_id,
                    "window_id": window_id,
                    "response": response_text  # Include response in webhook data
                }
            }
            headers = {
                'Authorization': f'Bearer {VAPI_PUBLIC_KEY}',
                'Content-Type': 'application/json'
            }
            logger.info(f"Sending final TTS request with webhook for session {session_id}")
            response = requests.post(
                'https://api.vapi.ai/v1/audio/tts',  # Use correct TTS endpoint
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"Final TTS webhook request failed: {response.status_code}, {response.text}")
                voice_responses[session_id] = {
                    "status": "error",
                    "error": f"TTS request failed: {response.status_code}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                logger.info(f"Final TTS webhook request sent successfully for session {session_id}")
                voice_responses[session_id] = {
                    "status": "processing_webhook",
                    "response": response_text,  # Store text response, even if webhook fails
                    "timestamp": datetime.utcnow().isoformat()
                }

        else:
          # Synchronous version (using your existing text_to_speech function)
            logger.info(f"Sending TTS request for window {window_id}, session {session_id}")
            tts_response = await text_to_speech(text=response_text)

            if "audio_url" in tts_response:
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
                    "response": response_text,
                    "timestamp": datetime.utcnow().isoformat()
                }

    except Exception as e:
        logger.error(f"Error in background task: {e}", exc_info=True)
        voice_responses[session_id] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
# Remove the duplicate text_to_speech route
# @router.post("/text-to-speech/") ...

# Remove duplicate copies of the file.

async def generate_final_speech_webhook(
    user_text: str,
    window_id: str,
    session_id: str,
    memory_system,
    llm_service,
    case_response_template
):
  """empty duplicate func"""
  pass

async def generate_final_speech_sync(
    user_text: str,
    window_id: str,
    session_id: str,
    memory_system,
    llm_service,
    case_response_template
):
      """empty duplicate func"""
      pass