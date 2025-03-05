# api/dependencies.py
from fastapi import Depends, HTTPException
from api.core.memory.memory import MemorySystem
from api.utils.llm_service import LLMService
from api.utils.prompt_templates import case_response_template

# Import the components variable (define it externally or import it)
# This should be defined here, not imported from main
from api.system_components import components

async def get_memory_system() -> MemorySystem:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment",
        )
    return components.memory_system

async def get_llm_service() -> LLMService:
    if not components._initialized:
        raise HTTPException(
            status_code=503,
            detail="System initializing, please try again in a moment",
        )
    return components.llm_service

def get_case_response_template():
    return case_response_template