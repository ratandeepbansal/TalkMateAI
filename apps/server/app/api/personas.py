"""
Persona management API endpoints.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..rag.persona_manager import get_persona_manager, Persona

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/personas", tags=["personas"])


class PersonaCreate(BaseModel):
    """Request model for creating a persona."""

    name: str
    role: str
    description: str
    system_prompt: str
    knowledge_base_ids: Optional[List[str]] = []
    settings: Optional[dict] = None


class PersonaUpdate(BaseModel):
    """Request model for updating a persona."""

    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    knowledge_base_ids: Optional[List[str]] = None
    settings: Optional[dict] = None


class PersonaResponse(BaseModel):
    """Response model for a persona."""

    id: str
    name: str
    role: str
    description: str
    system_prompt: str
    knowledge_base_ids: List[str]
    settings: dict
    created_at: str
    updated_at: str


class PersonaListResponse(BaseModel):
    """Response model for list of personas."""

    personas: List[PersonaResponse]
    total: int


class DocumentLinkRequest(BaseModel):
    """Request to link/unlink a document."""

    document_id: str


def persona_to_response(persona: Persona) -> PersonaResponse:
    """Convert Persona object to response model."""
    data = persona.to_dict()
    return PersonaResponse(**data)


@router.post("/", response_model=PersonaResponse)
async def create_persona(persona_data: PersonaCreate):
    """
    Create a new persona.
    """
    manager = get_persona_manager()

    persona = manager.create_persona(
        name=persona_data.name,
        role=persona_data.role,
        description=persona_data.description,
        system_prompt=persona_data.system_prompt,
        knowledge_base_ids=persona_data.knowledge_base_ids,
        settings=persona_data.settings,
    )

    return persona_to_response(persona)


@router.get("/", response_model=PersonaListResponse)
async def list_personas():
    """
    Get all personas.
    """
    manager = get_persona_manager()
    personas = manager.list_personas()

    return PersonaListResponse(
        personas=[persona_to_response(p) for p in personas],
        total=len(personas),
    )


@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(persona_id: str):
    """
    Get a specific persona by ID.
    """
    manager = get_persona_manager()
    persona = manager.get_persona(persona_id)

    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    return persona_to_response(persona)


@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(persona_id: str, persona_data: PersonaUpdate):
    """
    Update an existing persona.
    """
    manager = get_persona_manager()

    persona = manager.update_persona(
        persona_id=persona_id,
        name=persona_data.name,
        role=persona_data.role,
        description=persona_data.description,
        system_prompt=persona_data.system_prompt,
        knowledge_base_ids=persona_data.knowledge_base_ids,
        settings=persona_data.settings,
    )

    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    return persona_to_response(persona)


@router.delete("/{persona_id}")
async def delete_persona(persona_id: str):
    """
    Delete a persona.
    """
    manager = get_persona_manager()
    success = manager.delete_persona(persona_id)

    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")

    return {"message": "Persona deleted successfully", "persona_id": persona_id}


@router.post("/{persona_id}/documents", response_model=PersonaResponse)
async def add_document_to_persona(persona_id: str, request: DocumentLinkRequest):
    """
    Add a document to a persona's knowledge base.
    """
    manager = get_persona_manager()
    success = manager.add_document_to_persona(persona_id, request.document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")

    persona = manager.get_persona(persona_id)
    return persona_to_response(persona)


@router.delete("/{persona_id}/documents/{document_id}", response_model=PersonaResponse)
async def remove_document_from_persona(persona_id: str, document_id: str):
    """
    Remove a document from a persona's knowledge base.
    """
    manager = get_persona_manager()
    success = manager.remove_document_from_persona(persona_id, document_id)

    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")

    persona = manager.get_persona(persona_id)
    return persona_to_response(persona)
