"""
Persona management module for RAG system.
Handles agent persona configuration and knowledge base linking.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class Persona:
    """Represents an agent persona configuration."""

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        system_prompt: str,
        knowledge_base_ids: Optional[List[str]] = None,
        persona_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ):
        """
        Initialize persona.

        Args:
            name: Persona name
            role: Persona role/title
            description: Description of persona
            system_prompt: System prompt for LLM
            knowledge_base_ids: List of document IDs in knowledge base
            persona_id: Unique ID (auto-generated if None)
            settings: Additional settings
            created_at: Creation timestamp
        """
        self.id = persona_id or str(uuid.uuid4())
        self.name = name
        self.role = role
        self.description = description
        self.system_prompt = system_prompt
        self.knowledge_base_ids = knowledge_base_ids or []
        self.settings = settings or {
            "retrieval_k": 5,
            "use_mmr": True,
            "mmr_lambda": 0.7,
            "similarity_threshold": 0.5,
        }
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "knowledge_base_ids": self.knowledge_base_ids,
            "settings": self.settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        """Create persona from dictionary."""
        return cls(
            persona_id=data.get("id"),
            name=data["name"],
            role=data["role"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            knowledge_base_ids=data.get("knowledge_base_ids", []),
            settings=data.get("settings"),
            created_at=data.get("created_at"),
        )


class PersonaManager:
    """Manage persona configurations."""

    def __init__(self, storage_path: str = "data/personas.json"):
        """
        Initialize persona manager.

        Args:
            storage_path: Path to JSON file for storing personas
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.personas: Dict[str, Persona] = {}
        self._load_personas()

        logger.info(
            f"PersonaManager initialized ({len(self.personas)} personas loaded)"
        )

    def _load_personas(self) -> None:
        """Load personas from storage."""
        if not self.storage_path.exists():
            logger.info("No existing personas file found, starting fresh")
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            for persona_data in data.get("personas", []):
                persona = Persona.from_dict(persona_data)
                self.personas[persona.id] = persona

            logger.info(
                f"Loaded {len(self.personas)} personas from {self.storage_path}"
            )

        except Exception as e:
            logger.error(f"Error loading personas: {e}")
            self.personas = {}

    def _save_personas(self) -> None:
        """Save personas to storage."""
        try:
            data = {
                "personas": [p.to_dict() for p in self.personas.values()],
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.personas)} personas to {self.storage_path}")

        except Exception as e:
            logger.error(f"Error saving personas: {e}")
            raise

    def create_persona(
        self,
        name: str,
        role: str,
        description: str,
        system_prompt: str,
        knowledge_base_ids: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Persona:
        """
        Create a new persona.

        Args:
            name: Persona name
            role: Persona role
            description: Description
            system_prompt: System prompt
            knowledge_base_ids: Document IDs
            settings: Additional settings

        Returns:
            Created Persona object
        """
        persona = Persona(
            name=name,
            role=role,
            description=description,
            system_prompt=system_prompt,
            knowledge_base_ids=knowledge_base_ids,
            settings=settings,
        )

        self.personas[persona.id] = persona
        self._save_personas()

        logger.info(f"Created persona: {persona.name} (ID: {persona.id})")
        return persona

    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """
        Get persona by ID.

        Args:
            persona_id: Persona ID

        Returns:
            Persona object or None if not found
        """
        return self.personas.get(persona_id)

    def list_personas(self) -> List[Persona]:
        """
        Get all personas.

        Returns:
            List of all Persona objects
        """
        return list(self.personas.values())

    def update_persona(
        self,
        persona_id: str,
        name: Optional[str] = None,
        role: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        knowledge_base_ids: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[Persona]:
        """
        Update an existing persona.

        Args:
            persona_id: Persona ID
            name: New name (optional)
            role: New role (optional)
            description: New description (optional)
            system_prompt: New system prompt (optional)
            knowledge_base_ids: New document IDs (optional)
            settings: New settings (optional)

        Returns:
            Updated Persona object or None if not found
        """
        persona = self.personas.get(persona_id)
        if not persona:
            logger.warning(f"Persona not found: {persona_id}")
            return None

        # Update fields
        if name is not None:
            persona.name = name
        if role is not None:
            persona.role = role
        if description is not None:
            persona.description = description
        if system_prompt is not None:
            persona.system_prompt = system_prompt
        if knowledge_base_ids is not None:
            persona.knowledge_base_ids = knowledge_base_ids
        if settings is not None:
            persona.settings.update(settings)

        persona.updated_at = datetime.now().isoformat()
        self._save_personas()

        logger.info(f"Updated persona: {persona.name} (ID: {persona_id})")
        return persona

    def delete_persona(self, persona_id: str) -> bool:
        """
        Delete a persona.

        Args:
            persona_id: Persona ID

        Returns:
            True if deleted, False if not found
        """
        if persona_id in self.personas:
            persona = self.personas[persona_id]
            del self.personas[persona_id]
            self._save_personas()
            logger.info(f"Deleted persona: {persona.name} (ID: {persona_id})")
            return True

        logger.warning(f"Persona not found for deletion: {persona_id}")
        return False

    def add_document_to_persona(self, persona_id: str, document_id: str) -> bool:
        """
        Add a document to persona's knowledge base.

        Args:
            persona_id: Persona ID
            document_id: Document ID to add

        Returns:
            True if successful, False otherwise
        """
        persona = self.personas.get(persona_id)
        if not persona:
            logger.warning(f"Persona not found: {persona_id}")
            return False

        if document_id not in persona.knowledge_base_ids:
            persona.knowledge_base_ids.append(document_id)
            persona.updated_at = datetime.now().isoformat()
            self._save_personas()
            logger.info(f"Added document {document_id} to persona {persona.name}")
            return True

        return False

    def remove_document_from_persona(self, persona_id: str, document_id: str) -> bool:
        """
        Remove a document from persona's knowledge base.

        Args:
            persona_id: Persona ID
            document_id: Document ID to remove

        Returns:
            True if successful, False otherwise
        """
        persona = self.personas.get(persona_id)
        if not persona:
            logger.warning(f"Persona not found: {persona_id}")
            return False

        if document_id in persona.knowledge_base_ids:
            persona.knowledge_base_ids.remove(document_id)
            persona.updated_at = datetime.now().isoformat()
            self._save_personas()
            logger.info(f"Removed document {document_id} from persona {persona.name}")
            return True

        return False


# Singleton instance
_persona_manager_instance = None


def get_persona_manager(storage_path: str = "data/personas.json") -> PersonaManager:
    """
    Get or create singleton PersonaManager instance.

    Args:
        storage_path: Path to personas storage file

    Returns:
        PersonaManager instance
    """
    global _persona_manager_instance
    if _persona_manager_instance is None:
        _persona_manager_instance = PersonaManager(storage_path=storage_path)
    return _persona_manager_instance
