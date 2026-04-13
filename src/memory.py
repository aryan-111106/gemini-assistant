"""Memory management for conversation history and persistence."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    attachments: list[str] = field(default_factory=list)  # file paths


@dataclass 
class Conversation:
    """A conversation with metadata and messages."""
    id: str
    title: str
    created_at: str
    messages: list[Message] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, attachments: list[str] = None):
        """Add a message to the conversation."""
        self.messages.append(Message(
            role=role,
            content=content,
            attachments=attachments or []
        ))
    
    def get_context(self, max_messages: int = 20) -> list[dict]:
        """Get recent messages formatted for Gemini API."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": m.role, "parts": [m.content]} for m in recent]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "messages": [asdict(m) for m in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Create from dictionary."""
        messages = [Message(**m) for m in data.get("messages", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            created_at=data["created_at"],
            messages=messages
        )


class MemoryManager:
    """Manages conversation persistence and retrieval."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path.home() / ".gemini-assistant" / "conversations"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_conversation: Optional[Conversation] = None
    
    def new_conversation(self, title: str = None) -> Conversation:
        """Start a new conversation."""
        conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_conversation = Conversation(
            id=conv_id,
            title=title or f"Conversation {conv_id}",
            created_at=datetime.now().isoformat()
        )
        return self.current_conversation
    
    def save_conversation(self):
        """Save current conversation to disk."""
        if not self.current_conversation:
            return
        
        filepath = self.storage_dir / f"{self.current_conversation.id}.json"
        with open(filepath, "w") as f:
            json.dump(self.current_conversation.to_dict(), f, indent=2)
    
    def load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a conversation by ID."""
        filepath = self.storage_dir / f"{conv_id}.json"
        if not filepath.exists():
            return None
        
        with open(filepath) as f:
            data = json.load(f)
        
        self.current_conversation = Conversation.from_dict(data)
        return self.current_conversation
    
    def list_conversations(self) -> list[dict]:
        """List all saved conversations."""
        conversations = []
        for filepath in sorted(self.storage_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                conversations.append({
                    "id": data["id"],
                    "title": data["title"],
                    "created_at": data["created_at"],
                    "message_count": len(data.get("messages", []))
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return conversations
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        filepath = self.storage_dir / f"{conv_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def add_message(self, role: str, content: str, attachments: list[str] = None):
        """Add a message to current conversation and auto-save."""
        if self.current_conversation:
            self.current_conversation.add_message(role, content, attachments)
            self.save_conversation()
    
    def get_context(self, max_messages: int = 20) -> list[dict]:
        """Get conversation context for API calls."""
        if self.current_conversation:
            return self.current_conversation.get_context(max_messages)
        return []
