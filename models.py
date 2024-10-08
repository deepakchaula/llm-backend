# models.py
# type: ignore
from enum import Enum
from beanie import Document
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional, Dict
from datetime import datetime, timezone
import uuid

SUPPORTED_MODELS = ['gpt-4', 'gpt-3.5-turbo']

class QueryRoleType(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    function = "function"

class Prompt(BaseModel):
    role: QueryRoleType
    content: str
    timestamp: datetime = Field(datetime.now(timezone.utc))

class Conversation(Document):
    id: UUID4 = Field(default_factory=uuid.uuid4, alias="_id")
    name: str
    params: Optional[Dict] = {}
    tokens: int = 0
    messages: List[Prompt] = []
    created_at: datetime = Field(datetime.now(timezone.utc))

    class Settings:
        name = "conversations"
