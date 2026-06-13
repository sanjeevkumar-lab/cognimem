from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    memory_type: str  
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance_score: float = 1.0
    tags: List[str] = []
    related_ids: List[str] = []
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}