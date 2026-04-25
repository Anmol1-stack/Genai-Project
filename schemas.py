from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ComplaintInput(BaseModel):
    image_caption: str
    voice_text: str
    hospital_name: str = "General Hospital"
    city: str = "Metropolis"

class ComplaintResponse(BaseModel):
    id: int
    timestamp: datetime
    category: str
    severity: str
    department: str
    description: str
    hospital_name: str
    city: str
    relevant_sop: Optional[str] = None

    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    status: str
    database: str
    chromadb: str
