from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ComplaintInput(BaseModel):
    image_caption: str = ""          # optional – provided by BLIP
    voice_text: str = ""             # optional – provided by Whisper
    hospital_name: str = "General Hospital"
    city: str = "Metropolis"
    name: str = "Anonymous"
    ward: str = "General Ward"

class ComplaintResponse(BaseModel):
    id: int
    timestamp: datetime
    category: str
    severity: str
    department: str
    description: str
    hospital_name: str
    city: str
    status: str = "active"
    relevant_sop: Optional[str] = None

    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    status: str
    database: str
    chromadb: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    hospital_name: str
    message: str
