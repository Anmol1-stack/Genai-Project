from pydantic import BaseModel, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


class ComplaintInput(BaseModel):
    image_caption: str = ""
    voice_text:    str = ""
    hospital_name: str = "General Hospital"
    city:          str = "Metropolis"
    name:          str = "Anonymous"
    ward:          str = "General Ward"


class ComplaintResponse(BaseModel):
    id:                 int
    timestamp:          datetime
    category:           str
    severity:           str
    department:         str
    description:        str
    hospital_name:      str
    city:               str
    status:             str = "active"
    relevant_sop:       Optional[str] = None
    # Extended pipeline fields
    location:           Optional[str] = None
    needs_human_review: Optional[bool] = None
    sla_hours:          Optional[int] = None
    first_responder:    Optional[str] = None
    action_on_report:   Optional[str] = None
    routing:            Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def parse_routing_json(cls, values):
        # When reading from ORM, routing_json (string) → routing (dict)
        if hasattr(values, "__dict__"):
            obj = values.__dict__
        elif isinstance(values, dict):
            obj = values
        else:
            return values

        routing_json = obj.get("routing_json")
        if routing_json and isinstance(routing_json, str):
            try:
                if isinstance(values, dict):
                    values["routing"] = json.loads(routing_json)
                else:
                    values.routing = json.loads(routing_json)
            except Exception:
                pass

        needs = obj.get("needs_human_review")
        if needs is not None and isinstance(needs, int):
            if isinstance(values, dict):
                values["needs_human_review"] = bool(needs)
            else:
                values.needs_human_review = bool(needs)

        return values

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status:   str
    database: str
    chromadb: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    hospital_name: str
    message:       str
