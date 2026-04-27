from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, desc
from sqlalchemy.orm import Session

load_dotenv(override=True)

from schemas import (
    ComplaintInput,
    ComplaintResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
)
from database import (
    ComplaintDatabaseModel,
    HospitalUserModel,
    get_db,
    init_default_hospitals,
)
from chroma_setup import initialize_chromadb, query_relevant_sop

try:
    from hospital_multillm_rag import build_orchestrator
    from hospital_multillm_rag import ComplaintInput as RagComplaintInput
except ImportError:
    build_orchestrator  = None
    RagComplaintInput   = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Hospital Complaint Management System")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global crash:\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"System Error: {str(exc)}"},
    )


app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Orchestrator singleton
# ---------------------------------------------------------------------------

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None and build_orchestrator is not None:
        _orchestrator = build_orchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/admin")
def serve_admin():
    return FileResponse("static/admin.html")

@app.get("/login")
def serve_login():
    return FileResponse("static/login.html")

@app.get("/hospital-dashboard")
def serve_hospital_dashboard():
    return FileResponse("static/hospital_dashboard.html")

# ---------------------------------------------------------------------------
# Startup: pre-warm models in background
# ---------------------------------------------------------------------------

def _prewarm_models():
    try:
        logger.info("Pre-warming Whisper model...")
        from whisper_handler import get_model as get_whisper
        get_whisper()
        logger.info("Whisper ready.")
    except Exception as e:
        logger.warning(f"Whisper pre-warm failed: {e}")

    try:
        logger.info("Pre-warming BLIP model...")
        from blip_handler import get_blip
        get_blip()
        logger.info("BLIP ready.")
    except Exception as e:
        logger.warning(f"BLIP pre-warm failed: {e}")

    try:
        logger.info("Pre-warming RAG Orchestrator...")
        orch = get_orchestrator()
        if orch:
            logger.info("Orchestrator ready.")
        else:
            logger.warning("Orchestrator could not be built.")
    except Exception as e:
        logger.warning(f"Orchestrator pre-warm failed: {e}")


@app.on_event("startup")
def startup_event():
    initialize_chromadb()
    db = next(get_db())
    init_default_hospitals(db)
    threading.Thread(target=_prewarm_models, daemon=True).start()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_severity_order():
    return case(
        (ComplaintDatabaseModel.severity == "Critical", 4),
        (ComplaintDatabaseModel.severity == "High",     3),
        (ComplaintDatabaseModel.severity == "Medium",   2),
        (ComplaintDatabaseModel.severity == "Low",      1),
        else_=0,
    )


def call_llm(
    image_caption: str,
    voice_text:    str,
    hospital_name: str = "",
    ward:          str = "",
    name:          str = "",
) -> dict:
    """
    Run the full Multi-LLM RAG pipeline and return a flat dict of all
    predicted + routing fields needed for storage and API response.
    """
    orch = get_orchestrator()

    if orch is None:
        logger.warning("Orchestrator unavailable — using fallback simulation.")
        import random
        return {
            "category":    random.choice(["Maintenance", "Patient Care", "Sanitation"]),
            "severity":    random.choice(["High", "Medium", "Low"]),
            "department":  random.choice(["Facilities", "Nursing", "Housekeeping"]),
            "description": f"[Simulated] Voice: '{voice_text}'. Image: '{image_caption}'.",
            "routing":     {},
            "needs_human_review": False,
            "location":    f"{hospital_name}, {ward}",
            "sla_hours":   None,
            "first_responder": None,
            "action_on_report": None,
        }

    rag_input = RagComplaintInput(
        name=name or "Anonymous",
        complaint=voice_text,
        hospital_name=hospital_name or "General Hospital",
        ward=ward or "General Ward",
        image_caption=image_caption,
        voice_text=voice_text,
    )

    try:
        result = orch.process(rag_input)
    except Exception as e:
        logger.error(f"Orchestrator failed:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    routing = result.get("routing", {}) or {}

    return {
        "category":           result["category"],
        # Capitalize for SQLAlchemy severity filters ("High", "Critical", etc.)
        "severity":           result["severity"].capitalize(),
        "department":         result["department"],
        "description":        result["complaint_description"],
        "routing":            routing,
        "needs_human_review": bool(result.get("needs_human_review", False)),
        "location":           result.get("location", f"{hospital_name}, {ward}"),
        "sla_hours":          routing.get("sla_hours"),
        "first_responder":    routing.get("first_responder"),
        "action_on_report":   routing.get("action_on_report"),
    }

# ---------------------------------------------------------------------------
# AI endpoints
# ---------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Accepts audio file → OpenAI Whisper → returns { "text": "..." }"""
    try:
        from whisper_handler import transcribe_audio as run_whisper
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Whisper not installed: {e}")
    try:
        audio_bytes = await file.read()
        text = run_whisper(audio_bytes, filename=file.filename or "audio.webm")
        return {"text": text}
    except Exception as e:
        logger.error(f"/transcribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    """Accepts image file → Salesforce BLIP → returns { "caption": "..." }"""
    try:
        from blip_handler import generate_caption as run_blip
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"BLIP not installed: {e}")
    try:
        image_bytes = await file.read()
        caption = run_blip(image_bytes)
        return {"caption": caption}
    except Exception as e:
        logger.error(f"/caption error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    return HealthResponse(
        status="ok",
        database="ok" if db else "failed",
        chromadb="initialized",
    )

# ---------------------------------------------------------------------------
# Core: process complaint
# ---------------------------------------------------------------------------

@app.post("/process", response_model=ComplaintResponse)
def process_complaint(complaint: ComplaintInput, db: Session = Depends(get_db)):
    # Duplicate detection
    voice_for_dedup = (complaint.voice_text or "").strip().lower()
    if voice_for_dedup:
        existing = db.query(ComplaintDatabaseModel).filter(
            ComplaintDatabaseModel.hospital_name == complaint.hospital_name,
            ComplaintDatabaseModel.status == "active",
        ).all()
        for c in existing:
            if voice_for_dedup in (c.description or "").lower():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"This complaint has already been submitted for {complaint.hospital_name} "
                        f"and is still active (ID #{c.id}). Please wait for it to be resolved."
                    ),
                )

    complaint_text = (complaint.voice_text or "").strip() or (complaint.image_caption or "").strip()
    if not complaint_text:
        raise HTTPException(
            status_code=422,
            detail="Please describe the complaint via voice recording, image, or type in the description field.",
        )

    llm_output = call_llm(
        image_caption=complaint.image_caption or "",
        voice_text=complaint_text,
        hospital_name=complaint.hospital_name,
        ward=complaint.ward or "General Ward",
        name=complaint.name or "Anonymous",
    )

    new_complaint = ComplaintDatabaseModel(
        category=llm_output["category"],
        severity=llm_output["severity"],
        department=llm_output["department"],
        description=llm_output["description"],
        hospital_name=complaint.hospital_name,
        city=complaint.city,
        status="active",
        location=llm_output["location"],
        needs_human_review=int(llm_output["needs_human_review"]),
        sla_hours=llm_output["sla_hours"],
        first_responder=llm_output["first_responder"],
        action_on_report=llm_output["action_on_report"],
        routing_json=json.dumps(llm_output["routing"], ensure_ascii=False),
    )

    db.add(new_complaint)
    db.commit()
    db.refresh(new_complaint)

    relevant_sop = query_relevant_sop(llm_output["description"])

    return ComplaintResponse(
        id=new_complaint.id,
        timestamp=new_complaint.timestamp,
        category=new_complaint.category,
        severity=new_complaint.severity,
        department=new_complaint.department,
        description=new_complaint.description,
        hospital_name=new_complaint.hospital_name,
        city=new_complaint.city,
        status=new_complaint.status,
        relevant_sop=relevant_sop,
        location=new_complaint.location,
        needs_human_review=bool(new_complaint.needs_human_review),
        sla_hours=new_complaint.sla_hours,
        first_responder=new_complaint.first_responder,
        action_on_report=new_complaint.action_on_report,
        routing=llm_output["routing"],
    )

# ---------------------------------------------------------------------------
# Resolve complaint
# ---------------------------------------------------------------------------

@app.patch("/complaints/{complaint_id}/resolve", response_model=ComplaintResponse)
def resolve_complaint(complaint_id: int, db: Session = Depends(get_db)):
    complaint = db.query(ComplaintDatabaseModel).filter(
        ComplaintDatabaseModel.id == complaint_id
    ).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    complaint.status = "resolved"
    db.commit()
    db.refresh(complaint)
    return ComplaintResponse.model_validate(complaint)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/api/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    hospital = db.query(HospitalUserModel).filter(
        HospitalUserModel.username == request.username,
        HospitalUserModel.password == request.password,
    ).first()
    if not hospital:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(hospital_name=hospital.hospital_name, message="Login successful")


@app.get("/api/hospitals")
def get_hospitals(db: Session = Depends(get_db)):
    hospitals = db.query(HospitalUserModel.hospital_name).all()
    return [h.hospital_name for h in hospitals]

# ---------------------------------------------------------------------------
# Complaints queries
# ---------------------------------------------------------------------------

@app.get("/complaints", response_model=list[ComplaintResponse])
def get_complaints(
    hospital_name: Optional[str] = None,
    status:        Optional[str] = "active",
    db:            Session = Depends(get_db),
):
    query = db.query(ComplaintDatabaseModel)
    if status:
        query = query.filter(ComplaintDatabaseModel.status == status)
    if hospital_name:
        query = query.filter(ComplaintDatabaseModel.hospital_name == hospital_name)
    rows = query.order_by(desc(get_severity_order())).all()
    return [ComplaintResponse.model_validate(r) for r in rows]


@app.get("/urgent", response_model=list[ComplaintResponse])
def get_urgent_complaints(
    hospital_name: Optional[str] = None,
    db:            Session = Depends(get_db),
):
    query = db.query(ComplaintDatabaseModel).filter(
        ComplaintDatabaseModel.severity.in_(["Critical", "High"]),
        ComplaintDatabaseModel.status == "active",
    )
    if hospital_name:
        query = query.filter(ComplaintDatabaseModel.hospital_name == hospital_name)
    rows = query.order_by(desc(get_severity_order())).all()
    return [ComplaintResponse.model_validate(r) for r in rows]


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
