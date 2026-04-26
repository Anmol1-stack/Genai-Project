from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, case
import uvicorn
import random
import os
import logging
import requests
import threading
import traceback
from dotenv import load_dotenv

# Load .env file (contains COLAB_LLM_URL)
load_dotenv(override=True)

from schemas import ComplaintInput, ComplaintResponse, HealthResponse, LoginRequest, LoginResponse
from database import get_db, ComplaintDatabaseModel, HospitalUserModel, init_default_hospitals
from typing import Optional
from chroma_setup import query_relevant_sop, initialize_chromadb

try:
    from hospital_multillm_rag import build_orchestrator, ComplaintInput as RagComplaintInput
except ImportError:
    # Fallback/placeholder if file is not present yet
    build_orchestrator = None
    RagComplaintInput = None

_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None and build_orchestrator is not None:
        _orchestrator = build_orchestrator()
    return _orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hospital Complaint Management System")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_details = traceback.format_exc()
    logger.error(f"Global crash: {error_details}")
    return JSONResponse(status_code=500, content={"detail": f"System Crash: {str(exc)}"})

app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Lazy imports for heavy AI models (loaded on first request) ───────────────
_whisper_loaded = False
_blip_loaded = False

# ─── Routes: Static Pages ─────────────────────────────────────────────────────

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

# ─── Startup ──────────────────────────────────────────────────────────────────

def _prewarm_models():
    """Pre-load BLIP and Whisper in the background so first requests don't timeout."""
    try:
        logger.info("Pre-warming Whisper model...")
        from whisper_handler import get_model as get_whisper
        get_whisper()
        logger.info("Whisper model ready.")
    except Exception as e:
        logger.warning(f"Whisper pre-warm failed: {e}")

    try:
        logger.info("Pre-warming BLIP model...")
        from blip_handler import get_blip
        get_blip()
        logger.info("BLIP model ready.")
    except Exception as e:
        logger.warning(f"BLIP pre-warm failed: {e}")

    try:
        logger.info("Pre-warming RAG Orchestrator...")
        if get_orchestrator():
            logger.info("Orchestrator ready.")
        else:
            logger.warning("Orchestrator could not be built (missing file).")
    except Exception as e:
        logger.warning(f"Orchestrator pre-warm failed: {e}")

@app.on_event("startup")
def startup_event():
    initialize_chromadb()
    db = next(get_db())
    init_default_hospitals(db)
    # Pre-warm AI models in background (avoids first-request timeout)
    threading.Thread(target=_prewarm_models, daemon=True).start()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_severity_order():
    return case(
        (ComplaintDatabaseModel.severity == "Critical", 4),
        (ComplaintDatabaseModel.severity == "High", 3),
        (ComplaintDatabaseModel.severity == "Medium", 2),
        (ComplaintDatabaseModel.severity == "Low", 1),
        else_=0
    )

def call_llm(image_caption: str, voice_text: str, hospital_name: str = "", ward: str = "", name: str = "") -> dict:
    orch = get_orchestrator()
    
    if orch is None:
        # Fallback simulation if hospital_multillm_rag.py is not available
        logger.info("Using fallback simulation because Orchestrator is missing.")
        severities  = ["Critical", "High", "Medium", "Low"]
        categories  = ["Maintenance", "Patient Care", "IT", "Security", "Sanitation"]
        departments = ["Facilities", "Nursing", "IT Support", "Security Guard", "Janitorial"]
        return {
            "category":    random.choice(categories),
            "severity":    random.choice(severities),
            "department":  random.choice(departments),
            "description": f"[Simulated — add hospital_multillm_rag.py] Voice: '{voice_text}'. Image: '{image_caption}'."
        }
        
    rag_input = RagComplaintInput(
        name=name or "Anonymous",
        complaint=voice_text,          # use voice text as the primary complaint
        hospital_name=hospital_name or "General Hospital",
        ward=ward or "General Ward",
        image_caption=image_caption,
        voice_text=voice_text,
    )
    
    try:
        result = orch.process(rag_input)
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Orchestrator failed: {error_details}")
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
    
    return {
        "category": result["category"],
        "severity": result["severity"],
        "department": result["department"],
        "description": result["complaint_description"],
    }

# ─── AI Endpoints ─────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file (WebM/WAV/MP4) recorded in the browser,
    runs OpenAI Whisper, and returns { "text": "..." }.
    """
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
    """
    Accepts an image file (JPEG/PNG), runs Salesforce BLIP,
    and returns { "caption": "..." }.
    """
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

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    colab_url = os.environ.get("COLAB_LLM_URL", "not configured")
    db_status = "ok" if db else "failed"
    return HealthResponse(
        status="ok",
        database=db_status,
        chromadb="initialized"
    )

# ─── Core: Process Complaint ──────────────────────────────────────────────────

@app.post("/process", response_model=ComplaintResponse)
def process_complaint(complaint: ComplaintInput, db: Session = Depends(get_db)):
    # Duplicate detection (skip when voice_text is empty)
    voice_for_dedup = (complaint.voice_text or "").strip().lower()
    if voice_for_dedup:
        existing = db.query(ComplaintDatabaseModel).filter(
            ComplaintDatabaseModel.hospital_name == complaint.hospital_name,
            ComplaintDatabaseModel.status == "active"
        ).all()

        for c in existing:
            if voice_for_dedup in c.description.lower():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"This complaint has already been submitted for {complaint.hospital_name} "
                        f"and is still active (ID #{c.id}). Please wait for it to be resolved."
                    )
                )

    # Use whichever text is available; require at least one field
    complaint_text = (complaint.voice_text or "").strip() or (complaint.image_caption or "").strip()
    if not complaint_text:
        raise HTTPException(
            status_code=422,
            detail="Please describe the complaint via voice recording, image, or type in the description field."
        )

    # Call local Multi-LLM RAG orchestrator
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
        status="active"
    )

    db.add(new_complaint)
    db.commit()
    db.refresh(new_complaint)

    relevant_sop = query_relevant_sop(llm_output["description"])

    response = ComplaintResponse.from_orm(new_complaint)
    response.relevant_sop = relevant_sop
    return response

# ─── Resolve Complaint ────────────────────────────────────────────────────────

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
    return complaint

# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/api/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    hospital = db.query(HospitalUserModel).filter(
        HospitalUserModel.username == request.username,
        HospitalUserModel.password == request.password
    ).first()
    if not hospital:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(hospital_name=hospital.hospital_name, message="Login successful")

@app.get("/api/hospitals")
def get_hospitals(db: Session = Depends(get_db)):
    hospitals = db.query(HospitalUserModel.hospital_name).all()
    return [h.hospital_name for h in hospitals]

# ─── Complaints Queries ───────────────────────────────────────────────────────

@app.get("/complaints", response_model=list[ComplaintResponse])
def get_complaints(
    hospital_name: Optional[str] = None,
    status: Optional[str] = "active",
    db: Session = Depends(get_db)
):
    query = db.query(ComplaintDatabaseModel)
    if status:
        query = query.filter(ComplaintDatabaseModel.status == status)
    if hospital_name:
        query = query.filter(ComplaintDatabaseModel.hospital_name == hospital_name)
    return query.order_by(desc(get_severity_order())).all()

@app.get("/urgent", response_model=list[ComplaintResponse])
def get_urgent_complaints(hospital_name: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(ComplaintDatabaseModel).filter(
        ComplaintDatabaseModel.severity.in_(["Critical", "High"]),
        ComplaintDatabaseModel.status == "active"
    )
    if hospital_name:
        query = query.filter(ComplaintDatabaseModel.hospital_name == hospital_name)
    return query.order_by(desc(get_severity_order())).all()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
