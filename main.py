from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, case
import uvicorn
import random

from schemas import ComplaintInput, ComplaintResponse, HealthResponse
from database import get_db, ComplaintDatabaseModel
from chroma_setup import query_relevant_sop, initialize_chromadb

app = FastAPI(title="Hospital Complaint Management System - Skeleton")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# Initialize ChromaDB on startup
@app.on_event("startup")
def startup_event():
    initialize_chromadb()

def get_severity_order():
    return case(
        (ComplaintDatabaseModel.severity == "Critical", 4),
        (ComplaintDatabaseModel.severity == "High", 3),
        (ComplaintDatabaseModel.severity == "Medium", 2),
        (ComplaintDatabaseModel.severity == "Low", 1),
        else_=0
    )

def simulate_llm_processing(image_caption: str, voice_text: str):
    """
    Since the model is not connected in this phase, we simulate the 
    structured output that Qwen2.5 would generate.
    """
    severities = ["Critical", "High", "Medium", "Low"]
    categories = ["Maintenance", "Patient Care", "IT", "Security", "Sanitation"]
    departments = ["Facilities", "Nursing", "IT Support", "Security Guard", "Janitorial"]
    
    return {
        "category": random.choice(categories),
        "severity": random.choice(severities),
        "department": random.choice(departments),
        "description": f"[Simulated LLM Output] User complained via voice about: '{voice_text}'. Image caption implies '{image_caption}'."
    }

@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    # Basic check
    db_status = "ok" if db else "failed"
    return HealthResponse(
        status="ok",
        database=db_status,
        chromadb="initialized"
    )

@app.post("/process", response_model=ComplaintResponse)
def process_complaint(complaint: ComplaintInput, db: Session = Depends(get_db)):
    # Step 1: Simulate the absent AI model
    llm_output = simulate_llm_processing(complaint.image_caption, complaint.voice_text)
    
    # Step 2: Insert structured complaint into SQLite
    new_complaint = ComplaintDatabaseModel(
        category=llm_output["category"],
        severity=llm_output["severity"],
        department=llm_output["department"],
        description=llm_output["description"],
        hospital_name=complaint.hospital_name,
        city=complaint.city
    )
    
    db.add(new_complaint)
    db.commit()
    db.refresh(new_complaint)
    
    # Step 4: Semantic Search in ChromaDB for SOP
    relevant_sop = query_relevant_sop(llm_output["description"])
    
    # Step 5: Format response
    response = ComplaintResponse.from_orm(new_complaint)
    response.relevant_sop = relevant_sop
    
    return response

@app.get("/complaints", response_model=list[ComplaintResponse])
def get_complaints(db: Session = Depends(get_db)):
    """Retrieve past complaints sorted by severity (Critical first)"""
    complaints = db.query(ComplaintDatabaseModel).order_by(desc(get_severity_order())).all()
    # Dummy attach SOP if requested, or just return basic
    return complaints

@app.get("/urgent", response_model=list[ComplaintResponse])
def get_urgent_complaints(db: Session = Depends(get_db)):
    """Retrieve only Critical and High severity complaints"""
    complaints = db.query(ComplaintDatabaseModel).filter(
        ComplaintDatabaseModel.severity.in_(["Critical", "High"])
    ).order_by(desc(get_severity_order())).all()
    return complaints

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
