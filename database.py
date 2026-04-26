from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./complaints.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ComplaintDatabaseModel(Base):
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    category = Column(String, index=True)
    severity = Column(String, index=True)
    department = Column(String, index=True)
    description = Column(Text)
    hospital_name = Column(String, index=True)
    city = Column(String, index=True)
    status = Column(String, default="active", index=True)  # "active" or "resolved"

class HospitalUserModel(Base):
    __tablename__ = "hospital_users"

    id = Column(Integer, primary_key=True, index=True)
    hospital_name = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)  # Plain text for prototype

# Create tables
Base.metadata.create_all(bind=engine)

def init_default_hospitals(db):
    default_hospitals = [
        {"hospital_name": "General Hospital",   "username": "General Hospital",   "password": "password"},
        {"hospital_name": "City Care",           "username": "City Care",           "password": "password"},
        {"hospital_name": "Apollo Medical",      "username": "Apollo Medical",      "password": "password"},
        {"hospital_name": "Sunrise Clinic",      "username": "Sunrise Clinic",      "password": "password"},
        {"hospital_name": "Metro Health Centre", "username": "Metro Health Centre", "password": "password"},
        {"hospital_name": "Green Valley Hospital","username": "Green Valley Hospital","password": "password"},
    ]
    for h in default_hospitals:
        try:
            existing = db.query(HospitalUserModel).filter_by(username=h["username"]).first()
            if not existing:
                new_hospital = HospitalUserModel(**h)
                db.add(new_hospital)
                db.commit()
        except Exception:
            db.rollback()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
