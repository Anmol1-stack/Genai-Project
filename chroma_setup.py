import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize Chroma Persistent Client
CHROMA_DB_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# We use a Sentence Transformer model exactly as required
model = SentenceTransformer('all-MiniLM-L6-v2')

SOPS = [
    "Protocol for handling broken equipment: report to maintenance within 2 hours, escalate to Head of Facilities if not resolved in 24 hours.",
    "Patient fall protocol: immediately assess vitals, notify attending physician, complete incident report in the adverse event system.",
    "Medication error procedure: notify pharmacy immediately, monitor patient for 4 hours, file a grievance report with the internal quality team.",
    "Fire emergency protocol (Code Red): activate nearest alarm, evacuate ambulatory patients first, close all fire doors.",
    "Aggressive behavior (Code Grey): contact hospital security immediately, do not approach alone, secure loose objects in the vicinity.",
    "Power outage procedure: emergency generators start automatically; manually check ventilators and life support systems within 2 minutes.",
    "Lost item protocol (patient belongings): log description in security database, check last known location, follow up with patient within 48 hours.",
    "Sanitation hazard (spill): isolate area with caution tape, contact environmental services, use biohazard kit for bodily fluids.",
    "Uninterruptible Power Supply (UPS) failure: notify IT network team immediately, switch critical servers to secondary power loop.",
    "Unauthorized access: challenge individual for ID badge, escort to security desk, log incidence in access control system.",
    "IT system downtime (EHR failure): switch to paper charting immediately, DO NOT reset workstations, await all-clear from IT command center.",
    "HVAC failure in operating rooms: halt non-emergency surgeries, seal sterile doors, notify facilities engineering immediately.",
    "Severe weather warning (Tornado/Hurricane): move patients away from windows, close all blinds, prepare emergency discharge protocols if ordered.",
    "Food hygiene standard violation: discard exposed items immediately, sanitize preparation areas with industrial bleach, notify safety inspector.",
    "Staff shortage in nursing unit: float nurses from less critical wards, contact nursing supervisor for agency dispatch, prioritize critical care patients."
]

def initialize_chromadb():
    print("Initializing ChromaDB collection...")
    # Get or create collection
    collection = client.get_or_create_collection(name="hospital_sops")
    
    # Check if already populated
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} SOPs. Skipping initialization.")
        return collection
        
    print("Embedding SOPs using sentence-transformers...")
    # Generate embeddings
    embeddings = model.encode(SOPS)
    
    # IDs for documents
    ids = [f"sop_{i}" for i in range(len(SOPS))]
    
    collection.add(
        documents=SOPS,
        embeddings=embeddings.tolist(),
        ids=ids
    )
    print("Successfully added SOPs to ChromaDB.")
    return collection

def get_sop_retriever():
    return client.get_collection(name="hospital_sops")

def query_relevant_sop(query_text: str, n_results: int = 1) -> str:
    collection = get_sop_retriever()
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    if results and results['documents'] and len(results['documents'][0]) > 0:
        return results['documents'][0][0]
    return "No relevant SOP found."

if __name__ == "__main__":
    initialize_chromadb()
