import os
from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

from backend.app.config import settings
from backend.app.database import Database, get_db_client
from backend.app.models.conversation import ConversationTurnInput, ConversationTurnOut
from backend.app.services.pipeline import EvaluationPipeline

# Define static directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Ensure static and templates folders exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A highly scalable evaluation engine matching turns to 300+ facets in microseconds.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup & Shutdown hooks
@app.on_event("startup")
def startup_db_client():
    Database.connect_db()

@app.on_event("shutdown")
def shutdown_db_client():
    Database.close_db()

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Server-Side Rendered Root endpoint
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "project_name": settings.PROJECT_NAME})

# API Routes
@app.post("/api/evaluate", response_model=Dict[str, Any])
def evaluate_turn(payload: ConversationTurnInput):
    try:
        result = EvaluationPipeline.run_evaluation(payload)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline Evaluation failed: {str(e)}")

@app.get("/api/facets")
def get_facets(
    search: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 1000
):
    db = get_db_client()
    if db is None:
        # DB connection failed fallback
        from backend.app.services.scorer import FacetScorerService
        facets = FacetScorerService.get_facets_from_db_or_fallback()
        # Filter in memory
        filtered = facets
        if search:
            search_l = search.lower()
            filtered = [f for f in filtered if search_l in f["name"].lower() or search_l in f["description"].lower()]
        if category:
            cat_l = category.lower()
            filtered = [f for f in filtered if cat_l in f["category"].lower()]
        return filtered[:limit]

    try:
        facets_coll = db["facets"]
        query = {}
        if search:
            query["$or"] = [
                {"name": {"$regex": search, "$options": "i"}},
                {"description": {"$regex": search, "$options": "i"}}
            ]
        if category:
            query["category"] = category
            
        facets = list(facets_coll.find(query, {"_id": 0, "raw_name": 1, "name": 1, "category": 1, "description": 1}).limit(limit))
        return facets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch facets: {str(e)}")

@app.get("/api/history")
def get_history(limit: int = 50):
    db = get_db_client()
    if db is None:
        return []
    try:
        conversations_coll = db["conversations"]
        records = list(conversations_coll.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
        # Format datetime objects for JSON serialization
        for r in records:
            if "created_at" in r:
                r["created_at"] = r["created_at"].isoformat()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/api/stats")
def get_stats():
    db = get_db_client()
    if db is None:
        return {"total_evaluations": 0, "avg_confidence": 0.0, "common_topics": []}
    try:
        conversations_coll = db["conversations"]
        total = conversations_coll.count_documents({})
        if total == 0:
            return {"total_evaluations": 0, "avg_confidence": 0.0, "common_topics": []}
            
        pipeline = [
            {"$group": {"_id": "$metadata.topic", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        topics = list(conversations_coll.aggregate(pipeline))
        
        pipeline_conf = [
            {"$group": {"_id": None, "avg_conf": {"$avg": "$metadata.confidence"}}}
        ]
        conf_res = list(conversations_coll.aggregate(pipeline_conf))
        avg_conf = conf_res[0]["avg_conf"] if len(conf_res) > 0 else 0.85
        
        return {
            "total_evaluations": total,
            "avg_confidence": round(avg_conf * 100, 1),
            "common_topics": [{"topic": t["_id"], "count": t["count"]} for t in topics]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@app.post("/api/clear")
def clear_data():
    db = get_db_client()
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected.")
    try:
        db["conversations"].delete_many({})
        return {"status": "success", "message": "Evaluations cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")

# Add startup message
@app.get("/api/health")
def health():
    is_placeholder = not settings.GROQ_API_KEY
    return {
        "status": "healthy",
        "groq_configured": bool(settings.GROQ_API_KEY and not is_placeholder),
        "mongodb_connected": Database.client is not None
    }
