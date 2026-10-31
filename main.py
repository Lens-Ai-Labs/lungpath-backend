from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from database import Base, DATA_DIR, engine, get_db
from models import Analysis
from schemas import AnalysisOut
from services.prediction import load_model, model_is_loaded, model_is_ready, predict_image

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Create fresh tables and add the detections field to an existing SQLite/PostgreSQL table.
Base.metadata.create_all(bind=engine)
columns = {column["name"] for column in inspect(engine).get_columns("analyses")}
if "detections_json" not in columns:
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE analyses ADD COLUMN detections_json TEXT"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Preload the YOLO model into RAM on startup to eliminate first-request cold starts.
    if model_is_ready():
        await run_in_threadpool(load_model)
    yield


app = FastAPI(
    title="LungPath AI Object Detection API",
    version="0.3.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
allow_origin_regex = os.getenv("CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allow_origin_regex if allow_origin_regex else None,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 15 * 1024 * 1024


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": model_is_ready(),
        "model_loaded": model_is_loaded(),
        "model_type": "YOLO object detection",
    }


@app.post("/api/analyze", response_model=AnalysisOut)
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")

    contents = await file.read(MAX_FILE_SIZE + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be smaller than 15 MB.")

    extension = ".png" if file.content_type == "image/png" else ".jpg"
    safe_name = f"{uuid4().hex}{extension}"
    image_path = UPLOAD_DIR / safe_name
    await run_in_threadpool(image_path.write_bytes, contents)

    record = Analysis(
        filename=Path(file.filename or safe_name).name[:255],
        image_path=str(image_path),
        status="processing",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    try:
        result = await run_in_threadpool(predict_image, str(image_path))
        if result is None:
            record.status = "model_not_ready"
            db.commit()
            db.refresh(record)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Image received and saved, but the trained YOLO model is not installed. "
                    "Place the trained best.pt file at model/best.pt or set MODEL_PATH."
                ),
            )

        record.prediction = result["prediction"]
        record.confidence = result["confidence"]
        record.detections_json = json.dumps(result["detections"])
        record.status = "completed"
        db.commit()
        db.refresh(record)
        return AnalysisOut.from_record(record)
    except HTTPException:
        raise
    except Exception as exc:
        record.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/api/analyses", response_model=list[AnalysisOut])
def list_analyses(db: Session = Depends(get_db)):
    records = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(50).all()
    return [AnalysisOut.from_record(record) for record in records]


# Keep the local one-URL experience when frontend directory exists locally.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
