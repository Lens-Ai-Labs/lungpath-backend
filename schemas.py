import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DetectionOut(BaseModel):
    class_name: str
    confidence: float
    # Normalized coordinates (0..1) relative to the original image.
    x: float
    y: float
    width: float
    height: float


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    prediction: str | None = None
    confidence: float | None = None
    status: str
    detections: list[DetectionOut] = []
    created_at: datetime

    @classmethod
    def from_record(cls, record: Any) -> "AnalysisOut":
        try:
            detections = json.loads(record.detections_json or "[]")
        except (TypeError, json.JSONDecodeError):
            detections = []
        return cls(
            id=record.id,
            filename=record.filename,
            prediction=record.prediction,
            confidence=record.confidence,
            status=record.status,
            detections=detections,
            created_at=record.created_at,
        )
