"""YOLO object-detection inference for the trained Roboflow dataset model."""

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model" / "best.pt")))

_model: Any | None = None


def model_is_ready() -> bool:
    return MODEL_PATH.is_file()


def model_is_loaded() -> bool:
    return _model is not None


def load_model() -> Any | None:
    """Load the YOLO model once into memory and reuse across requests."""
    global _model
    if _model is not None:
        return _model
    if not model_is_ready():
        return None

    from ultralytics import YOLO

    _model = YOLO(str(MODEL_PATH))
    return _model


def predict_image(image_path: str) -> dict[str, Any] | None:
    """Return actual bounding-box detections, or None until best.pt is installed."""
    model = load_model()
    if model is None:
        return None

    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    image_size = int(os.getenv("IMAGE_SIZE", "640"))
    results = model.predict(
        source=image_path,
        imgsz=image_size,
        conf=confidence_threshold,
        device=os.getenv("MODEL_DEVICE", "cpu"),
        verbose=False,
    )
    result = results[0]
    height, width = result.orig_shape
    names = result.names
    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            detections.append({
                "class_name": str(label),
                "confidence": score,
                "x": max(0.0, min(1.0, x1 / width)),
                "y": max(0.0, min(1.0, y1 / height)),
                "width": max(0.0, min(1.0, (x2 - x1) / width)),
                "height": max(0.0, min(1.0, (y2 - y1) / height)),
            })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    best_confidence = detections[0]["confidence"] if detections else 0.0
    summary = f"{len(detections)} object(s) detected" if detections else "No objects detected"
    return {
        "prediction": summary,
        "confidence": best_confidence,
        "detections": detections,
    }
