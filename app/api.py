from pathlib import Path
import sys
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import (
    load_trained_model,
    predict_image,
)


MODEL_NAME = "convnext_tiny"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_convnext_tiny.pth"
DEFAULT_LUNG_OPACITY_THRESHOLD = 0.29

ALLOWED_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg"}


app = FastAPI(
    title="MedTriage-CXR API",
    description=(
        "FastAPI inference service for AI-assisted chest X-ray triage. "
        "This API is for educational and portfolio purposes only."
    ),
    version="1.0.0",
)

model = None
device = None


@app.on_event("startup")
def load_model_on_startup():
    """
    Load model once when the API starts.

    This avoids reloading the model for every request.
    """
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Train the model first or place best_convnext_tiny.pth in models/."
        )

    model = load_trained_model(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    print(f"Loaded {MODEL_NAME} from {CHECKPOINT_PATH}")
    print(f"Using device: {device}")


@app.get("/")
def root():
    return {
        "message": "MedTriage-CXR API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "device": str(device),
        "default_lung_opacity_threshold": DEFAULT_LUNG_OPACITY_THRESHOLD,
        "supported_file_types": sorted(list(ALLOWED_EXTENSIONS)),
        "disclaimer": (
            "This API is for educational and portfolio purposes only. "
            "It is not intended for clinical diagnosis."
        ),
    }


def validate_uploaded_file(uploaded_file: UploadFile) -> str:
    """
    Validate uploaded file extension.
    """
    suffix = Path(uploaded_file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {suffix}. "
                f"Allowed types: {sorted(list(ALLOWED_EXTENSIONS))}"
            ),
        )

    return suffix


def save_upload_to_temp_file(uploaded_file: UploadFile, suffix: str) -> Path:
    """
    Save uploaded file to a temporary file.

    On Windows, delete=False avoids file-locking issues when other libraries
    need to read the file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        contents = uploaded_file.file.read()
        temp_file.write(contents)
        return Path(temp_file.name)


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    mode: str = Query(
        default="balanced",
        description="Prediction mode: 'balanced' or 'high_sensitivity'",
    ),
    lung_opacity_threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional Lung Opacity threshold. "
            "If not provided, high_sensitivity mode uses 0.29."
        ),
    ),
):
    """
    Run chest X-ray triage prediction.

    Modes:
        balanced:
            Uses standard argmax prediction.

        high_sensitivity:
            Uses Lung Opacity thresholding to increase high-priority recall.
    """
    if model is None or device is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )

    if mode not in {"balanced", "high_sensitivity"}:
        raise HTTPException(
            status_code=400,
            detail="mode must be either 'balanced' or 'high_sensitivity'.",
        )

    suffix = validate_uploaded_file(file)

    if mode == "balanced":
        threshold_to_use = None
    else:
        threshold_to_use = (
            DEFAULT_LUNG_OPACITY_THRESHOLD
            if lung_opacity_threshold is None
            else lung_opacity_threshold
        )

    temp_image_path = save_upload_to_temp_file(
        uploaded_file=file,
        suffix=suffix,
    )

    try:
        result = predict_image(
            image_path=temp_image_path,
            model=model,
            device=device,
            lung_opacity_threshold=threshold_to_use,
        )

        result["api_mode"] = mode
        result["original_filename"] = file.filename
        result["disclaimer"] = (
            "This prediction is for educational and portfolio purposes only. "
            "It is not intended for clinical diagnosis."
        )

        return JSONResponse(content=result)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )

    finally:
        try:
            temp_image_path.unlink(missing_ok=True)
        except Exception:
            pass