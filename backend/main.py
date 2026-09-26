"""
ClinVision AI — FastAPI inference service.

This file is intentionally self-contained. The machine-learning workflow lives in
ClinVision_AI_RSNA.ipynb; this module contains only production-facing inference
logic needed by FastAPI so the repository has no extra Python modules.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pydicom
import requests
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "clinvision_pneumonia.keras"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
MODEL_VERSION = os.getenv("MODEL_VERSION", "clinvision-v2")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "320"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
DB_PATH = ROOT / "artifacts" / "history.sqlite3"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("clinvision-api")

# -----------------------------------------------------------------------------
# FastAPI application object
# -----------------------------------------------------------------------------
app = FastAPI(
    title="ClinVision AI API",
    version=MODEL_VERSION,
    description="AI-assisted RSNA chest X-ray pneumonia classification with Grad-CAM.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL = None


def get_model() -> tf.keras.Model:
    """Load the Keras model exactly once and reuse it across requests."""
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        logger.info("Loading model: %s", MODEL_PATH)
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return MODEL


# -----------------------------------------------------------------------------
# DICOM/image preprocessing
# -----------------------------------------------------------------------------
def _normalise_uint8(array: np.ndarray) -> np.ndarray:
    """Robustly normalize an intensity array to 8-bit without clipping everything."""
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(array, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def dicom_to_gray_bytes(raw: bytes) -> np.ndarray:
    """Decode DICOM bytes into a normalized grayscale array."""
    ds = pydicom.dcmread(io.BytesIO(raw), force=False)
    arr = ds.pixel_array.astype(np.float32)
    try:
        arr = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    return _normalise_uint8(arr)


def bytes_to_model_input(raw: bytes, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode either DICOM or a standard image, resize to the training contract,
    replicate grayscale into RGB, and apply DenseNet preprocessing.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".dcm":
        gray = dicom_to_gray_bytes(raw)
    else:
        try:
            pil = Image.open(io.BytesIO(raw)).convert("L")
            gray = np.asarray(pil, dtype=np.uint8)
        except Exception as exc:
            raise ValueError("The uploaded file is not a valid DICOM/JPG/PNG image.") from exc

    if min(gray.shape[:2]) < 32:
        raise ValueError("Image is too small for reliable inference.")

    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.repeat(resized[..., None], 3, axis=-1).astype(np.float32)
    batch = np.expand_dims(rgb, axis=0)
    # Le modèle embarque DenseNet preprocess_input et reçoit ici des pixels 0–255.
    return batch, resized


# -----------------------------------------------------------------------------
# Grad-CAM
# -----------------------------------------------------------------------------
def _find_target_layer(model: tf.keras.Model):
    """Prefer DenseNet's final convolutional tensor; otherwise find last 4-D layer."""
    try:
        backbone = model.get_layer("densenet121")
        return backbone.get_layer("conv5_block16_concat")
    except Exception:
        for layer in reversed(model.layers):
            try:
                output_shape = tuple(layer.output.shape)
                if len(output_shape) == 4:
                    return layer
            except Exception:
                continue
    raise RuntimeError("No suitable convolutional layer found for Grad-CAM.")


def gradcam_overlay(model: tf.keras.Model, input_batch: np.ndarray, original_gray: np.ndarray) -> np.ndarray:
    """Generate a normalized RGB Grad-CAM overlay for the model's predicted class."""
    target_layer = _find_target_layer(model)
    grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(input_batch, training=False)
        score = prediction[:, 0]
    gradients = tape.gradient(score, conv_output)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_output, axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = cam.numpy()

    heat = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    heat = np.uint8(np.clip(heat, 0, 1) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.60, heat, 0.40, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# Structured summary + optional LLM
# -----------------------------------------------------------------------------
def build_template_summary(probability: float, age: Optional[str], sex: Optional[str], symptoms: Optional[str], context: Optional[str]) -> dict:
    """Create a deterministic summary without inventing clinical facts."""
    prediction = "Pneumonia" if probability >= 0.5 else "No Pneumonia"
    return {
        "assessment": prediction,
        "model_probability": round(float(probability), 4),
        "clinical_context": {
            "age": age or None,
            "sex": sex or None,
            "symptoms": symptoms or None,
            "context": context or None,
        },
        "summary": (
            f"The model estimated a pneumonia probability of {probability:.1%}. "
            "This is an AI-assisted model output and must not be interpreted as a diagnosis. "
            "The supplied clinical context is shown only as context and was not used by the vision model."
        ),
        "limitations": [
            "Model output is probabilistic and requires professional review.",
            "Grad-CAM highlights influential image regions; it is not a validated lesion localization tool.",
            "Clinical context is not allowed to override the image model prediction.",
        ],
    }


def maybe_llm_summary(summary_payload: dict) -> dict:
    """Use an OpenAI-compatible endpoint only when explicitly configured; otherwise use the safe template."""
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "").strip()
    if not (base_url and api_key and model_name):
        return summary_payload

    system_prompt = (
        "You are generating a strictly controlled clinical-support summary. "
        "Use only the supplied fields. Never invent symptoms, findings, diagnosis, treatment, "
        "medication, urgency, or certainty. Explicitly state that this is an AI model output."
    )
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(summary_payload, ensure_ascii=False)},
        ],
        "temperature": 0.0,
    }
    try:
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=20,
        )
        response.raise_for_status()
        generated = response.json()["choices"][0]["message"]["content"]
        summary_payload = dict(summary_payload)
        summary_payload["summary"] = generated
        summary_payload["llm_enabled"] = True
        return summary_payload
    except Exception as exc:
        logger.warning("LLM summary unavailable; using deterministic fallback: %s", exc)
        summary_payload = dict(summary_payload)
        summary_payload["llm_enabled"] = False
        return summary_payload


# -----------------------------------------------------------------------------
# Demo history — no real patient data should be stored here.
# -----------------------------------------------------------------------------
def init_history() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS predictions (timestamp REAL, request_id TEXT, model_version TEXT, predicted_class TEXT, probability REAL)"
        )
        connection.commit()


def record_history(request_id: str, predicted_class: str, probability: float) -> None:
    init_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
            (time.time(), request_id, MODEL_VERSION, predicted_class, float(probability)),
        )
        connection.commit()


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Return service/model/device status for monitoring and demos."""
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False
    devices = [device.name for device in tf.config.list_physical_devices()]
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "tensorflow_devices": devices,
    }


async def _read_upload(file: UploadFile) -> bytes:
    """Read and bound the upload to avoid accidental huge request bodies."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    return raw


async def _predict_internal(file: UploadFile, include_gradcam: bool = False) -> dict:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    raw = await _read_upload(file)
    try:
        model = get_model()
        input_batch, original = bytes_to_model_input(raw, file.filename or "image")
        probability = float(model.predict(input_batch, verbose=0)[0][0])
        prediction = "Pneumonia" if probability >= 0.5 else "No Pneumonia"
        result = {
            "request_id": request_id,
            "prediction": prediction,
            "probability": round(probability, 6),
            "model_version": MODEL_VERSION,
            "threshold": 0.5,
            "inference_ms": round((time.perf_counter() - start) * 1000, 2),
            "explainability": "gradcam_available" if include_gradcam else "gradcam_available_on_request",
        }
        if include_gradcam:
            overlay = gradcam_overlay(model, input_batch, original)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if ok:
                import base64
                result["gradcam_png_base64"] = base64.b64encode(encoded.tobytes()).decode("ascii")
        record_history(request_id, prediction, probability)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    return JSONResponse(await _predict_internal(file, include_gradcam=False))


@app.post("/explain")
async def explain(file: UploadFile = File(...)) -> JSONResponse:
    return JSONResponse(await _predict_internal(file, include_gradcam=True))


@app.post("/summary")
async def summary(
    probability: float = Form(...),
    age: str = Form(""),
    sex: str = Form(""),
    symptoms: str = Form(""),
    context: str = Form(""),
) -> JSONResponse:
    if not 0.0 <= probability <= 1.0:
        raise HTTPException(status_code=400, detail="probability must be between 0 and 1")
    payload = build_template_summary(probability, age, sex, symptoms, context)
    return JSONResponse(maybe_llm_summary(payload))


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    age: str = Form(""),
    sex: str = Form(""),
    symptoms: str = Form(""),
    context: str = Form(""),
) -> JSONResponse:
    result = await _predict_internal(file, include_gradcam=True)
    summary_payload = build_template_summary(result["probability"], age, sex, symptoms, context)
    result["clinical_summary"] = maybe_llm_summary(summary_payload)
    return JSONResponse(result)


@app.exception_handler(Exception)
async def fallback_exception_handler(_, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
