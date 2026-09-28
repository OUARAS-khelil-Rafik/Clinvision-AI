"""
ClinVision AI — FastAPI inference service.

ML workflow lives in ClinVision_AI_RSNA.ipynb; this module contains only
production-facing inference logic (preprocessing, model serving, Grad-CAM,
auth, history, and free-LLM clinical summaries).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pydicom
import requests
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel
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
ALLOWED_EXTENSIONS = {".dcm", ".jpg", ".jpeg", ".png"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("clinvision-api")

# --- Doctor login (demo credentials, overridable via .env) -------------------
DOCTOR_USERNAME = os.getenv("DOCTOR_USERNAME", "doctor")
DOCTOR_PASSWORD = os.getenv("DOCTOR_PASSWORD", "doctor123")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes"}
AUTH_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_HOURS", "12")) * 3600

# --- FREE LLM configuration (all optional, tried in order) -------------------
# 1) Ollama (100% free, local, no API key) — recommended
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_S", "60"))
# 2) Groq free tier (OpenAI-compatible, needs free key from console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
# 3) Google Gemini free tier (needs free key from aistudio.google.com)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
# 4) Hugging Face free serverless (needs free token from huggingface.co)
HF_API_KEY = os.getenv("HF_API_KEY", os.getenv("HUGGINGFACE_API_KEY", "")).strip()
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3").strip()
# 5) Generic OpenAI-compatible endpoint (legacy support: LM Studio, vLLM, etc.)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

# -----------------------------------------------------------------------------
# App lifespan — preload model + init DB once
# -----------------------------------------------------------------------------
MODEL = None
_TOKENS: dict[str, float] = {}  # token -> expiry epoch


def get_model() -> tf.keras.Model:
    """Load the Keras model exactly once and reuse it across requests."""
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        logger.info("Loading model: %s", MODEL_PATH)
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return MODEL


def init_history() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS predictions "
            "(timestamp REAL, request_id TEXT, model_version TEXT, "
            "predicted_class TEXT, probability REAL)"
        )
        connection.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_history()
    try:
        get_model()
        logger.info("Model preloaded OK (%s)", MODEL_VERSION)
    except Exception as exc:  # don't crash boot when model file is missing
        logger.warning("Model preload skipped: %s", exc)
    yield


app = FastAPI(
    title="ClinVision AI API",
    version=MODEL_VERSION,
    description="AI-assisted RSNA chest X-ray pneumonia classification with Grad-CAM.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)


# -----------------------------------------------------------------------------
# Auth — simple demo token auth for the doctor portal
# -----------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""


def _password_ok(candidate: str) -> bool:
    return hmac.compare_digest(candidate or "", DOCTOR_PASSWORD)


def _issue_token(username: str) -> dict:
    token = secrets.token_urlsafe(32)
    _TOKENS[token] = time.time() + AUTH_TOKEN_TTL_SECONDS
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": username,
        "expires_in": AUTH_TOKEN_TTL_SECONDS,
    }


def _token_valid(token: str) -> bool:
    expiry = _TOKENS.get(token or "")
    if not expiry:
        return False
    if expiry < time.time():
        _TOKENS.pop(token, None)
        return False
    return True


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """Return username when a valid bearer token is supplied, else None."""
    if creds and creds.credentials and _token_valid(creds.credentials):
        return DOCTOR_USERNAME
    return None


async def require_user_if_enabled(
    user: Optional[str] = Depends(get_current_user),
) -> Optional[str]:
    if REQUIRE_AUTH and not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. POST /auth/login first.",
        )
    return user


@app.post("/auth/login")
async def login(
    username: str = Form(""),
    password: str = Form(""),
) -> JSONResponse:
    """Doctor login. Accepts form fields OR a JSON body {username, password}."""
    # FastAPI Form defaults make JSON bodies fail validation, so also try JSON manually.
    # When called with JSON, Form(...) still yields "" — fall back to request body.
    if not username and not password:
        # This branch is reached via JSON only when using the raw endpoint below.
        pass
    if username == DOCTOR_USERNAME and _password_ok(password):
        return JSONResponse(_issue_token(username))
    raise HTTPException(status_code=401, detail="Invalid username or password.")


@app.post("/auth/login/json")
async def login_json(body: LoginRequest) -> JSONResponse:
    if (body.username or "").strip() == DOCTOR_USERNAME and _password_ok(body.password):
        return JSONResponse(_issue_token(DOCTOR_USERNAME))
    raise HTTPException(status_code=401, detail="Invalid username or password.")


@app.get("/auth/verify")
async def verify(user: Optional[str] = Depends(get_current_user)) -> JSONResponse:
    return JSONResponse({"authenticated": bool(user), "username": user})


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
    replicate grayscale into RGB, and return (model_batch, resized_gray).
    The model embeds DenseNet preprocessing and expects 0-255 pixels here.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Use DCM, JPG or PNG.")
    if suffix == ".dcm":
        try:
            gray = dicom_to_gray_bytes(raw)
        except Exception as exc:
            raise ValueError("The uploaded file is not a valid DICOM image.") from exc
    else:
        try:
            with Image.open(io.BytesIO(raw)) as pil:
                gray = np.asarray(pil.convert("L"), dtype=np.uint8)
        except Exception as exc:
            raise ValueError("The uploaded file is not a valid JPG/PNG image.") from exc

    if min(gray.shape[:2]) < 32:
        raise ValueError("Image is too small for reliable inference.")

    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.repeat(resized[..., None], 3, axis=-1).astype(np.float32)
    batch = np.expand_dims(rgb, axis=0)
    return batch, resized


# -----------------------------------------------------------------------------
# Grad-CAM
# -----------------------------------------------------------------------------
def _gradcam_forward(model: tf.keras.Model, input_batch: np.ndarray):
    """Manual forward pass returning (conv_maps, grads) for Grad-CAM.

    The nested DenseNet backbone cannot be re-wired into a single
    ``tf.keras.Model(model.inputs, …)`` (Keras 3 raises "not connected to
    inputs"), so the pipeline is replayed layer by layer under one
    GradientTape: preprocess → backbone (conv target + output) → head.
    Prefers the fine ``conv5_block16_concat`` maps, falls back to the
    backbone output if the inner layer is missing.
    """
    preprocess = model.get_layer("densenet_preprocess")
    backbone = model.get_layer("densenet121")
    try:
        target_layer = backbone.get_layer("conv5_block16_concat")
    except Exception:
        target_layer = None
    target_out = target_layer.output if target_layer is not None else backbone.output
    inner = tf.keras.Model(backbone.inputs, [target_out, backbone.output])
    pre_model = tf.keras.Model(model.inputs, preprocess.output)
    gap = model.get_layer("global_average_pool")
    bn = model.get_layer("head_batch_norm")
    drop = model.get_layer("head_dropout")
    head = model.get_layer("pneumonia_probability")

    with tf.GradientTape() as tape:
        x0 = pre_model(input_batch, training=False)
        conv_out, bb_out = inner(x0, training=False)
        tape.watch(conv_out)
        h = drop(bn(gap(bb_out, training=False), training=False), training=False)
        score = head(h, training=False)[:, 0]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients did not flow.")
    return conv_out, grads


def gradcam_overlay(model: tf.keras.Model, input_batch: np.ndarray,
                      original_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate (RGB overlay, RGB heatmap) for the model's predicted class.

    The raw heatmap is returned alongside the pre-blended overlay so clients
    can re-blend at any opacity (e.g. an examination slider) without a
    second inference pass.
    """
    conv_output, gradients = _gradcam_forward(model, input_batch)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_output, axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = cam.numpy()

    heat = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    heat = np.uint8(np.clip(heat, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.60, heat_bgr, 0.40, 0)
    return (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB),
            heat)


# -----------------------------------------------------------------------------
# Structured summary + FREE LLM (Ollama local first, then free cloud tiers)
# -----------------------------------------------------------------------------
def build_template_summary(
    probability: float,
    age: Optional[str],
    sex: Optional[str],
    symptoms: Optional[str],
    context: Optional[str],
) -> dict:
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
        "llm_enabled": False,
        "llm_provider": "template",
    }


LLM_ROLE = (
    "You are 'ClinVision Assistant', a senior radiology copilot assisting a "
    "licensed physician. You explain, structure and contextualize — you NEVER "
    "diagnose, prescribe, order procedures, or replace clinical judgment."
)

LLM_SECTIONS = ("Summary", "Explanation", "Indications and next steps", "Limitations")


def _llm_prompt(payload: dict) -> str:
    return (
        f"{LLM_ROLE} "
        "Use ONLY the supplied JSON. Never invent symptoms, findings, diagnosis, "
        "treatment, medication, urgency, or certainty. If a context field is empty, "
        "say so instead of assuming. Explicitly state this is an AI model output "
        "requiring professional review. "
        "Write the note in English for the physician, using EXACTLY these four "
        "headings, each on its own line starting with '### ':\n"
        "### 1. Summary\n"
        "2-3 sentences: what was analyzed, the model assessment and probability.\n"
        "### 2. Explanation\n"
        "Explain in plain clinical terms what this probability means, how the "
        "Grad-CAM overlay should and should NOT be read, and why the supplied "
        "clinical context cannot override the image model.\n"
        "### 3. Indications and next steps\n"
        "Suggest general, non-prescriptive steps the physician could consider "
        "(e.g. clinical correlation, repeat or additional imaging, specialist "
        "referral, laboratory workup), phrased strictly as possibilities, never "
        "as orders. Never name drugs, doses, or specific procedures.\n"
        "### 4. Limitations\n"
        "Briefly restate the given limitations.\n\n"
        f"JSON: {json.dumps(payload, ensure_ascii=False)}"
    )


def split_llm_sections(text: str) -> list[tuple[str, str]]:
    """Split an LLM note into (heading, body) on '### ' markers.

    Returns [] when no headings are found so callers fall back to raw text.
    """
    sections: list[tuple[str, str]] = []
    current_title: Optional[str] = None
    current_lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = stripped[4:].strip(" #-*") or "Section"
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, b) for t, b in sections if b]


def _call_ollama(prompt: str) -> Optional[str]:
    """Ollama local LLM — 100% free, no API key (http://localhost:11434)."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}},
            timeout=OLLAMA_TIMEOUT,
        )
        if r.status_code == 404:
            return None  # model not pulled yet
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
        return text or None
    except Exception as exc:
        logger.info("Ollama unavailable: %s", exc)
        return None


def _call_openai_compatible(base_url: str, api_key: str, model: str, prompt: str, timeout: int = 25) -> Optional[str]:
    try:
        r = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [
                      {"role": "system", "content": LLM_ROLE + " Follow the user's "
                       "section structure exactly. Use only supplied fields. "
                       "Never invent clinical facts. State this is an AI model output."},
                      {"role": "user", "content": prompt}],
                  "temperature": 0.0},
            timeout=timeout,
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip() or None
    except Exception as exc:
        logger.info("OpenAI-compatible LLM unavailable (%s): %s", base_url, exc)
        return None


def _call_gemini(prompt: str) -> Optional[str]:
    try:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}],
                                     "generationConfig": {"temperature": 0.0}},
                          timeout=25)
        r.raise_for_status()
        return (r.json()["candidates"][0]["content"]["parts"][0]["text"] or "").strip() or None
    except Exception as exc:
        logger.info("Gemini unavailable: %s", exc)
        return None


def _call_huggingface(prompt: str) -> Optional[str]:
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt, "parameters": {"temperature": 0.01, "max_new_tokens": 220}},
            timeout=40,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip() or None
        return None
    except Exception as exc:
        logger.info("HuggingFace unavailable: %s", exc)
        return None


def llm_status() -> dict:
    """Report which free LLM backends are configured (no secrets leaked)."""
    return {
        "ollama": {"configured": True, "model": OLLAMA_MODEL, "base_url": OLLAMA_BASE_URL},
        "groq": {"configured": bool(GROQ_API_KEY), "model": GROQ_MODEL},
        "gemini": {"configured": bool(GEMINI_API_KEY), "model": GEMINI_MODEL},
        "huggingface": {"configured": bool(HF_API_KEY), "model": HF_MODEL},
        "custom_openai_compatible": {"configured": bool(LLM_BASE_URL and LLM_API_KEY and LLM_MODEL)},
    }


def maybe_llm_summary(summary_payload: dict) -> dict:
    """
    Try FREE LLM providers in order (Ollama local -> Groq -> Gemini ->
    HuggingFace -> custom endpoint). Always fall back to the safe template.
    """
    prompt = _llm_prompt(summary_payload)
    generated: Optional[str] = None
    provider = "template"

    # 1) Ollama — free & local, no key required
    generated = _call_ollama(prompt)
    if generated:
        provider = f"ollama:{OLLAMA_MODEL}"

    # 2) Groq free tier
    if not generated and GROQ_API_KEY:
        generated = _call_openai_compatible(
            "https://api.groq.com/openai/v1", GROQ_API_KEY, GROQ_MODEL, prompt)
        if generated:
            provider = f"groq:{GROQ_MODEL}"

    # 3) Gemini free tier
    if not generated and GEMINI_API_KEY:
        generated = _call_gemini(prompt)
        if generated:
            provider = f"gemini:{GEMINI_MODEL}"

    # 4) Hugging Face free serverless
    if not generated and HF_API_KEY:
        generated = _call_huggingface(prompt)
        if generated:
            provider = f"huggingface:{HF_MODEL}"

    # 5) Legacy custom OpenAI-compatible endpoint
    if not generated and (LLM_BASE_URL and LLM_API_KEY and LLM_MODEL):
        generated = _call_openai_compatible(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, prompt, timeout=20)
        if generated:
            provider = f"custom:{LLM_MODEL}"

    out = dict(summary_payload)
    if generated:
        out["summary"] = generated
        out["llm_enabled"] = True
        out["llm_provider"] = provider
    else:
        out["llm_enabled"] = False
        out["llm_provider"] = "template"
    return out


# -----------------------------------------------------------------------------
# Demo history — no real patient data should be stored here.
# -----------------------------------------------------------------------------
def record_history(request_id: str, predicted_class: str, probability: float) -> None:
    init_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
            (time.time(), request_id, MODEL_VERSION, predicted_class, float(probability)),
        )
        connection.commit()


def read_history(limit: int = 50) -> list[dict]:
    init_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT timestamp, request_id, model_version, predicted_class, probability "
            "FROM predictions ORDER BY timestamp DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "service": "ClinVision AI API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/auth/login", "/predict", "/explain", "/summary",
                      "/analyze", "/history", "/llm/status"],
    }


@app.get("/health")
def health() -> dict:
    """Return service/model/device/LLM status for monitoring and demos."""
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False
    devices = [d.name for d in tf.config.list_physical_devices()]
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "tensorflow_devices": devices,
        "auth_required": REQUIRE_AUTH,
        "llm": llm_status(),
    }


@app.get("/llm/status")
def llm_status_endpoint() -> dict:
    return llm_status()


@app.get("/history")
def history(limit: int = 50, _user: Optional[str] = Depends(require_user_if_enabled)) -> dict:
    return {"count": None, "items": read_history(limit)}


@app.delete("/history")
def clear_history(_user: Optional[str] = Depends(require_user_if_enabled)) -> dict:
    init_history()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM predictions")
        connection.commit()
    return {"cleared": True}


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
            # Grad-CAM must never break the prediction itself.
            try:
                overlay, heatmap, mask = gradcam_overlay(model, input_batch, original)
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                ok_h, encoded_h = cv2.imencode(".png", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                ok_m, encoded_m = cv2.imencode(".png", mask)
                if ok and ok_h and ok_m:
                    result["gradcam_png_base64"] = base64.b64encode(encoded.tobytes()).decode("ascii")
                    result["gradcam_heatmap_base64"] = base64.b64encode(encoded_h.tobytes()).decode("ascii")
                    result["gradcam_mask_base64"] = base64.b64encode(encoded_m.tobytes()).decode("ascii")
                else:
                    result["gradcam_error"] = "Grad-CAM encoding failed."
            except Exception as exc:
                logger.warning("Grad-CAM failed for %s: %s", request_id, exc)
                result["gradcam_error"] = "Grad-CAM unavailable for this image."
        record_history(request_id, prediction, probability)
        return result
    except HTTPException:
        raise
    except ValueError as exc:  # bad file type / corrupt image -> 422 with clear message
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  _user: Optional[str] = Depends(require_user_if_enabled)) -> JSONResponse:
    return JSONResponse(await _predict_internal(file, include_gradcam=False))


@app.post("/explain")
async def explain(file: UploadFile = File(...),
                  _user: Optional[str] = Depends(require_user_if_enabled)) -> JSONResponse:
    return JSONResponse(await _predict_internal(file, include_gradcam=True))


@app.post("/summary")
async def summary(
    probability: float = Form(...),
    age: str = Form(""),
    sex: str = Form(""),
    symptoms: str = Form(""),
    context: str = Form(""),
    _user: Optional[str] = Depends(require_user_if_enabled),
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
    _user: Optional[str] = Depends(require_user_if_enabled),
) -> JSONResponse:
    result = await _predict_internal(file, include_gradcam=True)
    summary_payload = build_template_summary(result["probability"], age, sex, symptoms, context)
    result["clinical_summary"] = maybe_llm_summary(summary_payload)
    return JSONResponse(result)


@app.exception_handler(Exception)
async def fallback_exception_handler(_, exc: Exception):
    # Preserve HTTPExceptions raised intentionally.
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
