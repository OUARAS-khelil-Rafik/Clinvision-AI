# ClinVision AI — RSNA Pneumonia Detection

> Clinical Imaging Copilot — educational / hackathon prototype.
> Chest X-ray pneumonia classification (DenseNet121) with DICOM-aware
> preprocessing, Grad-CAM explainability, a doctor login portal, and
> **free-LLM** clinical summaries.

NOT A MEDICAL DEVICE. Predictions, Grad-CAM overlays and summaries are AI
model outputs for educational purposes only. All clinical decisions must be
made by qualified professionals. No real patient data should be stored in
this demo (history keeps request id, class and probability only).

## Table of contents

1. [Quickstart](#1-quickstart)
2. [Demo credentials](#2-demo-credentials)
3. [What is implemented](#3-what-is-implemented)
4. [How it works](#4-how-it-works)
5. [Repository structure](#5-repository-structure)
6. [Backend API reference](#6-backend-api-reference)
7. [Frontend tour](#7-frontend-tour)
8. [Configuration](#8-configuration)
9. [Free LLM setup](#9-free-llm-setup)
10. [Local run](#10-local-run)
11. [Docker deployment](#11-docker-deployment)
12. [Training notebook](#12-training-notebook)
13. [Troubleshooting](#13-troubleshooting)
14. [Limitations](#14-limitations)
15. [References](#15-references)

## 1. Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env

# Terminal 1 — API
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
API_URL=http://localhost:8000 streamlit run frontend/app.py
```

Open `http://localhost:8501` and sign in (see credentials below).
API docs: `http://localhost:8000/docs`. Health: `http://localhost:8000/health`.

Trained weights must exist at `models/clinvision_pneumonia.keras`
(see [Training notebook](#12-training-notebook)).

## 2. Demo credentials

| Role   | Username | Password    |
|--------|----------|-------------|
| Doctor | `doctor` | `doctor123` |

Used by the Streamlit login and `POST /auth/login`. Change via
`DOCTOR_USERNAME` / `DOCTOR_PASSWORD` in `.env`. The UI also offers an
offline demo mode (local credential check) when the API is unreachable;
inference itself always requires the API.

## 3. What is implemented

- RSNA Pneumonia Detection Challenge dataset handling (notebook)
- Patient-level train/validation/test splitting (no leakage)
- DICOM preprocessing with `pydicom` (modality rescale + VOI LUT,
  `MONOCHROME1` polarity, percentile normalisation, 320x320 resize)
- DenseNet121 classifier (ImageNet transfer learning, sigmoid output)
- Full evaluation: accuracy, ROC-AUC, PR-AUC, precision, recall,
  specificity, F1, confusion matrix, threshold sweep
- Grad-CAM explainability (final dense block), failure-isolated so it can
  never break the prediction itself
- FastAPI service: request ids, timing metadata, file-size limits, typed
  errors, startup model preload, SQLite demo history
- Doctor login (bearer tokens, 12 h TTL) + token-gated history endpoints
- Streamlit doctor portal: stepper flow, preview, probability gauge with
  risk band, Grad-CAM view, clinical summary, TXT/JSON report export,
  session + server history
- Free-LLM summary chain (Ollama local, Groq, Gemini, Hugging Face, custom
  endpoint) with deterministic template fallback — the app runs with no
  LLM configured at all

## 4. How it works

```text
DICOM / JPG / PNG (<= MAX_UPLOAD_MB)
        |
DICOM-aware preprocessing -> 320x320 gray -> RGB replicate (0-255 px)
        |
DenseNet121 -> pneumonia probability (threshold 0.5)
        |
Grad-CAM overlay (failure-isolated)
        |
Structured summary -> optional free-LLM rewrite
        |
FastAPI (JSON) -> Streamlit doctor portal
```

Clinical context (age/sex/symptoms) is display-only for the summary; it is
never fed to the vision model.

## 5. Repository structure

```text
clinvision-ai-final/
├── ClinVision_AI_RSNA.ipynb   # full ML workflow (single source of truth)
├── backend/
│   └── main.py                # FastAPI: inference, auth, history, free LLM
├── frontend/
│   └── app.py                 # Streamlit doctor portal (login + analysis UI)
├── data/
│   ├── raw/                   # Kaggle DICOMs + CSVs (not in git)
│   ├── cache_320/             # generated resized PNG cache
│   └── metadata/              # generated split CSVs
├── models/
│   └── clinvision_pneumonia.keras   # trained weights (not in git)
├── artifacts/                 # metrics, plots, Grad-CAMs, history.sqlite3
├── Dockerfile                 # CPU serving image (API + UI)
├── docker-compose.yml         # api + streamlit services
├── requirements.txt           # base deps (TF 2.18.1, CPU)
├── requirements-cuda.txt      # NVIDIA training variant
├── requirements-mac.txt       # Apple Silicon (adds tensorflow-metal)
├── .env.example               # all settings documented
├── SOURCE_REQUIREMENTS.md
└── README.md
```

Dataset, trained weights and PNG cache are intentionally not committed.

## 6. Backend API reference

Start: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

| Method | Endpoint            | Auth* | Description                                   |
|--------|---------------------|-------|-----------------------------------------------|
| GET    | `/`                 | no    | Service info + endpoint list                  |
| GET    | `/health`           | no    | Model/device/LLM status (`ok` / `degraded`)   |
| GET    | `/llm/status`       | no    | Which free-LLM backends are configured        |
| POST   | `/auth/login`       | no    | Doctor login (form) returns bearer token      |
| POST   | `/auth/login/json`  | no    | Doctor login (JSON) returns bearer token      |
| GET    | `/auth/verify`      | token | Check current token                           |
| POST   | `/predict`          | †     | Probability only (multipart `file`)           |
| POST   | `/explain`          | †     | Probability + Grad-CAM PNG (base64)           |
| POST   | `/summary`          | †     | Summary for a probability + context (form)    |
| POST   | `/analyze`          | †     | All-in-one: prediction + Grad-CAM + summary   |
| GET    | `/history?limit=50` | †     | Recent demo predictions                       |
| DELETE | `/history`          | †     | Clear demo history                            |

\* Token = `Authorization: Bearer <token>` from `/auth/login`.
† Required only when `REQUIRE_AUTH=true` (default `false`, so existing
scripts keep working; the Streamlit UI always enforces its own login).

Examples:

```bash
# Login (JSON)
curl -X POST localhost:8000/auth/login/json \
  -H 'Content-Type: application/json' \
  -d '{"username":"doctor","password":"doctor123"}'

# Full analysis (prediction + Grad-CAM + summary)
curl -X POST localhost:8000/analyze \
  -F "file=@sample.png" -F "age=58" -F "sex=Male" -F "symptoms=cough, fever"

# Probability + Grad-CAM only
curl -X POST localhost:8000/explain -F "file=@sample.dcm"

# Summary for a known probability
curl -X POST localhost:8000/summary -F "probability=0.82" -F "symptoms=cough"
```

Error contract: `400` empty/oversize upload or bad probability, `401` bad
credentials (or missing token when `REQUIRE_AUTH=true`), `413` file over
`MAX_UPLOAD_MB`, `422` undecodable image / inference failure, `500`
unexpected.

## 7. Frontend tour

Start: `streamlit run frontend/app.py` → `http://localhost:8501`

1. **Doctor Sign In** — username/password form in a centered card; shows an
   API status pill; falls back to offline demo mode if the API is down.
2. **Analyze tab** — 3-step flow with a stepper header (Upload → Analyze →
   Review): file upload with preview + size, context confirmation pulled
   from the sidebar, **Analyze image** with staged progress. Results show a
   risk-band pill, native metrics, a Plotly probability gauge,
   request/model metadata, Grad-CAM with PNG download, clinical summary
   with LLM-source badge, expandable structured details + limitations, and
   TXT/JSON report export.
3. **History tab** — this-session table plus cached server history with a
   refresh button and a confirm-gated clear button (`DELETE /history`).
4. **About tab** — pipeline recap and the Ollama one-liner.

Sidebar: clinical context (age checkbox so "unknown" is explicit, sex,
symptoms, context), session info (API URL, upload limit, active free-LLM
backends), status refresh and clear actions. Health/history calls are
cached (15 s / 30 s) to keep the UI responsive.

## 8. Configuration

Copy `.env.example` → `.env`:

| Variable               | Default                                  | Purpose                              |
|------------------------|------------------------------------------|--------------------------------------|
| `MODEL_PATH`           | `models/clinvision_pneumonia.keras`      | Weights file (repo-root relative)    |
| `MODEL_VERSION`        | `clinvision-v2`                          | Version string in responses          |
| `API_URL`              | `http://localhost:8000`                  | Backend URL for Streamlit            |
| `MAX_UPLOAD_MB`        | `20`                                     | Upload size limit                    |
| `LOG_LEVEL`            | `INFO`                                   | `DEBUG` / `INFO` / `WARNING` …       |
| `DOCTOR_USERNAME`      | `doctor`                                 | Portal + API login                   |
| `DOCTOR_PASSWORD`      | `doctor123`                              | Portal + API login                   |
| `REQUIRE_AUTH`         | `false`                                  | `true` forces tokens on inference    |
| `AUTH_TOKEN_TTL_HOURS` | `12`                                     | Bearer token lifetime                |
| `OLLAMA_BASE_URL`      | `http://localhost:11434`                 | Free local LLM                       |
| `OLLAMA_MODEL`         | `llama3.2`                               | Ollama model name                    |
| `OLLAMA_TIMEOUT_S`     | `60`                                     | Ollama request timeout               |
| `GROQ_API_KEY/MODEL`   | — / `llama-3.3-70b-versatile`            | Groq free tier                       |
| `GEMINI_API_KEY/MODEL` | — / `gemini-2.0-flash`                   | Gemini free tier                     |
| `HF_API_KEY/MODEL`     | — / `mistralai/Mistral-7B-Instruct-v0.3` | HF serverless free tier              |
| `LLM_BASE_URL/API_KEY/MODEL` | —                                | Legacy custom OpenAI-compatible endpoint |

Never commit API keys or patient data (`.env` is git-ignored).

## 9. Free LLM setup

The summary chain tries providers in order and always falls back to the
deterministic template — **no LLM is required to run the app**. Check active
backends at `GET /llm/status` or in the sidebar.

**Recommended — Ollama (local, fully free, no key):**

```bash
brew install ollama
ollama serve &           # one terminal
ollama pull llama3.2     # one-time download (~2 GB)
```

Keep `OLLAMA_BASE_URL=http://localhost:11434` and `OLLAMA_MODEL=llama3.2`
in `.env`. Alternatives: free `GROQ_API_KEY` (console.groq.com),
`GEMINI_API_KEY` (aistudio.google.com), `HF_API_KEY` (huggingface.co).

Safety design: temperature 0, "use only supplied fields" prompting, no
invented symptoms/diagnoses/treatments, and the output always states it is
an AI model output requiring professional review.

## 10. Local run

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000          # API
API_URL=http://localhost:8000 streamlit run frontend/app.py  # UI
```

Hardware notes:

- **NVIDIA CUDA (Linux/WSL2):** install `requirements-cuda.txt`, verify with
  `nvidia-smi` then `tf.config.list_physical_devices('GPU')`.
- **Apple Silicon (M1–M4):** `pip install -r requirements-mac.txt` (adds
  `tensorflow-metal`) in a native arm64 Python 3.11 venv.
- **CPU:** base `requirements.txt` works; reduce `BATCH_SIZE`/`IMAGE_SIZE`
  in the notebook if memory is tight.

## 11. Docker deployment

```bash
docker compose build
docker compose up
```

| Service   | URL                            |
|-----------|--------------------------------|
| Streamlit | `http://localhost:8501`        |
| FastAPI   | `http://localhost:8000/docs`   |

The image is CPU-based (`python:3.11-slim` + OpenGL libs for OpenCV).
`./models` is mounted read-only, `./artifacts` read-write. For Ollama on
the host, point the API at `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
For GPU training use a CUDA host or Colab, not this serving image.

## 12. Training notebook

All ML lives in `ClinVision_AI_RSNA.ipynb` (run cells in order):

1. Environment + device checks (Python, TF version, GPU/Metal/CPU).
2. Dataset discovery under `data/raw/` (`stage_2_train_images/`,
   `stage_2_train_labels.csv`, `stage_2_detailed_class_info.csv`).
3. Data audit (patients, positives/negatives, missing/duplicates, bad files).
4. Patient-level split (default 75/15/10; test untouched until final eval).
5. DICOM preprocessing → PNG cache in `data/cache_320/`.
6. TF pipeline (train-only augmentation: rotation/translation/zoom/contrast;
   no horizontal flip, to preserve laterality).
7. Stage A: train head, backbone frozen (val-accuracy + val-AUC checkpoints,
   ReduceLROnPlateau, EarlyStopping).
8. Stage B: fine-tune last 80 layers at low LR, BatchNorm frozen.
9. Evaluation: accuracy @0.5 and @validation-selected threshold, ROC-AUC,
   PR-AUC, precision, recall, specificity, F1, confusion matrix, curves.

Model profile: input 320×320×3, DenseNet121 (ImageNet) + GAP + BatchNorm +
Dropout 0.20 + Dense(1, sigmoid), AdamW (3e-4 head / 5e-6 fine-tune).
Reaching 90% validation accuracy is not guaranteed — always read recall,
specificity and AUC alongside accuracy on this imbalanced task.

## 13. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/health` shows `degraded` | `MODEL_PATH` wrong or `.keras` missing — train notebook or restore file |
| Streamlit "API unreachable" | API not running, wrong `API_URL` (Docker: `http://api:8000`), or port clash (`lsof -i :8000`) |
| Login fails | Check `DOCTOR_USERNAME`/`DOCTOR_PASSWORD`; expired token → log in again |
| `422` on upload | File corrupt/unsupported (DCM/JPG/PNG only) or under 32 px |
| `413` on upload | Over `MAX_UPLOAD_MB`; raise limit or compress |
| Template summary instead of LLM | No backend reachable — check `GET /llm/status`, `ollama serve`, model pulled |
| TF sees no GPU (NVIDIA) | Check `nvidia-smi` first, then the CUDA TF install |
| TF sees no GPU (Mac) | Native arm64 venv + `tensorflow-metal==1.2.0` |
| OOM in training | Lower `BATCH_SIZE` (e.g. 4) / `IMAGE_SIZE` (e.g. 256), prefer local SSD |

## 14. Limitations

Engineering prototype on the RSNA Pneumonia Detection Challenge data: no
clinical/prospective validation, no regulatory clearance, no guaranteed
reliability on real hospital populations. Grad-CAM is explanatory, not a
validated lesion localiser. Do not use as a standalone diagnostic system.

## 15. References

- RSNA Pneumonia Detection Challenge: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- TensorFlow pip install: https://www.tensorflow.org/install/pip
- Apple Metal: https://developer.apple.com/metal/
- Keras Grad-CAM example: https://keras.io/examples/vision/grad_cam/

