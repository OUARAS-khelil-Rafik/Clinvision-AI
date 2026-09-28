# ClinVision AI — RSNA Pneumonia Detection

> Clinical Imaging Copilot — educational / hackathon prototype.
> Chest X-ray pneumonia classification (DenseNet121) with DICOM-aware
> preprocessing, Grad-CAM explainability, a doctor login portal, a 5-sample
> DICOM demo gallery, a patient archive (SQLite), and **free-LLM** clinical
> notes exported as colored PDF reports.

NOT A MEDICAL DEVICE. Predictions, Grad-CAM overlays and summaries are AI
model outputs for educational purposes only. All clinical decisions must be
made by qualified professionals. No real patient data should be stored in
this demo (the archive keeps request id, class and probability plus whatever
demo context was entered).

## Table of contents

1. [Quickstart](#1-quickstart)
2. [Demo credentials](#2-demo-credentials)
3. [Demo samples](#3-demo-samples)
4. [What is implemented](#4-what-is-implemented)
5. [How it works](#5-how-it-works)
6. [Repository structure](#6-repository-structure)
7. [Backend API reference](#7-backend-api-reference)
8. [Frontend tour](#8-frontend-tour)
9. [Configuration](#9-configuration)
10. [Free LLM setup](#10-free-llm-setup)
11. [Local run](#11-local-run)
12. [Docker deployment](#12-docker-deployment)
13. [Training notebook](#13-training-notebook)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations](#15-limitations)
16. [References](#16-references)

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

Open `http://localhost:8501` and sign in (see credentials below), or click a
demo sample to start immediately. API docs: `http://localhost:8000/docs`.
Health: `http://localhost:8000/health`.

Trained weights must exist at `models/clinvision_pneumonia.keras`
(see [Training notebook](#13-training-notebook)).

## 2. Demo credentials

| Role   | Username | Password    |
|--------|----------|-------------|
| Doctor | `doctor` | `doctor123` |

Used by the Streamlit login and `POST /auth/login`. Change via
`DOCTOR_USERNAME` / `DOCTOR_PASSWORD` in `.env`. The UI also offers an
offline demo mode (local credential check) when the API is unreachable;
inference itself always requires the API.

## 3. Demo samples

`data/demo/` holds 5 curated RSNA DICOMs (committed, ~700 KB) described by
`data/demo/manifest.json`:

| Sample | RSNA class | Use |
|--------|-----------|-----|
| `00436515…` / `00704310…` | Lung Opacity | Positive examples |
| `003d8fa0…` / `009482dc…` | Normal | Healthy negatives |
| `0004cfab…` | No Lung Opacity / Not Normal | Abnormal non-pneumonia |

The Analyze tab shows them as a thumbnail gallery: **Use** loads a sample
through the exact same pipeline as an upload (preview, context reset, DICOM
autofill, analysis, evidence, archive). A manual upload always wins over an
active demo; **Remove demo** goes back to an empty uploader.

## 4. What is implemented

- RSNA Pneumonia Detection Challenge dataset handling (notebook)
- Patient-level train/validation/test splitting (no leakage)
- DICOM preprocessing with `pydicom` (modality rescale + VOI LUT,
  `MONOCHROME1` polarity, percentile normalisation, 320x320 resize)
- DenseNet121 classifier (ImageNet transfer learning, sigmoid output)
- Full evaluation: accuracy, ROC-AUC, PR-AUC, precision, recall,
  specificity, F1, confusion matrix, threshold sweep
- Grad-CAM explainability replayed layer-by-layer (nested-backbone safe),
  returning overlay + raw heatmap + mask so the UI can re-blend at any
  opacity with the uploaded image kept pixel-identical where cold
- FastAPI service: request ids, timing metadata, file-size limits, typed
  errors, startup model preload, SQLite demo log + full patient archive
- Doctor login (bearer tokens, 12 h TTL) + token-gated endpoints
- Streamlit doctor portal: Light/Dark themes synced with Streamlit, stepper
  flow, DICOM demo gallery, upload-replaces logic, DICOM sidebar autofill,
  probability gauge with risk band, Original-vs-Grad-CAM evidence with
  intensity slider, clinical summary, patient history table, PDF export
- Free-LLM summary chain (Ollama local, Groq, Gemini, Hugging Face, custom
  endpoint) with a "ClinVision Assistant" role and deterministic template
  fallback — the app runs with no LLM configured at all

## 5. How it works

```text
DICOM / JPG / PNG  (or 1-click demo sample, <= MAX_UPLOAD_MB)
        |
DICOM-aware preprocessing -> 320x320 gray -> RGB replicate (0-255 px)
        |
DenseNet121 -> pneumonia probability (threshold 0.5)
        |
Grad-CAM (overlay + heatmap + mask; failure-isolated, never breaks prediction)
        |
Structured summary -> optional free-LLM rewrite (4 imposed sections)
        |
FastAPI (JSON) -> Streamlit portal -> SQLite patient archive + PDF report
```

Clinical context (age/sex/symptoms) is display-only for the summary; it is
never fed to the vision model. Uploading a new file replaces the previous
analysis display; removing the file empties the area. Every new file resets
the sidebar context to defaults, refilled from DICOM tags when present.

## 6. Repository structure

```text
clinvision-ai-final/
├── ClinVision_AI_RSNA.ipynb   # full ML workflow (single source of truth)
├── backend/
│   └── main.py                # FastAPI: S0 config … S8 endpoints (see header)
├── frontend/
│   └── app.py                 # Streamlit portal: F0 setup … F8 tabs (see header)
├── data/
│   ├── demo/                  # 5 curated DICOMs + manifest.json (COMMITTED)
│   ├── raw/                   # full Kaggle dataset (not in git)
│   ├── cache_320/             # generated resized PNG cache (not in git)
│   └── metadata/              # generated split CSVs (not in git)
├── models/
│   └── clinvision_pneumonia.keras   # trained weights (not in git)
├── artifacts/                 # metrics, plots, history.sqlite3,
│                              # patient_images/ (not in git)
├── .streamlit/config.toml     # server settings (upload limit 20 MB)
├── Dockerfile                 # CPU serving image (API + UI + demo subset)
├── docker-compose.yml         # api + streamlit services
├── requirements.txt           # single commented requirements file
├── .env.example               # all settings documented
├── SOURCE_REQUIREMENTS.md
└── README.md
```

Dataset, trained weights, PNG cache and archives are intentionally not
committed — except the 5-file demo subset, which is part of the product.

## 7. Backend API reference

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
| POST   | `/explain`          | †     | Probability + Grad-CAM trio (base64 PNGs)     |
| POST   | `/summary`          | †     | Summary for a probability + context (form)    |
| POST   | `/analyze`          | †     | All-in-one + patient archive + DICOM tags     |
| GET    | `/history?limit=50` | †     | Recent demo predictions                       |
| DELETE | `/history`          | †     | Clear demo predictions                        |
| GET    | `/patients`         | †     | Patient archive (metadata + LLM note)         |
| GET    | `/patients/{id}`    | †     | One full patient record                       |
| GET    | `/patients/{id}/image`   | † | Archived original upload                      |
| GET    | `/patients/{id}/gradcam` | † | Archived Grad-CAM overlay                     |
| DELETE | `/patients/{id}`    | †     | Delete record + files                         |
| DELETE | `/patients`         | †     | Clear the whole archive                       |

\* Token = `Authorization: Bearer <token>` from `/auth/login`.
† Required only when `REQUIRE_AUTH=true` (default `false`, so existing
scripts keep working; the Streamlit UI always enforces its own login).

Examples:

```bash
# Login (JSON)
curl -X POST localhost:8000/auth/login/json \
  -H 'Content-Type: application/json' \
  -d '{"username":"doctor","password":"doctor123"}'

# Full analysis (prediction + Grad-CAM + summary + archive)
curl -X POST localhost:8000/analyze \
  -F "file=@data/demo/00436515-870c-4b36-a041-de91049b9ab4.dcm" \
  -F "age=58" -F "sex=Male" -F "symptoms=cough, fever"

# Probability + Grad-CAM trio only
curl -X POST localhost:8000/explain -F "file=@sample.dcm"

# Patient archive
curl localhost:8000/patients?limit=10
```

Error contract: `400` empty/oversize upload or bad probability, `401` bad
credentials (or missing token when `REQUIRE_AUTH=true`), `413` file over
`MAX_UPLOAD_MB`, `422` undecodable image / inference failure, `404`
unknown patient record, `500` unexpected.

## 8. Frontend tour

Start: `streamlit run frontend/app.py` → `http://localhost:8501`

1. **Doctor Sign In** — username/password form in a centered card; shows an
   API status pill; falls back to offline demo mode if the API is down.
2. **Analyze tab** — stepper-guided flow (Upload → Analyze → Review):
   file upload (replace-not-add: a new file clears the old display,
   removing the file empties everything) **or** the 5-sample demo gallery;
   DICOM sidebar autofill (age/sex from tags, full reset otherwise);
   preview with DICOM metadata line (`Patient ID · Study · Modality`);
   **Analyze image** with staged progress. Results show a risk-band pill,
   native metrics, a Plotly gauge, request/model metadata, DICOM metadata,
   Original-vs-Grad-CAM evidence with view switcher + intensity slider
   (masked blend: cold anatomy stays pixel-identical to the upload),
   clinical summary with LLM-source badge, limitations, and PDF export.
3. **History tab** — patient archive table (Date, File, Patient ID,
   Prediction, Probability, Age, Sex, LLM) with record viewer (info,
   archived images, full LLM summary, delete), refresh + clear-all; below
   it the session table and the lightweight server log.
4. **About tab** — pipeline recap and the Ollama one-liner.

Sidebar: patient context (text areas fixed-size with scrollbars, never
resizable), display theme (Light/Dark, synced into Streamlit), session info
(API URL, upload limit, active free-LLM backends), status refresh and clear
actions. Health/history calls are cached to keep the UI responsive.

## 9. Configuration

Copy `.env.example` → `.env`:

| Variable               | Default                                  | Purpose                              |
|------------------------|------------------------------------------|--------------------------------------|
| `MODEL_PATH`           | `models/clinvision_pneumonia.keras`      | Weights file (repo-root relative)    |
| `MODEL_VERSION`        | `clinvision-v2`                          | Version string in responses          |
| `API_URL`              | `http://localhost:8000`                  | Backend URL for Streamlit            |
| `MAX_UPLOAD_MB`        | `20`                                     | Upload size limit (also `.streamlit`)| 
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

Never commit API keys or patient data (`.env`, `artifacts/` ignored).

## 10. Free LLM setup

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

The **ClinVision Assistant** role structures every note into Summary,
Explanation, Indications and next steps, Limitations — temperature 0,
fields-only grounding, no invented findings, no drugs/doses, always framed
as AI output for professional review. The PDF report renders these sections
with teal headings (graceful fallback to raw text for the template).

## 11. Local run

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000          # API
API_URL=http://localhost:8000 streamlit run frontend/app.py  # UI
```

Hardware notes:

- **NVIDIA CUDA (Linux/WSL2):** `pip install "tensorflow[and-cuda]==2.18.1`,
  verify with `nvidia-smi` then `tf.config.list_physical_devices('GPU')`.
- **Apple Silicon (M1–M4):** `pip install tensorflow-metal==1.2.0` in a
  native arm64 Python 3.11 venv.
- **CPU:** base `requirements.txt` works; reduce `BATCH_SIZE`/`IMAGE_SIZE`
  in the notebook if memory is tight.

## 12. Docker deployment

```bash
docker compose build
docker compose up
```

| Service   | URL                            |
|-----------|--------------------------------|
| Streamlit | `http://localhost:8501`        |
| FastAPI   | `http://localhost:8000/docs`   |

The image is CPU-based (`python:3.11-slim` + OpenGL libs for OpenCV) with a
healthcheck on `/health`. `COPY` bakes in code + the 700 KB demo subset;
compose mounts `./models` read-only and `./artifacts` read-write. The
uploader limit is aligned in three places: `MAX_UPLOAD_MB` (both services)
and `.streamlit/config.toml`. For host Ollama, point the API at
`OLLAMA_BASE_URL=http://host.docker.internal:11434`. For GPU training use a
CUDA host or Colab, not this serving image.

## 13. Training notebook

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
8. Stage B: fine-tune last layers at low LR, BatchNorm frozen.
9. Evaluation: accuracy @0.5 and @validation-selected threshold, ROC-AUC,
   PR-AUC, precision, recall, specificity, F1, confusion matrix, curves.

Model profile: input 320×320×3, DenseNet121 (ImageNet) + GAP + BatchNorm +
Dropout 0.20 + Dense(1, sigmoid), AdamW. Reaching 90% validation accuracy is
not guaranteed — always read recall, specificity and AUC alongside accuracy
on this imbalanced task.

## 14. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/health` shows `degraded` | `MODEL_PATH` wrong or `.keras` missing — train notebook or restore file |
| Streamlit "API unreachable" | API not running, wrong `API_URL` (Docker: `http://api:8000`), or port clash (`lsof -i :8000`) |
| Login fails | Check `DOCTOR_USERNAME`/`DOCTOR_PASSWORD`; expired token → log in again |
| `422` on upload | File corrupt/unsupported (DCM/JPG/PNG only) or under 32 px |
| `413` on upload | Over `MAX_UPLOAD_MB`; raise limit in `.env`, compose AND `.streamlit/config.toml` |
| Template summary instead of LLM | No backend reachable — check `GET /llm/status`, `ollama serve`, model pulled |
| Overlay looks like another image | Fixed: blends are keyed per-image and the Original panel always decodes the analyzed bytes; re-pull latest code |
| Gallery missing | `data/demo/` absent (minimal installs hide the section gracefully) |
| TF sees no GPU (NVIDIA) | Check `nvidia-smi` first, then the CUDA TF install |
| TF sees no GPU (Mac) | Native arm64 venv + `tensorflow-metal==1.2.0` |
| OOM in training | Lower `BATCH_SIZE` (e.g. 4) / `IMAGE_SIZE` (e.g. 256), prefer local SSD |

## 15. Limitations

Engineering prototype on the RSNA Pneumonia Detection Challenge data: no
clinical/prospective validation, no regulatory clearance, no guaranteed
reliability on real hospital populations. Grad-CAM is explanatory, not a
validated lesion localiser. Do not use as a standalone diagnostic system.

## 16. References

- RSNA Pneumonia Detection Challenge: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- TensorFlow pip install: https://www.tensorflow.org/install/pip
- Apple Metal: https://developer.apple.com/metal/
- Keras Grad-CAM example: https://keras.io/examples/vision/grad_cam/

