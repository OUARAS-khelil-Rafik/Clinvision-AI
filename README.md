# ClinVision AI — RSNA Pneumonia Detection

> **Clinical Imaging Copilot — educational / hackathon MVP**
>
> This repository implements the project described in the supplied RSNA/ClinVision AI implementation plan: DICOM/JPG/PNG upload, chest-X-ray preprocessing, DenseNet121 pneumonia classification, Grad-CAM explainability, clinical context, structured summary, FastAPI inference, Streamlit UI, and optional YOLO detection.

## 1. Project goal

ClinVision AI is an AI-assisted workflow for chest radiography. The core MVP is:

```text
DICOM / JPG / PNG
        ↓
DICOM-aware preprocessing
        ↓
Cached 320×320 image
        ↓
DenseNet121 (ImageNet transfer learning)
        ↓
Pneumonia probability
        ↓
Grad-CAM explanation
        ↓
Structured clinical summary
        ↓
FastAPI + Streamlit
```

The system is **not a medical device and not an autonomous diagnostic system**. The probability and Grad-CAM are model outputs and require professional review.

## 2. Source specification implemented

The supplied plan requires:

- RSNA Pneumonia Detection Challenge data.
- Patient-level splitting, not row-level splitting.
- DICOM processing with `pydicom`.
- Resize to 224×224 in the original plan; this implementation defaults to **320×320** as an accuracy-oriented extension while keeping the same DenseNet121 architecture.
- DenseNet121 + GlobalAveragePooling2D + Dropout + sigmoid output.
- Binary cross-entropy / Adam-family optimization.
- AUC, precision, recall, accuracy, PR-AUC, specificity and confusion matrix.
- Grad-CAM for explanation.
- FastAPI endpoints `/health`, `/predict`, `/explain`, `/summary`, `/analyze`.
- Streamlit upload + clinical context + probability + Grad-CAM + structured summary.
- Optional YOLO extension only after the classification MVP is stable.

## 3. Why only one notebook?

All data work, preprocessing, caching, model creation, training, fine-tuning, evaluation, Grad-CAM, threshold analysis and artifact generation are contained in:

```text
ClinVision_AI_RSNA.ipynb
```

The only Python application files are:

```text
backend/main.py      # FastAPI
frontend/app.py      # Streamlit
```

This avoids the previous multi-script architecture and makes the complete ML workflow easy to execute in Jupyter or Colab.

## 4. Final tree

```text
clinvision-ai/
├── ClinVision_AI_RSNA.ipynb
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── data/
│   ├── raw/                 # Kaggle ZIP / extracted DICOMs
│   ├── cache_320/           # generated resized PNG cache
│   └── metadata/            # generated CSV split metadata
├── models/
│   ├── clinvision_pneumonia.keras
│   └── ...                  # generated best checkpoints
├── artifacts/               # plots, metrics, Grad-CAMs, reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-cuda.txt
├── requirements-mac.txt
├── .env.example
├── .gitignore
└── README.md
```

The dataset, trained weights and generated cache are intentionally not committed.

## 5. Hardware: CUDA, Apple Silicon/Metal, or CPU

The notebook detects the TensorFlow execution device automatically.

### NVIDIA CUDA — Linux / WSL2

TensorFlow's official pip guide documents the CUDA-enabled installation as:

```bash
python -m pip install "tensorflow[and-cuda]==2.18.1"
```

Then verify:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

A recent NVIDIA driver is required by the host. The notebook then reports the visible TensorFlow GPU.

### Apple Silicon — M1/M2/M3/M4

The project remains TensorFlow/Keras because that is the technology specified by the source plan. Apple currently provides a Metal plugin for TensorFlow acceleration; this is **not the same device API name as PyTorch `mps`**. TensorFlow sees the Metal-backed accelerator through its GPU device mechanism.

Recommended environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow-metal==1.2.0
```

Check:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the Metal plugin is unavailable, TensorFlow automatically falls back to CPU. Apple documentation describes Metal as the compute/ML acceleration layer for Apple GPUs; TensorFlow's own install page currently states that macOS GPU support is not official in core TensorFlow, so the Metal plugin should be treated as a platform-specific accelerator rather than a TensorFlow-native CUDA equivalent.

### CPU

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CPU training, reduce `IMAGE_SIZE` or `BATCH_SIZE` in the notebook to fit your machine.

## 6. Dataset

Download the **RSNA Pneumonia Detection Challenge** dataset from Kaggle. The required files are:

```text
stage_2_train_images/
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
stage_2_sample_submission.csv
```

### Kaggle on Colab

1. Create/download your Kaggle API token (`kaggle.json`).
2. Upload it to Colab.
3. Run the dataset setup cell in the notebook.

The notebook never expects patient data to be committed to Git.

## 7. Run the complete notebook

Open:

```text
ClinVision_AI_RSNA.ipynb
```

Execute cells in order.

### Step 1 — Environment and device

The notebook installs/checks dependencies and prints:

- Python version
- TensorFlow version
- GPU devices
- CPU fallback status
- Apple Metal availability when relevant

### Step 2 — Dataset discovery

Place/extract the RSNA dataset under:

```text
clinvision-ai/data/raw/
```

The notebook locates the labels and image directory automatically.

### Step 3 — Data audit

The notebook calculates:

- number of rows
- unique patients
- positive/negative patients
- missing values
- duplicate patient IDs
- number of positive images with multiple boxes
- invalid/corrupted files

### Step 4 — Patient-level split

The split is performed on a unique patient table. A patient never appears in two partitions.

Default:

```text
Train      75%
Validation 15%
Test       10%
```

The test partition is an engineering extension beyond the original plan and remains untouched until final evaluation.

### Step 5 — DICOM preprocessing and cache

The notebook:

1. reads DICOM with `pydicom`;
2. applies available VOI LUT information when possible;
3. applies modality rescale slope/intercept;
4. handles `MONOCHROME1` polarity;
5. robustly normalizes intensities using percentiles;
6. resizes to `IMAGE_SIZE`;
7. writes a compact normalized PNG cache.

Caching is critical: it avoids calling `pydicom` for every training epoch, which was one of the main causes of the very long training times in the earlier pipeline.

### Step 6 — tf.data pipeline

The model reads cached PNGs directly through TensorFlow. Training augmentation is applied only to training data:

- small rotation
- small translation
- small zoom
- mild contrast variation

**No horizontal flip is used by default**, preserving laterality information.

### Step 7 — Accuracy-oriented DenseNet121

The default training profile is designed for your stated goal of improving raw train/validation accuracy while still reporting medical metrics.

```text
Input                320×320×3
Backbone             DenseNet121 ImageNet
Pooling              GlobalAveragePooling2D
Normalization        BatchNormalization
Dropout              0.20
Output               Dense(1, sigmoid)
Optimizer            AdamW
Initial LR           3e-4
Fine-tune LR         5e-6
Fine-tune layers     80
Class weighting      OFF by default for accuracy profile
```

The `class_weight="balanced"` strategy used in the previous run is intentionally not the default here because it prioritizes minority-class sensitivity and can lower raw accuracy at threshold 0.5.

### Step 8 — Stage A

Train the classifier head while DenseNet121 is frozen.

Callbacks:

- best validation accuracy checkpoint
- best validation AUC checkpoint
- ReduceLROnPlateau
- EarlyStopping
- CSV logger

### Step 9 — Stage B

The last DenseNet layers are unfrozen and fine-tuned at a much smaller learning rate. Batch Normalization layers remain frozen for stability.

### Step 10 — Evaluation

The notebook computes:

- accuracy
- ROC-AUC
- PR-AUC
- precision
- recall / sensitivity
- specificity
- F1
- confusion matrix
- threshold sweep
- ROC curve
- precision-recall curve

Two accuracy values are reported:

1. **default 0.50 threshold** — the model's standard binary decision;
2. **validation-selected threshold** — selected only on validation and then frozen for final test evaluation.

This prevents using the test set to tune the threshold.

## 8. Important: reaching 90% accuracy

The code is explicitly optimized toward higher accuracy, but **90% validation accuracy is not guaranteed**. It depends on the actual RSNA split, preprocessing, hardware, random seed and training convergence.

More importantly, 90% accuracy by itself can be misleading on an imbalanced medical dataset. The notebook therefore reports recall, specificity, precision, ROC-AUC and PR-AUC next to accuracy.

A model that reaches 90% accuracy by predicting almost everything as negative is not equivalent to a clinically useful classifier.

## 9. Grad-CAM

Grad-CAM is generated from the final convolutional DenseNet feature tensor. It is presented as **model explainability**, not as a certified lesion localization method.

The notebook generates:

```text
Original X-ray
Grad-CAM heatmap
Original + Grad-CAM overlay
```

The Streamlit interface uses the same logic for uploaded images.

## 10. API

Start FastAPI:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

```text
GET  /health
POST /predict
POST /explain
POST /summary
POST /analyze
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.dcm"
```

## 11. Streamlit

Start the UI:

```bash
streamlit run frontend/app.py
```

The application provides:

- age
- sex
- symptoms
- clinical context
- DICOM/JPG/PNG upload
- pneumonia probability
- confidence visualization
- Grad-CAM
- structured summary
- safety notice

## 12. Docker

The default Docker image is CPU-compatible and intentionally keeps the deployment simple.

```bash
docker compose build
docker compose up
```

Then:

```text
Streamlit: http://localhost:8501
FastAPI:   http://localhost:8000/docs
```

The trained model must exist at:

```text
models/clinvision_pneumonia.keras
```

and is mounted read-only into the containers.

For NVIDIA CUDA training, use the notebook on a CUDA-capable host or Colab/Brev rather than forcing CUDA into the default lightweight serving image.

## 13. Secrets

Copy:

```text
.env.example → .env
```

Optional LLM settings:

```text
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
```

The application works without an LLM. In that case the `/summary` endpoint uses a deterministic template-based summary.

Never commit API keys or patient data.

## 14. Production-minded behavior

The application includes:

- file-size limit
- content-type validation
- DICOM/JPG/PNG handling
- corrupted-file errors
- model-file existence checks
- model version in responses
- request IDs
- inference timing
- optional SQLite demo history
- deterministic fallback clinical summary
- Grad-CAM failure isolation
- JSON responses suitable for frontend integration

## 15. Optional YOLO extension

The original plan proposes YOLO/Faster R-CNN as a later extension. This repository deliberately does not make detection part of the core inference path. The RSNA bounding-box annotations remain available in the notebook for exploration and for a future detector.

A true detector should be trained and evaluated separately from Grad-CAM. Grad-CAM is not a substitute for bounding-box detection.

## 16. Troubleshooting

### `Thumbs.db` or invalid DICOM/image files

The notebook's cache builder validates every image and records failures in `artifacts/preprocess_errors.csv` instead of allowing one bad file to crash the entire TensorFlow pipeline.

### TensorFlow sees no GPU

Run:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

For NVIDIA, verify `nvidia-smi` first.

For Apple Silicon, install the compatible `tensorflow-metal` package in a native arm64 Python environment. TensorFlow core's macOS page currently describes macOS as CPU-only officially; Metal acceleration is provided by Apple's plugin ecosystem.

### Out-of-memory

Reduce:

```python
BATCH_SIZE = 4
IMAGE_SIZE = 256
```

before reducing the DenseNet capacity.

### Training is too slow

The cache architecture is intentionally designed to avoid decoding DICOMs inside every training step. If an epoch is still slow, benchmark:

```text
storage → PNG decode → augmentation → GPU
```

and use a local SSD rather than network storage.

## 17. Evaluation benchmark template

After training, save the generated report from `artifacts/final_metrics.json` and record:

| Split | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | generated | generated | generated | generated | generated | generated | generated |
| Validation | generated | generated | generated | generated | generated | generated | generated |
| Test | generated | generated | generated | generated | generated | generated | generated |

Do **not** insert fabricated benchmark numbers into the README.

## 18. Scientific / clinical limitation

This repository is an engineering prototype based on the RSNA challenge dataset. Dataset performance cannot be interpreted as clinical validation, prospective validation, regulatory clearance, or diagnostic reliability in a real hospital population.

## 19. References

- RSNA Pneumonia Detection Challenge dataset: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- TensorFlow installation guide: https://www.tensorflow.org/install/pip
- Apple Metal: https://developer.apple.com/metal/
- Grad-CAM example: https://keras.io/examples/vision/grad_cam/
