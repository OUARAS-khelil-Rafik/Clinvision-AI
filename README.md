# ClinVision AI — RSNA Pneumonia Detection

> Clinical Imaging Copilot — educational / hackathon MVP

ClinVision AI is a chest X-ray analysis workflow built around the RSNA Pneumonia Detection Challenge. The project combines DICOM-aware preprocessing, a DenseNet121 classifier, Grad-CAM explainability, structured clinical context, FastAPI inference, and a Streamlit frontend.

This repository is intended as an engineering prototype and is not a medical device, autonomous diagnostic system, or clinical decision tool.

## 1. Project goal

The core workflow is:

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

The model output should be interpreted alongside professional review and contextual clinical information.

## 2. What is implemented

The repository follows the supplied implementation plan and includes:

- RSNA Pneumonia Detection Challenge dataset handling
- patient-level train/validation/test splitting
- DICOM preprocessing with `pydicom`
- PNG cache generation for faster training
- DenseNet121 backbone with GlobalAveragePooling2D, dropout, and sigmoid output
- binary cross-entropy training with Adam-family optimization
- AUC, PR-AUC, accuracy, precision, recall, specificity, F1, confusion matrix
- Grad-CAM explainability
- FastAPI endpoints for inference and reporting
- Streamlit application with clinical context and image upload
- optional YOLO detection as a separate future extension

## 3. Repository structure

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
│   └── ...                  # checkpoints and generated artifacts
├── artifacts/               # plots, metrics, Grad-CAMs, reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-cuda.txt
├── requirements-mac.txt
├── .env.example
├── .gitignore
├── README.md
└── .venv/                  # local environment (optional)
```

The dataset, trained weights, and generated cache are intentionally not checked into Git.

## 4. Why only one notebook?

All training, preprocessing, cache generation, model creation, evaluation, Grad-CAM, threshold analysis, and artifact export are centralized in:

```text
ClinVision_AI_RSNA.ipynb
```

The only runtime application files are:

```text
backend/main.py
frontend/app.py
```

This keeps the project simple: the full ML workflow lives in one notebook while the serving layer remains lightweight and deployable.

## 5. Model and training profile

This implementation keeps the DenseNet121 architecture while slightly extending the original pipeline for accuracy-focused training.

```text
Input                320×320×3
Backbone             DenseNet121 (ImageNet pretrained)
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

Important notes:

- The dataset is split at the patient level, not at the image-row level.
- Validation is used to select thresholds, and the test set remains untouched until final evaluation.
- `class_weight="balanced"` is not the default here, since it can improve sensitivity but may reduce raw accuracy at a 0.5 decision threshold.

## 6. Hardware support

The notebook detects the active TensorFlow device automatically.

### NVIDIA CUDA — Linux / WSL2

```bash
python -m pip install "tensorflow[and-cuda]==2.18.1"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

A compatible NVIDIA driver is required on the host machine.

### Apple Silicon — M1/M2/M3/M4

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow-metal==1.2.0
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

TensorFlow on Apple Silicon uses the Metal plugin, which is platform-specific and is not the same as the PyTorch `mps` API.

### CPU

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CPU training, reduce `IMAGE_SIZE` or `BATCH_SIZE` as needed.

## 7. Dataset

Download the RSNA Pneumonia Detection Challenge dataset from Kaggle. Required files:

```text
stage_2_train_images/
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
stage_2_sample_submission.csv
```

Place the extracted dataset under:

```text
data/raw/
```

The notebook will discover the labels and DICOM directory automatically.

## 8. Notebook workflow

Open and run:

```text
ClinVision_AI_RSNA.ipynb
```

Execute the notebook cells in order.

### Step 1 — Environment and device checks

The notebook prints:

- Python version
- TensorFlow version
- visible GPU devices
- CPU fallback status
- Apple Metal availability (if relevant)

### Step 2 — Dataset discovery

The notebook locates the extracted data and validates the expected files.

### Step 3 — Data audit

The notebook computes:

- row counts
- unique patient counts
- positive/negative patient counts
- missing values
- duplicate patient identifiers
- images with multiple positive boxes
- invalid or corrupted files

### Step 4 — Patient-level split

The pipeline splits by patient to avoid leakage between partitions.

Default split:

```text
Train      75%
Validation 15%
Test       10%
```

The test set remains untouched until the final evaluation stage.

### Step 5 — DICOM preprocessing and cache

The notebook:

1. reads DICOM files using `pydicom`
2. applies VOI LUT when available
3. applies modality rescale slope/intercept
4. handles `MONOCHROME1` polarity
5. normalizes intensity values robustly using percentiles
6. resizes to the configured `IMAGE_SIZE`
7. writes compact normalized PNGs to the cache

This avoids expensive DICOM decoding at every training step.

### Step 6 — TensorFlow dataset pipeline

The model consumes cached PNGs directly. Training augmentation is applied only to training data:

- small rotation
- small translation
- small zoom
- mild contrast variation

Horizontal flipping is intentionally avoided to preserve laterality information.

### Step 7 — Stage A training

The classifier head is trained while the DenseNet121 backbone remains frozen.

Callbacks include:

- best validation accuracy checkpoint
- best validation AUC checkpoint
- `ReduceLROnPlateau`
- `EarlyStopping`
- CSV logger

### Step 8 — Stage B fine-tuning

The last DenseNet layers are unfrozen and fine-tuned with a lower learning rate. BatchNorm layers remain frozen for stability.

### Step 9 — Evaluation

The notebook reports:

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

Two accuracy forms are reported:

1. Default 0.50 threshold
2. Validation-selected threshold (frozen before final test evaluation)

This avoids tuning the threshold on the test set.

## 9. Accuracy expectations

The code is optimized for higher accuracy, but reaching 90% validation accuracy is not guaranteed. It depends on the split, preprocessing, hardware, random seed, and convergence behavior.

Accuracy alone is not enough for an imbalanced medical dataset. The notebook also reports recall, specificity, precision, ROC-AUC, and PR-AUC because a model that predicts almost everything as negative can look deceptively accurate.

## 10. Grad-CAM

Grad-CAM is generated from the final convolutional DenseNet feature map. It is included as explainability output rather than as a certification of lesion localization.

The notebook generates:

- original X-ray
- Grad-CAM heatmap
- original + Grad-CAM overlay

The Streamlit interface uses the same logic for uploaded images.

## 11. FastAPI backend

Start the API:

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

## 12. Streamlit frontend

Start the app:

```bash
streamlit run frontend/app.py
```

The UI includes:

- age
- sex
- symptoms
- clinical context
- DICOM/JPG/PNG upload
- pneumonia probability
- confidence visualization
- Grad-CAM output
- structured summary
- safety notice

## 13. Docker deployment

The default Docker image is CPU-friendly and intentionally simple.

```bash
docker compose build
docker compose up
```

Access points:

```text
Streamlit: http://localhost:8501
FastAPI:   http://localhost:8000/docs
```

The trained model should exist at:

```text
models/clinvision_pneumonia.keras
```

and is mounted read-only into the containers.

For CUDA-based training, use a GPU-enabled host or Colab/Brev instead of trying to force CUDA into the default lightweight serving container.

## 14. Secrets and environment

Copy:

```text
.env.example -> .env
```

Optional LLM configuration:

```text
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
```

The application works without an LLM. In that case, the summary endpoint uses a deterministic template-based summary.

Never commit API keys or patient data.

## 15. Production-minded behavior

The application includes:

- file size checks
- content-type validation
- DICOM / JPG / PNG handling
- corrupted-file error handling
- model-existence checks
- model version in responses
- request IDs
- timing metadata
- optional SQLite demo history
- deterministic fallback summary
- Grad-CAM failure isolation
- JSON responses suitable for frontend integration

## 16. Optional YOLO extension

The original plan proposes YOLO or Faster R-CNN as a later extension. This repository intentionally keeps detection out of the core inference path.

A proper detector should be trained and evaluated separately from the classification model. Grad-CAM is explanatory only; it is not a substitute for bounding-box detection.

## 17. Troubleshooting

### Invalid DICOM or image files

The cache builder validates each image and records failures in `artifacts/preprocess_errors.csv` instead of crashing the full TensorFlow pipeline.

### TensorFlow sees no GPU

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

For NVIDIA, verify `nvidia-smi` first. For Apple Silicon, install the correct `tensorflow-metal` package in a native arm64 environment.

### Out-of-memory

Reduce:

```python
BATCH_SIZE = 4
IMAGE_SIZE = 256
```

before reducing DenseNet capacity.

### Training is too slow

The cache architecture is designed to avoid DICOM decoding inside every training step. If training remains slow, benchmark:

```text
storage → PNG decode → augmentation → GPU
```

and prefer local SSD storage over network storage.

## 18. Evaluation benchmark template

After training, save the generated report from `artifacts/final_metrics.json` and record the results in your project log.

```text
Split         Accuracy   ROC-AUC   PR-AUC   Precision   Recall   Specificity   F1
Train         generated  generated  generated  generated  generated  generated  generated
Validation    generated  generated  generated  generated  generated  generated  generated
Test          generated  generated  generated  generated  generated  generated  generated
```

Do not insert fabricated benchmark numbers into the README.

## 19. Scientific and clinical limitations

This repository is an engineering prototype based on the RSNA Pneumonia Detection Challenge dataset. It does not constitute clinical validation, prospective validation, regulatory clearance, or guaranteed diagnostic reliability in a real hospital population.

## 20. References

- RSNA Pneumonia Detection Challenge dataset: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- TensorFlow installation guide: https://www.tensorflow.org/install/pip
- Apple Metal: https://developer.apple.com/metal/
- Keras Grad-CAM example: https://keras.io/examples/vision/grad_cam/
