# Source requirements traceability

This file maps the supplied ClinVision AI implementation plan to this repository.

- Objective and MVP: chest X-ray analysis, pneumonia probability, Grad-CAM, clinical context, structured summary.
- Data: RSNA Pneumonia Detection Challenge, stage_2_train_images, labels and detailed class info.
- Split: patient-level split.
- Model: DenseNet121 transfer learning.
- Metrics: accuracy, ROC-AUC, precision, recall, specificity, confusion matrix, PR-AUC.
- Explainability: Grad-CAM.
- API: /health, /predict, /explain, /summary, /analyze.
- UI: Streamlit upload, clinical context, probability, chart, Grad-CAM, summary, safety notice.
- Advanced detection: RSNA boxes are retained for exploration; a true detector remains an extension.
- Deployment: FastAPI + Streamlit + Docker.

Engineering extensions introduced intentionally: a third untouched test split, 320×320 cached preprocessing for speed/accuracy, AdamW, an accuracy-focused default profile without class weights, and a validation-only threshold selection step.
