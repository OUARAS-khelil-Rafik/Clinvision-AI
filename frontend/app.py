"""
ClinVision AI — Streamlit frontend.

The Streamlit application is intentionally self-contained and communicates with
FastAPI through HTTP. The notebook remains the single source of truth for ML
training and evaluation.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

st.set_page_config(page_title="ClinVision AI", page_icon="🩻", layout="wide")

st.title("ClinVision AI")
st.caption("Clinical Imaging Copilot — RSNA Pneumonia Detection")
st.warning(
    "AI assistance only. The prediction and Grad-CAM are model outputs, not a medical diagnosis. "
    "Clinical decisions must be made by qualified professionals."
)

with st.sidebar:
    st.header("Clinical context")
    age = st.text_input("Age", placeholder="e.g. 58")
    sex = st.selectbox("Sex", ["", "Female", "Male", "Other / not specified"])
    symptoms = st.text_area("Symptoms", placeholder="e.g. cough, fever, dyspnea")
    context = st.text_area("Additional context", placeholder="History or context provided for the summary only")
    st.divider()
    st.write(f"FastAPI: `{API_URL}`")
    st.write(f"Upload limit: `{MAX_UPLOAD_MB} MB`")

uploaded = st.file_uploader("Upload chest X-ray", type=["dcm", "jpg", "jpeg", "png"])

if uploaded:
    columns = st.columns(2)
    with columns[0]:
        st.subheader("Input")
        if uploaded.name.lower().endswith(".dcm"):
            st.info("DICOM uploaded. The API will perform DICOM-aware preprocessing.")
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if st.button("Analyze", type="primary", use_container_width=True):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
        data = {"age": age, "sex": sex, "symptoms": symptoms, "context": context}
        try:
            with st.spinner("Running ClinVision AI..."):
                response = requests.post(f"{API_URL}/analyze", files=files, data=data, timeout=120)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"FastAPI is unavailable or returned an error: {exc}")
            st.stop()
        except ValueError as exc:
            st.error(f"The API returned invalid JSON: {exc}")
            st.stop()

        with columns[1]:
            st.subheader("Model result")
            prediction = result["prediction"]
            probability = float(result["probability"])
            st.metric("Prediction", prediction)
            st.metric("Pneumonia probability", f"{probability:.1%}")
            st.progress(probability, text="Pneumonia probability")
            st.caption(f"Model: {result.get('model_version')} · {result.get('inference_ms')} ms")

        st.divider()
        left, right = st.columns(2)
        with left:
            st.subheader("Grad-CAM")
            encoded = result.get("gradcam_png_base64")
            if encoded:
                st.image(base64.b64decode(encoded), caption="Model attention visualization", use_container_width=True)
                st.caption("Grad-CAM indicates influential model regions; it is not a validated lesion-localization method.")
            else:
                st.info("Grad-CAM was not returned by the API.")

        with right:
            st.subheader("Clinical summary")
            summary = result.get("clinical_summary", {})
            st.write(summary.get("summary", "No summary available."))
            st.json({
                "assessment": summary.get("assessment"),
                "model_probability": summary.get("model_probability"),
                "llm_enabled": summary.get("llm_enabled", False),
                "clinical_context": summary.get("clinical_context", {}),
            })
            st.subheader("Limitations")
            for item in summary.get("limitations", []):
                st.write(f"• {item}")
else:
    st.info("Upload a DICOM, JPG or PNG chest X-ray to begin.")

st.divider()
st.caption("ClinVision AI — AI-assisted prototype. Do not use as a standalone diagnostic system.")
