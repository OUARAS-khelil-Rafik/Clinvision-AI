"""
ClinVision AI — Streamlit frontend (doctor portal).

Self-contained: talks to FastAPI over HTTP, gates the UI behind a doctor
login, and renders a premium clinical UX (cards, gauge, Grad-CAM, summary,
history, report downloads).
"""
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
# Local fallback so the demo login works even if the API is briefly down.
LOCAL_USER = os.getenv("DOCTOR_USERNAME", "doctor")
LOCAL_PASS = os.getenv("DOCTOR_PASSWORD", "doctor123")

st.set_page_config(
    page_title="ClinVision AI — Doctor Portal",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Premium clinical theme (CSS)
# -----------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.stApp { background: linear-gradient(180deg, #f4f8fc 0%, #eef4fa 100%); }
.block-container { padding-top: 1.2rem; max-width: 1200px; }

/* Hero / brand */
.hero {
  background: linear-gradient(120deg, #0b3d5f 0%, #0e6ba8 55%, #12a5af 100%);
  color: white; border-radius: 18px; padding: 26px 30px;
  box-shadow: 0 12px 32px rgba(11,61,95,.25); margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; }
.hero p { margin: 6px 0 0; opacity: .92; font-size: .98rem; }
.hero .badge {
  display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3);
  padding: 4px 12px; border-radius: 999px; font-size: .75rem; font-weight: 600; margin-bottom: 10px;
}

/* Cards */
.card {
  background: white; border-radius: 16px; padding: 20px 22px;
  box-shadow: 0 6px 22px rgba(15,45,70,.08); border: 1px solid #e3edf5; margin-bottom: 16px;
}
.card h3 { margin: 0 0 8px; font-size: 1.05rem; color: #0b3d5f; }
.metric-card {
  background: white; border-radius: 14px; padding: 16px 18px; text-align: center;
  border: 1px solid #e3edf5; box-shadow: 0 4px 14px rgba(15,45,70,.07);
}
.metric-card .label { font-size: .75rem; font-weight: 600; color: #5b7a93; text-transform: uppercase; letter-spacing: .06em; }
.metric-card .value { font-size: 1.5rem; font-weight: 800; color: #0b3d5f; margin-top: 4px; }
.metric-card.positive .value { color: #c0392b; }
.metric-card.negative .value { color: #1e8449; }

/* Login */
.login-wrap { max-width: 460px; margin: 6vh auto 0; }
.login-card {
  background: white; border-radius: 20px; padding: 34px 34px 28px;
  box-shadow: 0 18px 50px rgba(11,61,95,.18); border: 1px solid #e3edf5; text-align: center;
}
.login-card .logo {
  width: 64px; height: 64px; border-radius: 18px; margin: 0 auto 12px;
  background: linear-gradient(135deg, #0b3d5f, #12a5af);
  display: flex; align-items: center; justify-content: center; font-size: 30px;
}
.login-card h2 { margin: 0; color: #0b3d5f; font-size: 1.45rem; font-weight: 800; }
.login-card p.sub { color: #5b7a93; font-size: .9rem; margin: 6px 0 18px; }
.hint-box {
  background: #f0f7fd; border: 1px dashed #9cc3e3; border-radius: 10px;
  padding: 10px 12px; font-size: .82rem; color: #0b3d5f; margin-top: 14px; text-align: left;
}

/* Buttons */
.stButton > button[kind="primary"] {
  background: linear-gradient(120deg, #0e6ba8, #12a5af) !important;
  border: none; border-radius: 12px; font-weight: 700; padding: .6rem 1rem;
  box-shadow: 0 6px 16px rgba(14,107,168,.3);
}
.stButton > button { border-radius: 12px; font-weight: 600; }

/* Pills */
.pill {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: .75rem; font-weight: 700;
}
.pill-ok { background: #e6f6ec; color: #1e8449; border: 1px solid #bfe6cd; }
.pill-warn { background: #fdf1e3; color: #b7791f; border: 1px solid #f5d9a8; }
.pill-bad { background: #fdecea; color: #c0392b; border: 1px solid #f5c6c0; }
.pill-llm { background: #eef2ff; color: #3b5bdb; border: 1px solid #cdd8ff; }

.disclaimer {
  background: #fff8e6; border: 1px solid #f0d48a; border-radius: 12px;
  padding: 12px 16px; font-size: .85rem; color: #7a5b00; margin-top: 10px;
}
.footer { text-align: center; color: #7d99b0; font-size: .78rem; margin: 22px 0 8px; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, default in {
    "authenticated": False,
    "token": None,
    "username": None,
    "result": None,
    "result_filename": None,
    "session_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def api_headers() -> dict:
    if st.session_state.get("token"):
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def do_login(username: str, password: str) -> tuple[bool, str]:
    """Try FastAPI login first, fall back to local demo credentials offline."""
    username = (username or "").strip()
    # 1) JSON endpoint (preferred)
    try:
        r = requests.post(f"{API_URL}/auth/login/json",
                          json={"username": username, "password": password}, timeout=8)
        if r.status_code == 200:
            data = r.json()
            st.session_state.authenticated = True
            st.session_state.token = data.get("access_token")
            st.session_state.username = data.get("username", username)
            return True, "Signed in via ClinVision API."
        if r.status_code == 401:
            return False, "Invalid username or password."
    except requests.RequestException:
        pass
    # 2) Form endpoint (legacy compat)
    try:
        r = requests.post(f"{API_URL}/auth/login",
                          data={"username": username, "password": password}, timeout=8)
        if r.status_code == 200:
            data = r.json()
            st.session_state.authenticated = True
            st.session_state.token = data.get("access_token")
            st.session_state.username = data.get("username", username)
            return True, "Signed in via ClinVision API."
        if r.status_code == 401:
            return False, "Invalid username or password."
    except requests.RequestException:
        pass
    # 3) Offline fallback — API unreachable, check local demo credentials
    if username == LOCAL_USER and password == LOCAL_PASS:
        st.session_state.authenticated = True
        st.session_state.token = None
        st.session_state.username = username
        return True, "API unreachable — signed in offline (demo mode)."
    if api_health() is None:
        return False, "API unreachable and local credentials did not match."
    return False, "Invalid username or password."


def do_logout() -> None:
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.result = None
    st.rerun()


def gauge(probability: float):
    color = "#c0392b" if probability >= 0.5 else "#1e8449"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "font": {"size": 30, "color": "#0b3d5f"}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color},
               "bgcolor": "#eef4fa",
               "steps": [{"range": [0, 50], "color": "#e6f6ec"},
                         {"range": [50, 100], "color": "#fdecea"}],
               "threshold": {"line": {"color": "#0b3d5f", "width": 3}, "value": 50}},
        title={"text": "Pneumonia probability", "font": {"size": 14, "color": "#5b7a93"}},
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_family="Inter")
    return fig


def build_txt_report(result: dict, filename: str) -> str:
    s = result.get("clinical_summary", {})
    lines = [
        "ClinVision AI — Analysis Report", "=" * 34,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"File: {filename}", f"Model: {result.get('model_version')} "
        f"({result.get('inference_ms')} ms)",
        "", f"Prediction: {result.get('prediction')}",
        f"Probability: {float(result.get('probability', 0)):.2%}",
        "", "Clinical summary:", s.get("summary", "n/a"), "",
        f"LLM: {'yes (' + str(s.get('llm_provider')) + ')' if s.get('llm_enabled') else 'template fallback'}",
        "", "Limitations:",
    ]
    lines += [f" - {x}" for x in s.get("limitations", [])]
    lines += ["", "AI assistance only — not a diagnosis. Review by a qualified physician required."]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown(
            """<div class="hero">
            <div class="badge">🏥 CLINICAL IMAGING COPILOT</div>
            <h1>ClinVision AI</h1>
            <p>RSNA pneumonia detection with DICOM-aware preprocessing,<br>
            DenseNet121 classification, Grad-CAM explainability and a free-LLM clinical summary.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="metric-card"><div class="label">Input</div><div class="value" style="font-size:1rem">DICOM · JPG · PNG</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="metric-card"><div class="label">Explainability</div><div class="value" style="font-size:1rem">Grad-CAM</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="metric-card"><div class="label">Summary LLM</div><div class="value" style="font-size:1rem">100% Free</div></div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="card"><h3>🛡️ Safety first</h3>
            <span style="font-size:.88rem;color:#43617a">AI assistance only. Predictions and Grad-CAM are model
            outputs — never a standalone diagnosis. All decisions require a qualified physician.</span></div>""",
            unsafe_allow_html=True,
        )
        health = api_health()
        if health and health.get("model_loaded"):
            st.markdown('<span class="pill pill-ok">● API online · model loaded</span>', unsafe_allow_html=True)
        elif health:
            st.markdown('<span class="pill pill-warn">● API online · model warming up</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-bad">● API unreachable — offline demo login still works</span>', unsafe_allow_html=True)

    with right:
        st.markdown(
            """<div class="login-wrap"><div class="login-card">
            <div class="logo">🩺</div><h2>Doctor Sign In</h2>
            <p class="sub">Restricted to authorized clinicians</p></div></div>""",
            unsafe_allow_html=True,
        )
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="doctor", autocomplete="username")
            password = st.text_input("Password", type="password", placeholder="••••••••", autocomplete="current-password")
            submit = st.form_submit_button("Sign in →", type="primary", use_container_width=True)
        if submit:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                ok, msg = do_login(username, password)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        st.markdown(
            f"""<div class="hint-box">🔑 <b>Demo credentials</b><br>
            Username: <code>{LOCAL_USER}</code><br>Password: <code>{LOCAL_PASS}</code><br>
            <span style="opacity:.75">Change them in <code>.env</code> via
            <code>DOCTOR_USERNAME</code> / <code>DOCTOR_PASSWORD</code>.</span></div>""",
            unsafe_allow_html=True,
        )
    st.markdown('<div class="footer">ClinVision AI · educational prototype · not a medical device</div>', unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# AUTHENTICATED APP
# -----------------------------------------------------------------------------
health = api_health()

top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(
        """<div class="hero"><div class="badge">🩻 CLINVISION AI · DOCTOR PORTAL</div>
        <h1>Chest X-ray Copilot</h1><p>Upload a radiograph, review the model output, Grad-CAM and clinical summary.</p></div>""",
        unsafe_allow_html=True,
    )
with top_r:
    st.markdown(f"""<div class="card" style="text-align:center">
    <div style="font-size:2rem">👨‍⚕️</div>
    <div style="font-weight:700;color:#0b3d5f">Dr. {st.session_state.username or 'doctor'}</div>
    <div style="font-size:.78rem;color:#5b7a93;margin-bottom:8px">Authenticated session</div></div>""", unsafe_allow_html=True)
    if st.button("Log out", use_container_width=True):
        do_logout()
    if health and health.get("model_loaded"):
        st.markdown('<span class="pill pill-ok">● Model ready</span>', unsafe_allow_html=True)
    elif health:
        st.markdown('<span class="pill pill-warn">● API degraded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-bad">● API offline</span>', unsafe_allow_html=True)

st.markdown(
    """<div class="disclaimer">⚠️ <b>AI assistance only.</b> Prediction + Grad-CAM are model outputs,
    not a diagnosis. Confirm with clinical findings and a qualified radiologist.</div>""",
    unsafe_allow_html=True,
)

# Sidebar — clinical context
with st.sidebar:
    st.markdown("### 📋 Clinical context")
    st.caption("Context is shown in the summary only — it never changes the vision model.")
    age = st.number_input("Age", min_value=0, max_value=120, value=0,
                          help="0 = not specified")
    sex = st.selectbox("Sex", ["Not specified", "Female", "Male", "Other"])
    symptoms = st.text_area("Symptoms", placeholder="e.g. cough, fever, dyspnea", height=70)
    context = st.text_area("Additional context", placeholder="History, onset, comorbidities…", height=70)
    st.divider()
    st.markdown("### ⚙️ Session")
    st.caption(f"API: `{API_URL}` · limit `{MAX_UPLOAD_MB} MB`")
    if health and health.get("llm"):
        llm = health["llm"]
        active = [k for k, v in llm.items() if isinstance(v, dict) and v.get("configured")]
        st.caption("Free LLM: " + (", ".join(active) if active else "template fallback"))
    if st.button("🧹 Clear result", use_container_width=True):
        st.session_state.result = None
        st.rerun()

tab_analyze, tab_history, tab_about = st.tabs(["🔍 Analyze", "📚 History", "ℹ️ About & LLM setup"])

with tab_analyze:
    st.markdown('<div class="card"><h3>1 · Upload chest X-ray</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader("DICOM, JPG or PNG", type=["dcm", "jpg", "jpeg", "png"],
                                help=f"Max {MAX_UPLOAD_MB} MB. DICOM gets modality/VOI LUT preprocessing.")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        prev_l, prev_r = st.columns([1, 1])
        with prev_l:
            st.markdown('<div class="card"><h3>Input preview</h3>', unsafe_allow_html=True)
            if uploaded.name.lower().endswith(".dcm"):
                st.info("📄 DICOM received — the API will apply DICOM-aware preprocessing (modality + VOI LUT, MONOCHROME1 handling).")
            else:
                try:
                    st.image(Image.open(io.BytesIO(uploaded.getvalue())),
                             caption=uploaded.name, use_container_width=True)
                except Exception:
                    st.warning("Preview unavailable — the API will still try to decode the file.")
            st.markdown("</div>", unsafe_allow_html=True)
        with prev_r:
            st.markdown('<div class="card"><h3>2 · Run analysis</h3>', unsafe_allow_html=True)
            st.caption("Sends the image + clinical context to `POST /analyze` (prediction + Grad-CAM + summary in one call).")
            run = st.button("🚀 Analyze image", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if run:
            files = {"file": (uploaded.name, uploaded.getvalue(),
                              uploaded.type or "application/octet-stream")}
            payload = {
                "age": "" if age == 0 else str(age),
                "sex": "" if sex == "Not specified" else sex,
                "symptoms": symptoms.strip(),
                "context": context.strip(),
            }
            try:
                with st.spinner("Running ClinVision AI (inference + Grad-CAM + summary)…"):
                    resp = requests.post(f"{API_URL}/analyze", files=files,
                                         data=payload, headers=api_headers(), timeout=180)
                if resp.status_code == 401:
                    st.error("Session expired — please log in again.")
                    do_logout()
                resp.raise_for_status()
                result = resp.json()
                st.session_state.result = result
                st.session_state.result_filename = uploaded.name
                st.session_state.session_history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "file": uploaded.name,
                    "prediction": result.get("prediction"),
                    "probability": result.get("probability"),
                })
            except requests.RequestException as exc:
                st.error(f"FastAPI request failed: {exc}")
            except ValueError as exc:
                st.error(f"Invalid API response: {exc}")

    result = st.session_state.get("result")
    if result:
        prob = float(result.get("probability", 0))
        pred = result.get("prediction", "—")
        is_pos = pred.lower().startswith("pneumonia") and "no" not in pred.lower()
        st.markdown('<div class="card"><h3>3 · Model result</h3>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="metric-card {"positive" if is_pos else "negative"}"><div class="label">Prediction</div><div class="value">{"🟥 " if is_pos else "🟩 "}{pred}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="label">Probability</div><div class="value">{prob:.1%}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="label">Inference</div><div class="value">{result.get("inference_ms", "—")} ms</div></div>', unsafe_allow_html=True)
        st.plotly_chart(gauge(prob), use_container_width=True)
        st.caption(f"Request `{result.get('request_id')}` · model `{result.get('model_version')}` · threshold `{result.get('threshold', 0.5)}`")
        st.markdown("</div>", unsafe_allow_html=True)

        g_col, s_col = st.columns(2)
        with g_col:
            st.markdown('<div class="card"><h3>🔥 Grad-CAM attention</h3>', unsafe_allow_html=True)
            enc = result.get("gradcam_png_base64")
            if enc:
                raw = base64.b64decode(enc)
                st.image(raw, caption="Model attention overlay", use_container_width=True)
                st.download_button("⬇️ Download Grad-CAM (PNG)", data=raw,
                                   file_name="gradcam.png", mime="image/png",
                                   use_container_width=True)
                st.caption("Warm regions influenced the model most. Not a validated lesion localizer.")
            else:
                st.info(result.get("gradcam_error", "Grad-CAM was not returned by the API."))
            st.markdown("</div>", unsafe_allow_html=True)
        with s_col:
            st.markdown('<div class="card"><h3>📝 Clinical summary</h3>', unsafe_allow_html=True)
            summary = result.get("clinical_summary", {}) or {}
            if summary.get("llm_enabled"):
                st.markdown(f'<span class="pill pill-llm">✨ Free LLM · {summary.get("llm_provider", "llm")}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill pill-warn">📄 Template summary (no LLM)</span>', unsafe_allow_html=True)
            st.write(summary.get("summary", "No summary available."))
            with st.expander("Structured details"):
                st.json({
                    "assessment": summary.get("assessment"),
                    "model_probability": summary.get("model_probability"),
                    "llm_enabled": summary.get("llm_enabled", False),
                    "llm_provider": summary.get("llm_provider", "template"),
                    "clinical_context": summary.get("clinical_context", {}),
                })
            with st.expander("Limitations"):
                for item in summary.get("limitations", []):
                    st.write(f"• {item}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>📥 Export report</h3>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        fname = st.session_state.get("result_filename") or "image"
        d1.download_button("⬇️ Report (.txt)", data=build_txt_report(result, fname),
                           file_name="clinvision_report.txt", mime="text/plain",
                           use_container_width=True)
        d2.download_button("⬇️ Full result (.json)", data=json.dumps(result, indent=2),
                           file_name="clinvision_result.json", mime="application/json",
                           use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    elif not uploaded:
        st.info("👆 Upload a DICOM, JPG or PNG chest X-ray to begin.")

with tab_history:
    st.markdown('<div class="card"><h3>Session history (this login)</h3>', unsafe_allow_html=True)
    if st.session_state.session_history:
        st.dataframe(st.session_state.session_history, use_container_width=True, hide_index=True)
    else:
        st.caption("No analyses yet in this session.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Server history (shared demo SQLite)</h3>', unsafe_allow_html=True)
    try:
        r = requests.get(f"{API_URL}/history?limit=50", headers=api_headers(), timeout=10)
        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                st.dataframe(items, use_container_width=True, hide_index=True)
            else:
                st.caption("Server history is empty.")
        else:
            st.caption(f"History unavailable (HTTP {r.status_code}).")
    except Exception as exc:
        st.caption(f"History unavailable: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_about:
    st.markdown("""<div class="card"><h3>How it works</h3>
    <span style="font-size:.9rem;color:#43617a">
    1️⃣ DICOM-aware preprocessing (modality/VOI LUT, MONOCHROME1, percentile normalisation) →
    2️⃣ DenseNet121 classifier (320×320) → 3️⃣ Grad-CAM from the last dense block →
    4️⃣ Structured summary, optionally rewritten by a <b>free</b> LLM.</span></div>""",
                unsafe_allow_html=True)
    st.markdown("""<div class="card"><h3>🆓 Free LLM options (no paid key needed)</h3>
    <span style="font-size:.9rem;color:#43617a">
    <b>Recommended — Ollama (local, fully free):</b><br>
    <code>brew install ollama && ollama serve && ollama pull llama3.2</code><br>
    then set <code>OLLAMA_BASE_URL=http://localhost:11434</code> and <code>OLLAMA_MODEL=llama3.2</code>.<br><br>
    <b>Alternatives (free tiers):</b> Groq (<code>GROQ_API_KEY</code>),
    Gemini (<code>GEMINI_API_KEY</code>), Hugging Face (<code>HF_API_KEY</code>).<br>
    Without any of these the API safely falls back to the deterministic template summary.</span></div>""",
                unsafe_allow_html=True)

st.markdown('<div class="footer">ClinVision AI — AI-assisted prototype · not a medical device · © 2026</div>', unsafe_allow_html=True)
