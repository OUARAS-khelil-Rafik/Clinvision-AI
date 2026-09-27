"""
ClinVision AI — Streamlit frontend (doctor portal).

Self-contained: talks to FastAPI over HTTP, gates the UI behind a doctor
login, and renders a clinical UX (stepper, gauge, Grad-CAM, summary,
history, report downloads).

UI notes
--------
* Cards are real ``st.container(border=True)`` blocks styled via the
  ``stVerticalBlockBorderWrapper`` test-id — never fake ``<div>`` wrappers
  opened in one ``st.markdown`` call and closed in another (Streamlit renders
  each markdown in its own element, so that pattern produces broken DOM).
* Button CSS targets both the current (``stBaseButton-*``) and legacy
  (``stButton``) test-ids plus a ``kind="primary"`` fallback, so styling
  survives Streamlit upgrades.
* ``/health`` and ``/history`` are cached to avoid a request per rerun.
"""
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime

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
# Theme — lean CSS on top of native Streamlit components
# -----------------------------------------------------------------------------
THEME_CSS = """
<style>
html, body, [class*="st-"] {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: linear-gradient(180deg, #f4f8fc 0%, #e9f1f8 100%); }
.block-container { padding-top: 1.2rem; max-width: 1180px; }

/* Hero banner */
.hero {
  background: linear-gradient(120deg, #0b3d5f 0%, #0e6ba8 55%, #12a5af 100%);
  color: #fff; border-radius: 18px; padding: 24px 28px;
  box-shadow: 0 12px 32px rgba(11,61,95,.25); margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -.02em; }
.hero p { margin: 6px 0 0; opacity: .93; font-size: .95rem; }
.hero .badge {
  display: inline-block; background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.35);
  padding: 3px 12px; border-radius: 999px;
  font-size: .72rem; font-weight: 700; letter-spacing: .06em; margin-bottom: 10px;
}

/* Real bordered containers -> card look */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 16px; border-color: #dfeaf3;
  box-shadow: 0 6px 22px rgba(15,45,70,.07);
}
div[data-testid="stVerticalBlockBorderWrapper"] h3 {
  color: #0b3d5f; font-size: 1.02rem; margin-bottom: .4rem;
}

/* Stepper */
.stepper { display: flex; gap: 8px; margin: .2rem 0 1rem; flex-wrap: wrap; }
.step {
  flex: 1 1 0; min-width: 150px; text-align: center; font-size: .8rem; font-weight: 700;
  color: #5b7a93; background: #fff; border: 1px solid #dfeaf3;
  border-radius: 12px; padding: 8px 6px;
}
.step.active { color: #fff; background: linear-gradient(120deg, #0e6ba8, #12a5af); border: none; }
.step.done { color: #1e8449; border-color: #bfe6cd; background: #eef9f1; }

/* Buttons — version-proof selectors */
div[data-testid="stButton"] > button,
div[data-testid="stBaseButton-secondary"] > button,
div[data-testid="stFormSubmitButton"] > button,
button[kind="secondary"] { border-radius: 12px; font-weight: 600; }
div[data-testid="stBaseButton-primary"] > button,
button[kind="primary"] {
  background: linear-gradient(120deg, #0e6ba8, #12a5af);
  color: #fff; border: none; border-radius: 12px; font-weight: 700;
  box-shadow: 0 6px 16px rgba(14,107,168,.3);
}

/* Status pills */
.pill {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: .75rem; font-weight: 700; margin: 2px 4px 2px 0;
}
.pill-ok { background: #e6f6ec; color: #1e8449; border: 1px solid #bfe6cd; }
.pill-warn { background: #fdf1e3; color: #9a6410; border: 1px solid #f5d9a8; }
.pill-bad { background: #fdecea; color: #c0392b; border: 1px solid #f5c6c0; }
.pill-llm { background: #eef2ff; color: #3b5bdb; border: 1px solid #cdd8ff; }
.pill-risk-high { background: #fdecea; color: #c0392b; border: 1px solid #f5c6c0; }
.pill-risk-mid { background: #fdf1e3; color: #9a6410; border: 1px solid #f5d9a8; }
.pill-risk-low { background: #e6f6ec; color: #1e8449; border: 1px solid #bfe6cd; }

.disclaimer {
  background: #fff8e6; border: 1px solid #f0d48a; border-radius: 12px;
  padding: 10px 16px; font-size: .85rem; color: #7a5b00; margin: .6rem 0 1rem;
}
.hint-box {
  background: #f0f7fd; border: 1px dashed #9cc3e3; border-radius: 10px;
  padding: 10px 12px; font-size: .82rem; color: #0b3d5f; margin-top: 12px;
}
.footer { text-align: center; color: #7d99b0; font-size: .78rem; margin: 22px 0 8px; }
.login-logo {
  width: 60px; height: 60px; border-radius: 18px; margin: 0 auto 10px;
  background: linear-gradient(135deg, #0b3d5f, #12a5af);
  display: flex; align-items: center; justify-content: center; font-size: 28px;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for _key, _default in {
    "authenticated": False,
    "token": None,
    "username": None,
    "result": None,
    "result_filename": None,
    "session_history": [],
    "uploader_key": 0,  # bumped to reset the file uploader on "Clear"
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


def api_headers() -> dict:
    if st.session_state.get("token"):
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(api_url: str) -> dict | None:
    """Cached backend health probe (avoids a request on every rerun)."""
    try:
        r = requests.get(f"{api_url}/health", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_server_history(api_url: str, token: str | None, limit: int = 50) -> list | None:
    """Cached server history; None means unreachable/unauthorised."""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(f"{api_url}/history?limit={limit}", headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json().get("items", [])
        return None
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
    if fetch_health(API_URL) is None:
        return False, "API unreachable and local credentials did not match."
    return False, "Invalid username or password."


def do_logout() -> None:
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.result = None
    fetch_health.clear()
    fetch_server_history.clear()
    st.rerun()


def health_pill(health: dict | None) -> str:
    if health and health.get("model_loaded"):
        return '<span class="pill pill-ok">● API online · model loaded</span>'
    if health:
        return '<span class="pill pill-warn">● API online · model warming up</span>'
    return '<span class="pill pill-bad">● API unreachable</span>'


def risk_band(probability: float) -> tuple[str, str]:
    """Probability -> (label, pill CSS class)."""
    if probability >= 0.75:
        return "High model confidence — pneumonia likely", "pill-risk-high"
    if probability >= 0.5:
        return "Moderate — leaning pneumonia, review carefully", "pill-risk-mid"
    if probability >= 0.25:
        return "Moderate — leaning clear, review carefully", "pill-risk-mid"
    return "Low model confidence — likely clear", "pill-risk-low"


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
    s = result.get("clinical_summary", {}) or {}
    lines = [
        "ClinVision AI — Analysis Report", "=" * 34,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"File: {filename}",
        f"Model: {result.get('model_version')} ({result.get('inference_ms')} ms)",
        "",
        f"Prediction: {result.get('prediction')}",
        f"Probability: {float(result.get('probability', 0)):.2%}",
        "",
        "Clinical summary:",
        s.get("summary", "n/a"),
        "",
        "LLM: " + (f"yes ({s.get('llm_provider')})" if s.get("llm_enabled")
                    else "template fallback"),
        "",
        "Limitations:",
    ]
    lines += [f" - {x}" for x in s.get("limitations", [])]
    lines += ["", "AI assistance only — not a diagnosis. Review by a qualified physician required."]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:
    _, mid, _ = st.columns([1, 1.35, 1])
    with mid:
        st.markdown(
            """<div class="hero" style="text-align:center">
            <div class="badge">🏥 CLINICAL IMAGING COPILOT</div>
            <h1>ClinVision AI</h1>
            <p>RSNA pneumonia detection · Grad-CAM · free-LLM summary</p>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown('<div class="login-logo">🩺</div>', unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:center;margin:0'>Doctor Sign In</h3>",
                        unsafe_allow_html=True)
            st.caption("Restricted to authorized clinicians.")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="doctor",
                                         autocomplete="username")
                password = st.text_input("Password", type="password",
                                         placeholder="••••••••",
                                         autocomplete="current-password")
                submit = st.form_submit_button("Sign in →", type="primary",
                                               use_container_width=True)
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
                Username: <code>{LOCAL_USER}</code><br>
                Password: <code>{LOCAL_PASS}</code><br>
                <span style="opacity:.75">Change them in <code>.env</code> via
                <code>DOCTOR_USERNAME</code> / <code>DOCTOR_PASSWORD</code>.</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(health_pill(fetch_health(API_URL)), unsafe_allow_html=True)
        st.markdown(
            """<div class="disclaimer">🛡️ <b>Safety first.</b> Predictions and Grad-CAM
            are model outputs — never a standalone diagnosis.</div>""",
            unsafe_allow_html=True,
        )
    st.markdown('<div class="footer">ClinVision AI · educational prototype · not a medical device</div>',
                unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------------------------
# AUTHENTICATED APP
# -----------------------------------------------------------------------------
health = fetch_health(API_URL)

st.markdown(
    """<div class="hero"><div class="badge">🩻 CLINVISION AI · DOCTOR PORTAL</div>
    <h1>Chest X-ray Copilot</h1>
    <p>Upload a radiograph, review the model output, Grad-CAM and clinical summary.</p></div>""",
    unsafe_allow_html=True,
)

acct_l, acct_r = st.columns([3, 1])
with acct_l:
    st.markdown(health_pill(health), unsafe_allow_html=True)
    if health and health.get("model_version"):
        st.caption(f"Model `{health['model_version']}` · API `{API_URL}`")
with acct_r:
    with st.container(border=True):
        st.markdown(f"👨‍⚕️ **Dr. {st.session_state.username or 'doctor'}**")
        st.caption("Authenticated session")
        if st.button("Log out", use_container_width=True):
            do_logout()

st.markdown(
    """<div class="disclaimer">⚠️ <b>AI assistance only.</b> Prediction + Grad-CAM are
    model outputs, not a diagnosis. Confirm with clinical findings and a qualified
    radiologist.</div>""",
    unsafe_allow_html=True,
)

# Sidebar — clinical context
with st.sidebar:
    st.header("📋 Clinical context")
    st.caption("Context appears in the summary only — it never changes the vision model.")
    age_known = st.checkbox("Patient age known", value=False)
    age = st.number_input("Age", min_value=0, max_value=120, value=30,
                          disabled=not age_known)
    sex = st.selectbox("Sex", ["Not specified", "Female", "Male", "Other"])
    symptoms = st.text_area("Symptoms", placeholder="e.g. cough, fever, dyspnea",
                            height=70, max_chars=500)
    context = st.text_area("Additional context",
                           placeholder="History, onset, comorbidities…",
                           height=70, max_chars=500)
    st.divider()
    st.subheader("⚙️ Session")
    st.caption(f"API `{API_URL}` · limit `{MAX_UPLOAD_MB} MB`")
    if health and health.get("llm"):
        active = [k for k, v in health["llm"].items()
                  if isinstance(v, dict) and v.get("configured")]
        st.caption("Free LLM: " + (", ".join(active) if active else "template fallback"))
    c_refresh, c_clear = st.columns(2)
    with c_refresh:
        if st.button("↻ Status", use_container_width=True,
                     help="Refresh backend health probe"):
            fetch_health.clear()
            fetch_server_history.clear()
            st.rerun()
    with c_clear:
        if st.button("🧹 Clear", use_container_width=True,
                     help="Clear result and uploaded file"):
            st.session_state.result = None
            st.session_state.uploader_key += 1
            st.rerun()

tab_analyze, tab_history, tab_about = st.tabs(
    ["🔍 Analyze", "📚 History", "ℹ️ About & LLM setup"])

with tab_analyze:
    result = st.session_state.get("result")
    step = 3 if result else 1
    step_cls = (["active", "", ""] if step == 1 else ["done", "active", ""])
    if result:
        step_cls = ["done", "done", "active"]
    st.markdown(
        f"""<div class="stepper">
        <div class="step {step_cls[0]}">1 · Upload</div>
        <div class="step {step_cls[1]}">2 · Analyze</div>
        <div class="step {step_cls[2]}">3 · Review</div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("### 1 · Upload chest X-ray")
        uploaded = st.file_uploader(
            "DICOM, JPG or PNG",
            type=["dcm", "jpg", "jpeg", "png"],
            key=f"cxr_{st.session_state.uploader_key}",
            help=f"Max {MAX_UPLOAD_MB} MB. DICOM gets modality/VOI LUT preprocessing.")

    if uploaded:
        prev_l, prev_r = st.columns(2)
        with prev_l:
            with st.container(border=True):
                st.markdown("### Input preview")
                st.caption(f"📎 `{uploaded.name}` · {len(uploaded.getvalue()) / 1024:.0f} KB")
                if uploaded.name.lower().endswith(".dcm"):
                    st.info("📄 DICOM received — the API applies DICOM-aware "
                            "preprocessing (modality + VOI LUT, MONOCHROME1 handling).")
                else:
                    try:
                        st.image(Image.open(io.BytesIO(uploaded.getvalue())),
                                 caption=uploaded.name, use_container_width=True)
                    except Exception:
                        st.warning("Preview unavailable — the API will still try to decode the file.")
        with prev_r:
            with st.container(border=True):
                st.markdown("### 2 · Run analysis")
                st.caption("Calls `POST /analyze`: prediction + Grad-CAM + summary in one request.")
                ctx_bits = [
                    f"Age: {age if age_known else '—'}",
                    f"Sex: {sex}",
                    f"Symptoms: {'✓' if symptoms.strip() else '—'}",
                ]
                st.caption(" · ".join(ctx_bits))
                run = st.button("🚀 Analyze image", type="primary",
                                use_container_width=True)

        if run:
            files = {"file": (uploaded.name, uploaded.getvalue(),
                              uploaded.type or "application/octet-stream")}
            payload = {
                "age": str(age) if age_known else "",
                "sex": "" if sex == "Not specified" else sex,
                "symptoms": symptoms.strip(),
                "context": context.strip(),
            }
            try:
                with st.status("Running ClinVision AI…", expanded=False) as s:
                    s.write("Uploading image…")
                    resp = requests.post(f"{API_URL}/analyze", files=files,
                                         data=payload, headers=api_headers(), timeout=180)
                    if resp.status_code == 401:
                        s.update(label="Session expired", state="error")
                        st.error("Session expired — please log in again.")
                        do_logout()
                        st.stop()
                    resp.raise_for_status()
                    result = resp.json()
                    s.write("Grad-CAM + summary ready.")
                    s.update(label="Analysis complete", state="complete")
                st.session_state.result = result
                st.session_state.result_filename = uploaded.name
                st.session_state.session_history.insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "file": uploaded.name,
                    "prediction": result.get("prediction"),
                    "probability": result.get("probability"),
                })
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"FastAPI request failed: {exc}")
            except ValueError as exc:
                st.error(f"Invalid API response: {exc}")

    result = st.session_state.get("result")
    if result:
        prob = float(result.get("probability", 0))
        pred = result.get("prediction", "—")
        band_label, band_cls = risk_band(prob)
        with st.container(border=True):
            st.markdown("### 3 · Model result")
            st.markdown(f'<span class="pill {band_cls}">{band_label}</span>',
                        unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", pred)
            m2.metric("Pneumonia probability", f"{prob:.1%}")
            m3.metric("Inference time", f"{result.get('inference_ms', '—')} ms")
            st.plotly_chart(gauge(prob), use_container_width=True)
            st.caption(f"Request `{result.get('request_id')}` · "
                       f"model `{result.get('model_version')}` · "
                       f"threshold `{result.get('threshold', 0.5)}`")

        g_col, s_col = st.columns(2)
        with g_col:
            with st.container(border=True):
                st.markdown("### 🔥 Grad-CAM attention")
                enc = result.get("gradcam_png_base64")
                if enc:
                    raw = base64.b64decode(enc)
                    st.image(raw, caption="Model attention overlay",
                             use_container_width=True)
                    st.download_button("⬇️ Download Grad-CAM (PNG)", data=raw,
                                       file_name="gradcam.png", mime="image/png",
                                       use_container_width=True)
                    st.caption("Warm regions influenced the model most. "
                               "Not a validated lesion localizer.")
                else:
                    st.info(result.get("gradcam_error",
                                       "Grad-CAM was not returned by the API."))
        with s_col:
            with st.container(border=True):
                st.markdown("### 📝 Clinical summary")
                summary = result.get("clinical_summary", {}) or {}
                if summary.get("llm_enabled"):
                    st.markdown(f'<span class="pill pill-llm">✨ Free LLM · '
                                f'{summary.get("llm_provider", "llm")}</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<span class="pill pill-warn">📄 Template summary '
                                '(no LLM)</span>', unsafe_allow_html=True)
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

        with st.container(border=True):
            st.markdown("### 📥 Export report")
            d1, d2 = st.columns(2)
            fname = st.session_state.get("result_filename") or "image"
            d1.download_button("⬇️ Report (.txt)",
                               data=build_txt_report(result, fname),
                               file_name="clinvision_report.txt",
                               mime="text/plain", use_container_width=True)
            d2.download_button("⬇️ Full result (.json)",
                               data=json.dumps(result, indent=2),
                               file_name="clinvision_result.json",
                               mime="application/json", use_container_width=True)
    elif not uploaded:
        with st.container(border=True):
            st.markdown("### 👋 How to start")
            st.markdown("1. Upload a **DICOM, JPG or PNG** chest X-ray above.\n"
                        "2. (Optional) Fill in clinical context in the sidebar.\n"
                        "3. Press **Analyze image** and review prediction, Grad-CAM and summary.")

with tab_history:
    with st.container(border=True):
        st.markdown("### Session history (this login)")
        if st.session_state.session_history:
            st.dataframe(st.session_state.session_history,
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No analyses yet in this session.")

    with st.container(border=True):
        st.markdown("### Server history (shared demo SQLite)")
        st.caption("Stored without patient identifiers — request id, class and probability only.")
        h1, h2 = st.columns(2)
        with h1:
            if st.button("↻ Refresh history", use_container_width=True):
                fetch_server_history.clear()
                st.rerun()
        with h2:
            confirm = st.checkbox("Confirm clear", value=False,
                                  help="Tick to enable the delete button")
            if st.button("🗑️ Clear server history", use_container_width=True,
                         disabled=not confirm):
                try:
                    r = requests.delete(f"{API_URL}/history",
                                        headers=api_headers(), timeout=10)
                    if r.status_code == 200:
                        fetch_server_history.clear()
                        st.success("Server history cleared.")
                        st.rerun()
                    elif r.status_code == 401:
                        st.error("Session expired — please log in again.")
                        do_logout()
                    else:
                        st.error(f"Clear failed (HTTP {r.status_code}).")
                except requests.RequestException as exc:
                    st.error(f"Clear failed: {exc}")
        items = fetch_server_history(API_URL, st.session_state.get("token"))
        if items:
            st.dataframe(items, use_container_width=True, hide_index=True)
        elif items == []:
            st.caption("Server history is empty.")
        else:
            st.caption("History unavailable (API offline or unauthorized).")

with tab_about:
    with st.container(border=True):
        st.markdown("### How it works")
        st.markdown("1. **DICOM-aware preprocessing** — modality/VOI LUT, MONOCHROME1 "
                    "handling, percentile normalisation, resize to 320×320.\n"
                    "2. **DenseNet121 classifier** (ImageNet transfer learning) → "
                    "pneumonia probability.\n"
                    "3. **Grad-CAM** from the last dense block visualises influential "
                    "regions (explanatory only).\n"
                    "4. **Structured summary**, optionally rewritten by a **free** LLM. "
                    "Clinical context never overrides the image model.")
    with st.container(border=True):
        st.markdown("### 🆓 Free LLM options (no paid key needed)")
        st.code("brew install ollama && ollama serve && ollama pull llama3.2",
                language="bash")
        st.caption("Then set `OLLAMA_BASE_URL=http://localhost:11434` and "
                   "`OLLAMA_MODEL=llama3.2`. Alternatives (free tiers): "
                   "Groq (`GROQ_API_KEY`), Gemini (`GEMINI_API_KEY`), "
                   "Hugging Face (`HF_API_KEY`). Without any of these the API safely "
                   "falls back to the deterministic template summary.")

st.markdown('<div class="footer">ClinVision AI — AI-assisted prototype · not a medical device · © 2026</div>',
            unsafe_allow_html=True)
