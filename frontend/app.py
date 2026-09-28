"""
ClinVision AI - Streamlit frontend (doctor portal).

Self-contained: talks to FastAPI over HTTP, gates the UI behind a doctor
login, and renders a clinical UX (stepper, gauge, Original-vs-Grad-CAM
visual evidence, summary, history, report downloads).

UI notes
--------
* Single icon system: every icon comes from the ``ICONS`` registry and is
  rendered as a Material Symbol via ``:material/<name>:`` shortcodes in
  native Streamlit elements (markdown, headers, buttons, alerts, captions).
  ``ic()`` falls back to a safe icon so a typo can never leak a literal
  ``:material/...:`` string. Shortcodes do NOT expand inside raw-HTML blocks,
  Plotly figures, metric labels or tab labels, so those stay text-only and
  the login logo is an inline SVG cross.
* Light/dark mode: Light or Dark (no Auto). The choice is mirrored into
  Streamlit's own theme engine via ``set_option("theme.base", …)`` so native
  widgets follow, and our CSS palette is forced to match. The Plotly gauge
  receives matching colors explicitly.
* Cards keep NATIVE bordered-container styling (the
  ``stVerticalBlockBorderWrapper`` testid no longer exists in Streamlit
  1.64+, so it must not be targeted) - native containers already adapt to
  light/dark. Custom CSS covers only our own classes (hero, stepper, pills,
  disclaimer) plus version-proof button selectors (runtime testids are
  ``stBaseButton-<kind>``; form submits use ``stFormSubmitButton``).
* Font is applied by inheritance (``.stApp``) and inside markdown containers
  only - never via ``[class*="st-"]``, which would override the
  Material-Symbol ligature font of Streamlit's internal icons and leak
  stray text such as "visibili" in the password field.
* ``/health`` and ``/history`` are cached to avoid a request per rerun.
* The original image is decoded locally (DICOM-aware via pydicom) so the
  Visual-evidence section can always show Original vs Grad-CAM side by side,
  even for DICOM uploads which have no browser preview.
"""
from __future__ import annotations

import base64
import hashlib
import io
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

# -----------------------------------------------------------------------------
# Icon system - single source of truth (Material Symbols, Streamlit-native).
# Only long-established icon names are used so they always resolve.
# -----------------------------------------------------------------------------
ICONS = {
    "brand": "local_hospital",
    "login": "login",
    "logout": "logout",
    "user": "person",
    "key": "key",
    "context": "assignment",
    "settings": "settings",
    "theme": "contrast",
    "upload": "upload_file",
    "attach": "attach_file",
    "image": "image",
    "compare": "compare_arrows",
    "layers": "layers",
    "run": "biotech",
    "go": "play_arrow",
    "check": "check",
    "result": "analytics",
    "gradcam": "visibility",
    "summary": "description",
    "download": "download",
    "start": "play_circle",
    "how": "lightbulb",
    "history": "history",
    "server": "storage",
    "folder": "folder_shared",
    "refresh": "refresh",
    "clear": "delete_sweep",
    "delete": "delete",
    "llm": "auto_awesome",
    "info": "info",
}
_ICON_FALLBACK = "info"


def ic(name: str) -> str:
    """Material-Symbol shortcode; never leaks a literal on typo."""
    return f":material/{ICONS.get(name, _ICON_FALLBACK)}:"


st.set_page_config(
    page_title="ClinVision AI - Doctor Portal",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Theme - CSS variables, light + dark palettes
# -----------------------------------------------------------------------------
_LIGHT_VARS = """:root{
  --cv-appbg: linear-gradient(180deg, #f4f8fc 0%, #e9f1f8 100%);
  --cv-card:#ffffff; --cv-ink:#0b3d5f; --cv-muted:#5b7a93;
  --cv-border:#dfeaf3; --cv-shadow:rgba(15,45,70,.07);
  --cv-step-bg:#ffffff; --cv-step-ink:#5b7a93;
  --cv-step-done-ink:#1e8449; --cv-step-done-bd:#bfe6cd; --cv-step-done-bg:#eef9f1;
  --cv-ok-bg:#e6f6ec; --cv-ok-ink:#1e8449; --cv-ok-bd:#bfe6cd;
  --cv-mid-bg:#fdf1e3; --cv-mid-ink:#9a6410; --cv-mid-bd:#f5d9a8;
  --cv-bad-bg:#fdecea; --cv-bad-ink:#c0392b; --cv-bad-bd:#f5c6c0;
  --cv-llm-bg:#eef2ff; --cv-llm-ink:#3b5bdb; --cv-llm-bd:#cdd8ff;
  --cv-warn-bg:#fff8e6; --cv-warn-bd:#f0d48a; --cv-warn-bar:#e6a800; --cv-warn-ink:#7a5b00;
  --cv-hint-bg:#f0f7fd; --cv-hint-bd:#9cc3e3; --cv-hint-ink:#0b3d5f;
  --cv-footer:#7d99b0;
}"""
_DARK_VARS = """:root{
  --cv-appbg: linear-gradient(180deg, #0a1322 0%, #0d1929 100%);
  --cv-card:#101c2e; --cv-ink:#e8f1f9; --cv-muted:#9db4c9;
  --cv-border:#22344d; --cv-shadow:rgba(0,0,0,.45);
  --cv-step-bg:#101c2e; --cv-step-ink:#9db4c9;
  --cv-step-done-ink:#7be2a3; --cv-step-done-bd:#1f5c38; --cv-step-done-bg:#0e2318;
  --cv-ok-bg:#0e2318; --cv-ok-ink:#7be2a3; --cv-ok-bd:#1f5c38;
  --cv-mid-bg:#2a2111; --cv-mid-ink:#f0c979; --cv-mid-bd:#8a6d1f;
  --cv-bad-bg:#2c1512; --cv-bad-ink:#f2a49c; --cv-bad-bd:#7c2d24;
  --cv-llm-bg:#1a2140; --cv-llm-ink:#aebfff; --cv-llm-bd:#35448f;
  --cv-warn-bg:#2a2111; --cv-warn-bd:#8a6d1f; --cv-warn-bar:#e6a800; --cv-warn-ink:#f0d48a;
  --cv-hint-bg:#12283f; --cv-hint-bd:#2c5a86; --cv-hint-ink:#cfe6f7;
  --cv-footer:#8fa6bc;
}"""
_THEME_BODY = """
<style>
/* Font: inheritance only. Never override [class*="st-"] - that kills the
   Material-Symbol ligatures of Streamlit's internal icons (password eye,
   uploader glyph, chevrons), which render as stray text like "visibili". */
.stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
div[data-testid="stMarkdownContainer"] h3 { color: var(--cv-ink); }
.stApp { background: var(--cv-appbg); }
/* Generous top clearance so the Streamlit toolbar never overlaps the hero. */
.block-container { padding-top: 3.4rem; max-width: 1180px; }
.hero { margin-top: .4rem; }

/* Hero banner (text-only: shortcodes don't expand in raw HTML) */
.hero {
  background: linear-gradient(120deg, #0b3d5f 0%, #0e6ba8 55%, #12a5af 100%);
  color: #fff; border-radius: 18px; padding: 24px 28px;
  box-shadow: 0 12px 32px rgba(11,61,95,.25); margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -.02em; color:#fff; }
.hero p { margin: 6px 0 0; opacity: .93; font-size: .95rem; color:#fff; }
.hero .badge {
  display: inline-block; background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.35); color:#fff;
  padding: 3px 12px; border-radius: 999px;
  font-size: .72rem; font-weight: 700; letter-spacing: .06em; margin-bottom: 10px;
}

/* NOTE: st.container(border=True) keeps NATIVE styling (the
   stVerticalBlockBorderWrapper testid no longer exists in Streamlit 1.64+),
   which already adapts to light/dark. No override here on purpose. */

/* Stepper */
.stepper { display: flex; gap: 8px; margin: .2rem 0 1rem; flex-wrap: wrap; }
.step {
  flex: 1 1 0; min-width: 150px; text-align: center; font-size: .8rem; font-weight: 700;
  color: var(--cv-step-ink); background: var(--cv-step-bg);
  border: 1px solid var(--cv-border);
  border-radius: 12px; padding: 8px 6px;
}
.step.active { color: #fff; background: linear-gradient(120deg, #0e6ba8, #12a5af); border: none; }
.step.done { color: var(--cv-step-done-ink); border-color: var(--cv-step-done-bd);
  background: var(--cv-step-done-bg); }

/* Buttons - version-proof selectors.
   Runtime testids are stBaseButton-<kind>; form submits use stFormSubmitButton
   (without it the Sign-in button falls back to the default red primary). */
div[data-testid="stButton"] > button,
div[data-testid="stBaseButton-secondary"] > button,
div[data-testid="stFormSubmitButton"] > button,
button[kind="secondary"] { border-radius: 12px; font-weight: 600; }
div[data-testid="stBaseButton-primary"] > button,
div[data-testid="stFormSubmitButton"] > button[type="submit"],
div[data-testid="stFormSubmitButton"] > button,
button[kind="primary"] {
  background: linear-gradient(120deg, #0e6ba8, #12a5af);
  color: #fff; border: none; border-radius: 12px; font-weight: 700;
  box-shadow: 0 6px 16px rgba(14,107,168,.3);
}

/* Status pills (text-only: shortcodes don't expand in raw HTML).
   All colors are theme vars so forced Dark works, not just OS dark. */
.pill {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: .75rem; font-weight: 700; margin: 2px 4px 2px 0;
}
.pill-ok { background: var(--cv-ok-bg); color: var(--cv-ok-ink); border: 1px solid var(--cv-ok-bd); }
.pill-warn { background: var(--cv-mid-bg); color: var(--cv-mid-ink); border: 1px solid var(--cv-mid-bd); }
.pill-bad { background: var(--cv-bad-bg); color: var(--cv-bad-ink); border: 1px solid var(--cv-bad-bd); }
.pill-llm { background: var(--cv-llm-bg); color: var(--cv-llm-ink); border: 1px solid var(--cv-llm-bd); }
.pill-risk-high { background: var(--cv-bad-bg); color: var(--cv-bad-ink); border: 1px solid var(--cv-bad-bd); }
.pill-risk-mid { background: var(--cv-mid-bg); color: var(--cv-mid-ink); border: 1px solid var(--cv-mid-bd); }
.pill-risk-low { background: var(--cv-ok-bg); color: var(--cv-ok-ink); border: 1px solid var(--cv-ok-bd); }

.disclaimer {
  background: var(--cv-warn-bg); border: 1px solid var(--cv-warn-bd);
  border-left: 4px solid var(--cv-warn-bar);
  border-radius: 12px; padding: 10px 16px; font-size: .85rem; color: var(--cv-warn-ink);
  margin: .6rem 0 1rem;
}
.hint-box {
  background: var(--cv-hint-bg); border: 1px dashed var(--cv-hint-bd); border-radius: 10px;
  padding: 10px 12px; font-size: .82rem; color: var(--cv-hint-ink); margin-top: 12px;
}
.footer { text-align: center; color: var(--cv-footer); font-size: .78rem; margin: 22px 0 8px; }
/* Sidebar/clinical text frames: fixed size, internal scrollbar - never resizable. */
div[data-testid="stTextArea"] textarea {
  resize: none !important;
  overflow-y: auto !important;
}
div[data-testid="stTextArea"] textarea::-webkit-scrollbar { width: 8px; }
div[data-testid="stTextArea"] textarea::-webkit-scrollbar-track { background: transparent; }
div[data-testid="stTextArea"] textarea::-webkit-scrollbar-thumb {
  background: #9db4c9; border-radius: 8px;
}
div[data-testid="stTextArea"] textarea::-webkit-scrollbar-thumb:hover { background: #0e6ba8; }
.login-logo {
  width: 60px; height: 60px; border-radius: 18px; margin: 0 auto 10px;
  background: linear-gradient(135deg, #0b3d5f, #12a5af);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 8px 20px rgba(11,61,95,.3);
}
.evidence-label { font-size: .78rem; font-weight: 700; color: var(--cv-muted);
  text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
</style>
"""


def apply_theme(mode: str) -> None:
    """Inject the forced Light or Dark palette (no Auto mode).

    The palette re-declares ``:root`` so it always wins the cascade.
    """
    parts = ["<style>", _LIGHT_VARS]
    if mode == "Dark":
        parts.append(_DARK_VARS)
    else:
        parts.append(_LIGHT_VARS)
    parts.append("</style>")
    st.markdown("\n".join(parts) + _THEME_BODY, unsafe_allow_html=True)


def sync_streamlit_theme(mode: str) -> None:
    """Mirror our Light/Dark choice into Streamlit's own theme engine.

    Without this, native widgets (inputs, buttons, uploader, sidebar) keep
    whatever theme Streamlit started with, clashing with our palette.
    """
    try:
        from streamlit import _config as _cfg

        want = mode.lower()
        if _cfg.get_option("theme.base") != want:
            _cfg.set_option("theme.base", want)
            st.rerun()
    except Exception:
        pass  # private API may change; our CSS palette still applies


def theme_is_dark() -> bool:
    return st.session_state.get("theme_mode", "Light") == "Dark"


# Inline SVG medical cross (HTML blocks can't use :material: shortcodes).
LOGO_SVG = """<div class="login-logo">
<svg width="30" height="30" viewBox="0 0 24 24" fill="none"
stroke="#ffffff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 5v14M5 12h14"/></svg></div>"""

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for _key, _default in {
    "authenticated": False,
    "token": None,
    "username": None,
    "result": None,
    "result_filename": None,
    "result_bytes": None,  # exact analyzed bytes: overlay always pairs with THESE
    "session_history": [],
    "uploader_key": 0,  # bumped to reset the file uploader on "Clear"
    "theme_mode": "Light",
    "evidence_view": "Side by side",
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Pending clinical-context sync (queued when a NEW file arrives, applied here
# BEFORE the sidebar widgets instantiate — modifying widget state later in
# the run would raise). Rule: every new file first resets ALL context fields
# to defaults, then DICOM demographics refill what the file provides.
_pending = st.session_state.pop("pending_context", None)
if _pending:
    st.session_state["sb_age_known"] = False
    st.session_state["sb_age"] = 30
    st.session_state["sb_sex"] = "Not specified"
    st.session_state["sb_symptoms"] = ""
    st.session_state["sb_context"] = ""
    _meta = _pending.get("meta") or {}
    _applied = []
    if _meta.get("age") is not None:
        st.session_state["sb_age_known"] = True
        st.session_state["sb_age"] = _meta["age"]
        _applied.append(f"Age {_meta['age']}")
    if _meta.get("sex"):
        st.session_state["sb_sex"] = _meta["sex"]
        _applied.append(_meta["sex"])
    if _applied:
        st.toast(f"DICOM auto-filled: {', '.join(_applied)}")
    else:
        st.toast("Clinical context reset to defaults.")

sync_streamlit_theme(st.session_state.theme_mode)
apply_theme(st.session_state.theme_mode)


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


@st.cache_data(ttl=30, show_spinner=False)
def fetch_patients(api_url: str, token: str | None, limit: int = 100) -> list | None:
    """Patient records (metadata + LLM note, images via dedicated endpoints)."""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(f"{api_url}/patients?limit={limit}", headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json().get("items", [])
        return None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_patient_file(api_url: str, token: str | None,
                       request_id: str, kind: str) -> bytes | None:
    """Fetch archived original ('image') or overlay ('gradcam') bytes."""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(f"{api_url}/patients/{request_id}/{kind}",
                         headers=headers, timeout=30)
        if r.status_code == 200 and r.content:
            return r.content
        return None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
@st.cache_data(ttl=600, show_spinner=False)
def extract_dicom_metadata(raw: bytes, filename: str) -> dict:
    """Pull patient demographics embedded in a DICOM file, if present.

    Maps PatientAge (e.g. '058Y', also M/W/D units) to years and PatientSex
    (M/F/O) to the sidebar labels. Symptoms/context don't exist in standard
    DICOM tags and always stay manual. Returns {} when nothing usable.
    """
    if Path(filename or "").suffix.lower() != ".dcm":
        return {}
    try:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(raw), force=False, stop_before_pixels=True)
        meta: dict = {}

        def _tag(*names):
            for n in names:
                if n in ds and ds[n].value not in (None, ""):
                    return str(ds[n].value).strip()
            return ""

        raw_age = _tag("PatientAge")
        if raw_age:
            try:
                n = int("".join(c for c in raw_age if c.isdigit()) or 0)
                unit = (raw_age[-1] if raw_age[-1].isalpha() else "Y").upper()
                years = {"Y": n, "M": n // 12, "W": n // 52, "D": n // 365}.get(unit, n)
                if 0 < years <= 120:
                    meta["age"] = years
            except Exception:
                pass
        sex = _tag("PatientSex").upper()
        if sex == "M":
            meta["sex"] = "Male"
        elif sex == "F":
            meta["sex"] = "Female"
        elif sex == "O":
            meta["sex"] = "Other"
        for key, tag in (("patient_id", "PatientID"), ("study_date", "StudyDate"),
                         ("modality", "Modality")):
            val = _tag(tag)
            if val:
                meta[key] = val
        return meta
    except Exception:
        return {}


def decode_upload_for_display(raw: bytes, filename: str) -> bytes | None:
    """Render the uploaded file to displayable PNG bytes (DICOM-aware).

    Mirrors the backend's grayscale normalisation so the "Original" panel
    matches what the model saw - including DICOM, which browsers can't
    preview natively. Returns None when undecodable.

    NOTE: ``raw`` must NOT be underscore-prefixed - Streamlit excludes such
    params from the cache key, which would serve stale images across uploads.
    """
    import numpy as np

    suffix = Path(filename or "").suffix.lower()
    try:
        if suffix == ".dcm":
            import pydicom
            from pydicom.pixels import apply_modality_lut, apply_voi_lut

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
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            low, high = np.percentile(arr, [1.0, 99.0])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(arr.min()), float(arr.max())
            if high <= low:
                return None
            gray = ((np.clip((arr - low) / (high - low), 0.0, 1.0)) * 255).astype("uint8")
            img = Image.fromarray(gray, mode="L")
        else:
            img = Image.open(io.BytesIO(raw)).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _result_matches_upload(result_bytes: bytes | None) -> bool:
    """True when the displayed result was computed from the current upload."""
    up = st.session_state.get(f"cxr_{st.session_state.uploader_key}")
    if result_bytes is None or up is None:
        return result_bytes is None and up is None
    try:
        return up.getvalue() == result_bytes
    except Exception:
        return False


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
    # 3) Offline fallback - API unreachable, check local demo credentials
    if username == LOCAL_USER and password == LOCAL_PASS:
        st.session_state.authenticated = True
        st.session_state.token = None
        st.session_state.username = username
        return True, "API unreachable - signed in offline (demo mode)."
    if fetch_health(API_URL) is None:
        return False, "API unreachable and local credentials did not match."
    return False, "Invalid username or password."


def do_logout() -> None:
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.result = None
    st.session_state.result_bytes = None
    fetch_health.clear()
    fetch_server_history.clear()
    fetch_patients.clear()
    fetch_patient_file.clear()
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
        return "High model confidence - pneumonia likely", "pill-risk-high"
    if probability >= 0.5:
        return "Moderate - leaning pneumonia, review carefully", "pill-risk-mid"
    if probability >= 0.25:
        return "Moderate - leaning clear, review carefully", "pill-risk-mid"
    return "Low model confidence - likely clear", "pill-risk-low"


def gauge(probability: float, dark: bool = False):
    ink = "#e8f1f9" if dark else "#0b3d5f"
    muted = "#9db4c9" if dark else "#5b7a93"
    track = "#22344d" if dark else "#eef4fa"
    lo, hi = ("#0e2318", "#2c1512") if dark else ("#e6f6ec", "#fdecea")
    color = "#f2a49c" if (dark and probability >= 0.5) else ("#7be2a3" if dark else ("#c0392b" if probability >= 0.5 else "#1e8449"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "font": {"size": 30, "color": ink}},
        gauge={"axis": {"range": [0, 100], "tickfont": {"color": muted}},
               "bar": {"color": color},
               "bgcolor": track,
               "steps": [{"range": [0, 50], "color": lo},
                         {"range": [50, 100], "color": hi}],
               "threshold": {"line": {"color": ink, "width": 3}, "value": 50}},
        title={"text": "Pneumonia probability", "font": {"size": 14, "color": muted}},
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_family="Inter")
    return fig


def _latin(text: object) -> str:
    """Sanitize to latin-1 (fpdf2 core fonts); never crash on LLM unicode."""
    s = str(text or "")
    for a, b in {"\u2014": "-", "–": "-", "·": "-", "✓": "x", "…": "...",
                 "→": "->", "←": "<-", """: '"', """: '"',
                 "''": "'", "✔": "x", "⚠": "!", "•": "-"}.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(result: dict, filename: str,
                     original_png: bytes | None,
                     gradcam_png: bytes | None) -> bytes:
    """Colored, organized PDF report: context, result, evidence, LLM summary."""
    from fpdf import FPDF

    TEAL, NAVY, GRAY = (14, 107, 168), (11, 61, 95), (70, 90, 110)
    prob = float(result.get("probability", 0))
    pred = str(result.get("prediction", "N/A"))
    band = ("High model confidence - pneumonia likely", (192, 57, 43)) \
        if prob >= 0.75 else \
        (("Moderate - leaning pneumonia", (176, 121, 14)) if prob >= 0.5 else
         (("Moderate - leaning clear", (176, 121, 14)) if prob >= 0.25 else
          ("Low model confidence - likely clear", (30, 132, 73))))
    summary = result.get("clinical_summary", {}) or {}
    ctx = summary.get("clinical_context", {}) or {}

    class Report(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GRAY)
            self.cell(0, 10, _latin(
                "AI assistance only - not a diagnosis. "
                "Review by a qualified physician required.  "
                f"Page {self.page_no()}/{{nb}}"), align="C")

    pdf = Report()
    pdf.alias_nb_pages("{nb}")
    pdf.set_auto_page_break(True, 20)
    pdf.add_page()

    # Header band
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, 210, 34, "F")
    pdf.set_y(6)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "ClinVision AI - Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _latin(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}  |  File: {filename}"), align="C")

    # Risk band
    pdf.ln(8)
    pdf.set_fill_color(*band[1])
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, _latin(f"  {pred}  -  {prob:.1%}  -  {band[0]}"),
             fill=True, new_x="LMARGIN", new_y="NEXT")

    def section(title: str):
        pdf.ln(4)
        pdf.set_text_color(*NAVY)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _latin(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*TEAL)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(2)

    def body(text: str):
        pdf.set_text_color(*GRAY)
        pdf.set_font("Helvetica", "", 10)
        # fpdf2>=2.7 leaves the cursor at end-of-line after multi_cell;
        # force back to the left margin for consecutive paragraphs.
        pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")

    def kv(key: str, value: object):
        pdf.set_text_color(*NAVY)
        pdf.set_font("Helvetica", "B", 10)
        pdf.write(5.5, _latin(key + ": "))
        pdf.set_text_color(*GRAY)
        pdf.set_font("Helvetica", "", 10)
        pdf.write(5.5, _latin(value))
        pdf.ln(6.5)

    def subsection(title: str):
        pdf.set_text_color(*TEAL)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _latin(title), new_x="LMARGIN", new_y="NEXT")

    def split_sections(text: str) -> list[tuple[str, str]]:
        """Mirror of backend.split_llm_sections: '### ' headings → parts."""
        parts: list[tuple[str, str]] = []
        cur_t: str | None = None
        cur_l: list[str] = []
        for line in (text or "").splitlines():
            s = line.strip()
            if s.startswith("### "):
                if cur_t is not None:
                    parts.append((cur_t, "\n".join(cur_l).strip()))
                cur_t = s[4:].strip(" #-*") or "Section"
                cur_l = []
            elif cur_t is not None:
                cur_l.append(line)
        if cur_t is not None:
            parts.append((cur_t, "\n".join(cur_l).strip()))
        return [(t, b) for t, b in parts if b]

    def evidence_height(png: bytes | None, width_mm: float) -> float:
        try:
            with Image.open(io.BytesIO(png)) as im:
                w, h = im.size
            return width_mm * h / max(w, 1)
        except Exception:
            return width_mm  # square fallback

    section("1. Patient context")
    dicom_meta = result.get("dicom_metadata") or {}
    if dicom_meta.get("patient_id"):
        kv("Patient ID", dicom_meta["patient_id"])
    if dicom_meta.get("study_date"):
        kv("Study date", dicom_meta["study_date"])
    if dicom_meta.get("modality"):
        kv("Modality", dicom_meta["modality"])
    kv("Age", ctx.get("age") or "Not specified")
    kv("Sex", ctx.get("sex") or "Not specified")
    kv("Symptoms", ctx.get("symptoms") or "N/A")
    kv("Additional context", ctx.get("context") or "N/A")

    section("2. Model result")
    kv("Prediction", pred)
    kv("Pneumonia probability", f"{prob:.2%}")
    kv("Inference time", f"{result.get('inference_ms', 'N/A')} ms")
    kv("Model", f"{result.get('model_version')} (threshold {result.get('threshold', 0.5)})")
    kv("Request ID", result.get("request_id"))

    section("3. Visual evidence (Original vs Grad-CAM)")
    if original_png is not None or gradcam_png is not None:
        img_h = max(evidence_height(original_png, 85) if original_png else 0,
                    evidence_height(gradcam_png, 85) if gradcam_png else 0,
                    20)
        if pdf.get_y() + img_h + 14 > 277:  # keep image + captions together
            pdf.add_page()
        y0 = pdf.get_y()
        if original_png is not None:
            pdf.image(io.BytesIO(original_png), x=15, y=y0, w=85)
        if gradcam_png is not None:
            pdf.image(io.BytesIO(gradcam_png), x=110, y=y0, w=85)
        pdf.set_y(y0 + img_h + 2)
        pdf.set_text_color(*GRAY)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(85, 5, "Original radiograph", align="C")
        pdf.cell(10, 5, "")
        pdf.cell(85, 5, "Grad-CAM attention overlay", align="C",
                 new_x="LMARGIN", new_y="NEXT")
        body("Warm regions influenced the model most. Grad-CAM is explanatory, "
             "not a validated lesion localizer.")
    else:
        body("No images available for this report.")

    section("4. Clinical summary (LLM)")
    llm_line = (f"Generated with free LLM ({summary.get('llm_provider')})"
                if summary.get("llm_enabled") else "Deterministic template (no LLM)")
    pdf.set_text_color(*TEAL)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _latin(llm_line), new_x="LMARGIN", new_y="NEXT")
    parts = split_sections(summary.get("summary", ""))
    if parts:
        for title, content in parts:
            subsection(title)
            body(content)
    else:
        body(summary.get("summary", "n/a"))

    section("5. Limitations")
    for item in summary.get("limitations", []):
        body(f"- {item}")

    return bytes(pdf.output())


@st.cache_data(ttl=600, show_spinner=False)
def blend_gradcam(original_png: bytes, heatmap_png: bytes, alpha: float) -> bytes | None:
    """Legacy global re-blend (used only when the server sent no mask).

    NOTE: params must NOT be underscore-prefixed - Streamlit would exclude
    them from the cache key and serve blends from previous analyses.
    """
    import numpy as np

    try:
        orig = Image.open(io.BytesIO(original_png)).convert("RGB")
        heat = Image.open(io.BytesIO(heatmap_png)).convert("RGB").resize(
            orig.size, Image.BILINEAR)
        blended = Image.fromarray(
            (np.asarray(orig, dtype=np.float32) * (1.0 - alpha)
             + np.asarray(heat, dtype=np.float32) * alpha).astype("uint8"))
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def blend_gradcam_masked(original_png: bytes, heatmap_png: bytes,
                         mask_png: bytes, alpha: float) -> bytes | None:
    """Masked re-blend: the uploaded radiograph stays intact, heat paints
    ONLY where the model responded (per-pixel alpha = mask × intensity).

    At 0% the output is pixel-identical to the upload; raising the slider
    reveals hot regions without ever tinting cold anatomy purple.

    NOTE: params must NOT be underscore-prefixed - Streamlit would exclude
    them from the cache key and serve blends from previous analyses.
    """
    import numpy as np

    try:
        orig = Image.open(io.BytesIO(original_png)).convert("RGB")
        heat = Image.open(io.BytesIO(heatmap_png)).convert("RGB").resize(
            orig.size, Image.BILINEAR)
        mask = Image.open(io.BytesIO(mask_png)).convert("L").resize(
            orig.size, Image.BILINEAR)
        m = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None] * float(alpha)
        out = (np.asarray(orig, dtype=np.float32) * (1.0 - m)
               + np.asarray(heat, dtype=np.float32) * m).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def evidence_panel(original_png: bytes | None, gradcam_raw: bytes | None,
                   heatmap_raw: bytes | None, mask_raw: bytes | None,
                   filename: str) -> bytes | None:
    """Visual-evidence section: Original vs Grad-CAM with view + intensity.

    Returns the currently displayed overlay bytes (for PDF export/download).
    """
    view = st.segmented_control(
        "View", ["Side by side", "Overlay only", "Original only"],
        key="evidence_view",
        help="Compare the original radiograph with the model attention overlay.",
    )
    if gradcam_raw is None and original_png is None:
        st.info("No images available for comparison.")
        return None
    # Adjustable overlay anchored on the UPLOADED image: masked re-blend keeps
    # cold anatomy pixel-identical to the original; only hot regions are
    # painted. Falls back to the server overlay when no mask/heatmap exists.
    overlay_png = gradcam_raw
    if heatmap_raw is not None and original_png is not None:
        intensity = st.slider(
            f"{ic('layers')} Grad-CAM intensity", 0, 100, 55,
            help="0% = uploaded radiograph exactly, "
                 "100% = full heat on hot regions only.",
        )
        if mask_raw is not None:
            blended = blend_gradcam_masked(
                original_png, heatmap_raw, mask_raw, intensity / 100.0)
        else:
            blended = blend_gradcam(original_png, heatmap_raw, intensity / 100.0)
        if blended is not None:
            overlay_png = blended
            st.caption(f"Overlay at {intensity}% on the uploaded image - "
                       "drag to examine hot regions against the anatomy.")
    if view == "Overlay only":
        if overlay_png is not None:
            st.markdown('<div class="evidence-label">Grad-CAM overlay</div>',
                        unsafe_allow_html=True)
            st.image(overlay_png, caption="Model attention overlay",
                     use_container_width=True)
        else:
            st.info("Grad-CAM was not returned by the API.")
    elif view == "Original only":
        if original_png is not None:
            st.markdown('<div class="evidence-label">Original radiograph</div>',
                        unsafe_allow_html=True)
            st.image(original_png, caption=filename, use_container_width=True)
        else:
            st.warning("Original preview unavailable for this file.")
    else:  # Side by side
        c_orig, c_cam = st.columns(2)
        with c_orig:
            st.markdown('<div class="evidence-label">Original</div>',
                        unsafe_allow_html=True)
            if original_png is not None:
                st.image(original_png, caption=filename, use_container_width=True)
            else:
                st.warning("Original preview unavailable for this file.")
        with c_cam:
            st.markdown('<div class="evidence-label">Grad-CAM overlay</div>',
                        unsafe_allow_html=True)
            if overlay_png is not None:
                st.image(overlay_png, caption="Model attention overlay",
                         use_container_width=True)
            else:
                st.info("Grad-CAM was not returned by the API.")
    st.caption("Warm regions influenced the model most. "
               "Grad-CAM is explanatory - not a validated lesion localizer.")
    dl1, dl2 = st.columns(2)
    with dl1:
        if original_png is not None:
            st.download_button(f"{ic('download')} Original (PNG)", data=original_png,
                               file_name="original.png", mime="image/png",
                               use_container_width=True, key="dl_orig")
    with dl2:
        if overlay_png is not None:
            st.download_button(f"{ic('download')} Grad-CAM (PNG)", data=overlay_png,
                               file_name="gradcam.png", mime="image/png",
                               use_container_width=True, key="dl_cam")
    return overlay_png


# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:
    _, mid, _ = st.columns([1, 1.35, 1])
    with mid:
        st.markdown(
            """<div class="hero" style="text-align:center">
            <div class="badge">CLINICAL IMAGING COPILOT</div>
            <h1>ClinVision AI</h1>
            <p>RSNA pneumonia detection · Grad-CAM · free-LLM summary</p>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(LOGO_SVG, unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:center;margin:0'>Doctor Sign In</h3>",
                        unsafe_allow_html=True)
            st.caption("Restricted to authorized clinicians.")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input(f"{ic('user')} Username", placeholder="doctor",
                                         autocomplete="username")
                password = st.text_input(f"{ic('key')} Password", type="password",
                                         placeholder="••••••••",
                                         autocomplete="current-password")
                submit = st.form_submit_button(f"{ic('login')} Sign in", type="primary",
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
                f"""<div class="hint-box"><b>Demo credentials</b><br>
                Username: <code>{LOCAL_USER}</code><br>
                Password: <code>{LOCAL_PASS}</code><br>
                <span style="opacity:.75">Change them in <code>.env</code> via
                <code>DOCTOR_USERNAME</code> / <code>DOCTOR_PASSWORD</code>.</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(health_pill(fetch_health(API_URL)), unsafe_allow_html=True)
        st.markdown(
            """<div class="disclaimer"><b>Safety first.</b> Predictions and Grad-CAM
            are model outputs - never a standalone diagnosis.</div>""",
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
    """<div class="hero"><div class="badge">CLINVISION AI · DOCTOR PORTAL</div>
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
        st.markdown(f"{ic('user')} **Dr. {st.session_state.username or 'doctor'}**")
        st.caption("Authenticated session")
        if st.button(f"{ic('logout')} Log out", use_container_width=True):
            do_logout()

st.markdown(
    """<div class="disclaimer"><b>AI assistance only.</b> Prediction + Grad-CAM are
    model outputs, not a diagnosis. Confirm with clinical findings and a qualified
    radiologist.</div>""",
    unsafe_allow_html=True,
)

# Sidebar - reorganised: 1 Patient · 2 Display · 3 Session
with st.sidebar:
    st.header(f"{ic('context')} 1 · Patient context")
    st.caption("Context appears in the summary only - it never changes the vision model.")
    age_known = st.checkbox("Patient age known", value=False, key="sb_age_known")
    age = st.number_input("Age", min_value=0, max_value=120, value=30,
                          disabled=not age_known, key="sb_age")
    sex = st.selectbox("Sex", ["Not specified", "Female", "Male", "Other"],
                       key="sb_sex")
    symptoms = st.text_area("Symptoms", placeholder="e.g. cough, fever, dyspnea",
                            height=70, max_chars=500, key="sb_symptoms")
    context = st.text_area("Additional context",
                           placeholder="History, onset, comorbidities…",
                           height=70, max_chars=500, key="sb_context")
    st.divider()
    st.subheader(f"{ic('theme')} 2 · Display")
    theme_choice = st.selectbox(
        "Color theme",
        ["Light", "Dark"],
        index=["Light", "Dark"].index(st.session_state.theme_mode),
        help="Switches both this portal and Streamlit's native widgets.",
    )
    if theme_choice != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_choice
        st.rerun()
    st.divider()
    st.subheader(f"{ic('settings')} 3 · Session")
    st.caption(f"API `{API_URL}` · limit `{MAX_UPLOAD_MB} MB`")
    if health and health.get("llm"):
        active = [k for k, v in health["llm"].items()
                  if isinstance(v, dict) and v.get("configured")]
        st.caption(f"{ic('llm')} Free LLM: "
                   + (", ".join(active) if active else "template fallback"))
    c_refresh, c_clear = st.columns(2)
    with c_refresh:
        if st.button(f"{ic('refresh')} Status", use_container_width=True,
                     help="Refresh backend health probe"):
            fetch_health.clear()
            fetch_server_history.clear()
            st.rerun()
    with c_clear:
        if st.button(f"{ic('clear')} Clear", use_container_width=True,
                     help="Clear result and uploaded file"):
            st.session_state.result = None
            st.session_state.result_bytes = None
            st.session_state.uploader_key += 1
            st.rerun()

# Plain-text tab labels: tab labels don't expand :material: shortcodes.
tab_analyze, tab_history, tab_about = st.tabs(
    ["Analyze", "History", "About & LLM setup"])

with tab_analyze:
    result = st.session_state.get("result")
    result_bytes = st.session_state.get("result_bytes")
    synced = _result_matches_upload(result_bytes)

    with st.container(border=True):
        st.subheader(f"{ic('upload')} Step 1 · Upload chest X-ray")
        st.caption("Uploading a new file **replaces** the previous one: "
                   "its analysis display is cleared automatically.")
        uploaded = st.file_uploader(
            "DICOM, JPG or PNG",
            type=["dcm", "jpg", "jpeg", "png"],
            key=f"cxr_{st.session_state.uploader_key}",
            help=f"Max {MAX_UPLOAD_MB} MB. DICOM gets modality/VOI LUT preprocessing.")

    # Replace-not-add: a different file clears the previous analysis display
    # (metrics, Grad-CAM, summary, evidence) instead of stacking below it.
    if uploaded is not None:
        _sig = (uploaded.name, uploaded.size,
                hashlib.md5(uploaded.getvalue()).hexdigest())
        if st.session_state.get("upload_sig") != _sig:
            st.session_state.upload_sig = _sig
            st.session_state.result = None
            st.session_state.result_filename = None
            st.session_state.result_bytes = None
            st.session_state.pop("_evidence_original", None)
            st.session_state.pop("_evidence_gradcam", None)
            result = None
            result_bytes = None
            synced = False
            # New file = new patient: queue a full context reset, refilled
            # from DICOM tags when the file provides them, then refresh so
            # the reset applies before the sidebar widgets instantiate.
            meta = (extract_dicom_metadata(uploaded.getvalue(), uploaded.name)
                    if uploaded.name.lower().endswith(".dcm") else {})
            st.session_state.pending_context = {"meta": meta}
            st.rerun()
    else:
        # No file (fresh load or removed via the uploader X): nothing to
        # analyze and nothing to show - the area stays fully empty.
        st.session_state.upload_sig = None
        st.session_state.result = None
        st.session_state.result_filename = None
        st.session_state.result_bytes = None
        st.session_state.pop("_evidence_original", None)
        st.session_state.pop("_evidence_gradcam", None)
        result = None
        result_bytes = None
        synced = True

    # Stepper reflects the state AFTER replace-detection above, so swapping
    # the file immediately drops step 3 in the same run.
    if result and synced:
        step_cls = ["done", "done", "active"]
    elif uploaded is not None:
        step_cls = ["done", "active", ""]
    else:
        step_cls = ["active", "", ""]
    st.markdown(
        f"""<div class="stepper">
        <div class="step {step_cls[0]}">1 · Upload</div>
        <div class="step {step_cls[1]}">2 · Analyze</div>
        <div class="step {step_cls[2]}">3 · Review</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if uploaded:
        raw_bytes = uploaded.getvalue()
        original_png = decode_upload_for_display(raw_bytes, uploaded.name)
        prev_l, prev_r = st.columns(2)
        with prev_l:
            with st.container(border=True):
                st.subheader(f"{ic('attach')} Input preview")
                st.caption(f"`{uploaded.name}` · {len(raw_bytes) / 1024:.0f} KB")
                if original_png is not None:
                    st.image(original_png, caption=uploaded.name,
                             use_container_width=True)
                    if uploaded.name.lower().endswith(".dcm"):
                        st.caption("DICOM decoded locally with modality + VOI LUT handling.")
                        meta = extract_dicom_metadata(raw_bytes, uploaded.name)
                        meta_bits = []
                        if meta.get("patient_id"):
                            meta_bits.append(f"Patient ID: {meta['patient_id']}")
                        if meta.get("study_date"):
                            meta_bits.append(f"Study: {meta['study_date']}")
                        if meta.get("modality"):
                            meta_bits.append(f"Modality: {meta['modality']}")
                        if meta_bits:
                            st.caption(" · ".join(meta_bits))
                            st.caption("Age/Sex auto-filled the sidebar when empty. "
                                       "Symptoms and context have no DICOM tags - enter them manually.")
                        else:
                            st.caption("No usable demographics in this file - "
                                       "fill the sidebar manually.")
                else:
                    st.warning("Preview unavailable - the API will still try to decode the file.")
        with prev_r:
            with st.container(border=True):
                st.subheader(f"{ic('run')} Step 2 · Run analysis")
                st.caption("Calls `POST /analyze`: prediction + Grad-CAM + summary in one request.")
                ctx_bits = [
                    f"Age: {age if age_known else 'N/A'}",
                    f"Sex: {sex}",
                    f"Symptoms: {ic('check') if symptoms.strip() else 'N/A'}",
                ]
                st.caption(" · ".join(ctx_bits))
                run = st.button(f"{ic('go')} Analyze image", type="primary",
                                use_container_width=True)

        if run:
            files = {"file": (uploaded.name, raw_bytes,
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
                        st.error("Session expired - please log in again.")
                        do_logout()
                        st.stop()
                    resp.raise_for_status()
                    result = resp.json()
                    s.write("Grad-CAM + summary ready.")
                    s.update(label="Analysis complete", state="complete")
                st.session_state.result = result
                st.session_state.result_filename = uploaded.name
                st.session_state.result_bytes = raw_bytes
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
    result_bytes = st.session_state.get("result_bytes")
    synced = _result_matches_upload(result_bytes)
    if result and not synced:
        st.warning("The uploaded file changed since this analysis - the results "
                   "below belong to the previously analyzed image. Press "
                   "**Analyze image** to update them.")
    if result:
        prob = float(result.get("probability", 0))
        pred = result.get("prediction", "N/A")
        band_label, band_cls = risk_band(prob)
        with st.container(border=True):
            st.subheader(f"{ic('result')} Step 3 · Model result")
            st.markdown(f'<span class="pill {band_cls}">{band_label}</span>',
                        unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", pred)
            m2.metric("Pneumonia probability", f"{prob:.1%}")
            m3.metric("Inference time", f"{result.get('inference_ms', 'N/A')} ms")
            st.plotly_chart(gauge(prob, dark=theme_is_dark()),
                            use_container_width=True)
            st.caption(f"Request `{result.get('request_id')}` · "
                       f"model `{result.get('model_version')}` · "
                       f"threshold `{result.get('threshold', 0.5)}`")

        # Visual evidence - full width: Original vs Grad-CAM comparison.
        with st.container(border=True):
            st.subheader(f"{ic('compare')} Visual evidence · Original vs Grad-CAM")
            dicom_meta = result.get("dicom_metadata") or {}
            if dicom_meta:
                st.caption(" · ".join(
                    f"{k}: {v}" for k, v in
                    (("Patient ID", dicom_meta.get("patient_id")),
                     ("Study", dicom_meta.get("study_date")),
                     ("Modality", dicom_meta.get("modality"))) if v))
            enc = result.get("gradcam_png_base64")
            gradcam_raw = base64.b64decode(enc) if enc else None
            enc_h = result.get("gradcam_heatmap_base64")
            heatmap_raw = base64.b64decode(enc_h) if enc_h else None
            enc_m = result.get("gradcam_mask_base64")
            mask_raw = base64.b64decode(enc_m) if enc_m else None
            if enc is None and result.get("gradcam_error"):
                st.caption(result["gradcam_error"])
            fname = st.session_state.get("result_filename") or "image"
            # Synchronization: the Original panel ALWAYS decodes the exact
            # bytes that were analyzed (result_bytes), never the current
            # uploader content - overlay and original can never drift apart.
            if result_bytes is not None:
                orig_for_evidence = decode_upload_for_display(
                    result_bytes, fname)
            else:
                up = st.session_state.get(f"cxr_{st.session_state.uploader_key}")
                orig_for_evidence = (
                    decode_upload_for_display(up.getvalue(), up.name)
                    if up is not None else None
                )
            displayed = evidence_panel(orig_for_evidence, gradcam_raw,
                                           heatmap_raw, mask_raw, fname)
            # Export exactly what is on screen: the masked blend at the
            # current slider intensity (falls back to server overlay).
            st.session_state["_evidence_original"] = orig_for_evidence
            st.session_state["_evidence_gradcam"] = displayed or gradcam_raw

        with st.container(border=True):
            st.subheader(f"{ic('summary')} Clinical summary")
            summary = result.get("clinical_summary", {}) or {}
            if summary.get("llm_enabled"):
                st.markdown(f'<span class="pill pill-llm">Free LLM · '
                            f'{summary.get("llm_provider", "llm")}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="pill pill-warn">Template summary '
                            '(no LLM)</span>', unsafe_allow_html=True)
            st.write(summary.get("summary", "No summary available."))
            st.caption(
                f"Assessment: **{summary.get('assessment', 'N/A')}** · "
                f"Probability: **{float(summary.get('model_probability', prob)):.1%}** · "
                f"Source: **{summary.get('llm_provider', 'template')}**"
            )
            with st.expander("Limitations"):
                for item in summary.get("limitations", []):
                    st.write(f"• {item}")

        with st.container(border=True):
            st.subheader(f"{ic('download')} Export report")
            st.caption("Colored PDF with patient context, model result, "
                       "visual evidence, LLM summary and limitations.")
            pdf_bytes = build_pdf_report(
                result, fname,
                st.session_state.get("_evidence_original"),
                st.session_state.get("_evidence_gradcam"),
            )
            st.download_button(f"{ic('download')} Report (PDF)",
                               data=pdf_bytes,
                               file_name="clinvision_report.pdf",
                               mime="application/pdf", use_container_width=True)
    elif not uploaded:
        with st.container(border=True):
            st.subheader(f"{ic('start')} How to start")
            st.markdown("1. Upload a **DICOM, JPG or PNG** chest X-ray above.\n"
                        "2. (Optional) Fill in clinical context in the sidebar.\n"
                        "3. Press **Analyze image** and review prediction, Grad-CAM and summary.")

with tab_history:
    with st.container(border=True):
        st.subheader(f"{ic('folder')} Patient records (SQLite archive)")
        st.caption("Each analysis archives the DICOM/photo, patient info "
                   "(age, sex, symptoms, context) and the LLM summary. "
                   "Local demo storage only.")
        p1, p2 = st.columns(2)
        with p1:
            if st.button(f"{ic('refresh')} Refresh records", use_container_width=True,
                         key="refresh_patients"):
                fetch_patients.clear()
                st.rerun()
        with p2:
            confirm_all = st.checkbox("Confirm clear all", value=False,
                                      key="confirm_clear_patients")
            if st.button(f"{ic('delete')} Clear all records", use_container_width=True,
                         disabled=not confirm_all, key="clear_patients"):
                try:
                    r = requests.delete(f"{API_URL}/patients",
                                        headers=api_headers(), timeout=20)
                    if r.status_code == 200:
                        fetch_patients.clear()
                        st.success("Patient records cleared.")
                        st.rerun()
                    elif r.status_code == 401:
                        st.error("Session expired - please log in again.")
                        do_logout()
                    else:
                        st.error(f"Clear failed (HTTP {r.status_code}).")
                except requests.RequestException as exc:
                    st.error(f"Clear failed: {exc}")
        records = fetch_patients(API_URL, st.session_state.get("token"))
        if records:
            table = [{
                "Date": datetime.fromtimestamp(p.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M"),
                "File": p.get("filename"),
                "Patient ID": (p.get("patient_id") or "N/A")[:8],
                "Prediction": p.get("prediction"),
                "Probability": f"{float(p.get('probability', 0)):.1%}",
                "Age": p.get("age") or "N/A",
                "Sex": p.get("sex") or "N/A",
                "LLM": p.get("llm_provider") or "template",
            } for p in records]
            st.dataframe(table, use_container_width=True, hide_index=True)
            labels = {f"{t['Date']} · {t['File']} · {t['Prediction']}": p["request_id"]
                      for t, p in zip(table, records)}
            chosen = st.selectbox(f"{ic('folder')} Open record", list(labels),
                                  key="patient_record_select")
            rec = next(p for p in records if p["request_id"] == labels[chosen])
            d1, d2, d3 = st.columns(3)
            d1.metric("Prediction", rec.get("prediction"))
            d2.metric("Probability", f"{float(rec.get('probability', 0)):.1%}")
            d3.metric("Model", rec.get("model_version") or "N/A")
            st.caption(
                f"Age: **{rec.get('age') or 'N/A'}** · Sex: **{rec.get('sex') or 'N/A'}** · "
                f"Symptoms: **{rec.get('symptoms') or 'N/A'}** · "
                f"Context: **{rec.get('context') or 'N/A'}**"
            )
            if rec.get("patient_id") or rec.get("study_date") or rec.get("modality"):
                st.caption(" · ".join(
                    f"{k}: **{v}**" for k, v in
                    (("Patient ID", rec.get("patient_id")),
                     ("Study", rec.get("study_date")),
                     ("Modality", rec.get("modality"))) if v))
            tok = st.session_state.get("token")
            img_bytes = fetch_patient_file(API_URL, tok, rec["request_id"], "image")
            cam_bytes = fetch_patient_file(API_URL, tok, rec["request_id"], "gradcam")
            if img_bytes is not None and (rec.get("filename") or "").lower().endswith(".dcm"):
                img_bytes = decode_upload_for_display(img_bytes, rec.get("filename") or "x.dcm")
            i1, i2 = st.columns(2)
            with i1:
                st.markdown('<div class="evidence-label">Archived image</div>',
                            unsafe_allow_html=True)
                if img_bytes:
                    st.image(img_bytes, caption=rec.get("filename"),
                             use_container_width=True)
                else:
                    st.caption("Image unavailable.")
            with i2:
                st.markdown('<div class="evidence-label">Archived Grad-CAM</div>',
                            unsafe_allow_html=True)
                if cam_bytes:
                    st.image(cam_bytes, caption="Model attention overlay",
                             use_container_width=True)
                else:
                    st.caption("No Grad-CAM archived.")
            st.markdown("**LLM summary** "
                        f"(`{rec.get('llm_provider') or 'template'}`)")
            st.write(rec.get("summary") or "N/A")
            if st.button(f"{ic('delete')} Delete this record", key="del_patient"):
                try:
                    r = requests.delete(f"{API_URL}/patients/{rec['request_id']}",
                                        headers=api_headers(), timeout=15)
                    if r.status_code == 200:
                        fetch_patients.clear()
                        fetch_patient_file.clear()
                        st.success("Record deleted.")
                        st.rerun()
                    else:
                        st.error(f"Delete failed (HTTP {r.status_code}).")
                except requests.RequestException as exc:
                    st.error(f"Delete failed: {exc}")
        elif records == []:
            st.caption("No patient records yet - run an analysis first.")
        else:
            st.caption("Records unavailable (API offline or unauthorized).")

    with st.container(border=True):
        st.subheader(f"{ic('history')} Session history (this login)")
        if st.session_state.session_history:
            st.dataframe(st.session_state.session_history,
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No analyses yet in this session.")

    with st.container(border=True):
        st.subheader(f"{ic('server')} Server history (shared demo SQLite)")
        st.caption("Stored without patient identifiers - request id, class and probability only.")
        h1, h2 = st.columns(2)
        with h1:
            if st.button(f"{ic('refresh')} Refresh history", use_container_width=True):
                fetch_server_history.clear()
                st.rerun()
        with h2:
            confirm = st.checkbox("Confirm clear", value=False,
                                  help="Tick to enable the delete button")
            if st.button(f"{ic('delete')} Clear server history", use_container_width=True,
                         disabled=not confirm):
                try:
                    r = requests.delete(f"{API_URL}/history",
                                        headers=api_headers(), timeout=10)
                    if r.status_code == 200:
                        fetch_server_history.clear()
                        st.success("Server history cleared.")
                        st.rerun()
                    elif r.status_code == 401:
                        st.error("Session expired - please log in again.")
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
        st.subheader(f"{ic('how')} How it works")
        st.markdown("1. **DICOM-aware preprocessing** - modality/VOI LUT, MONOCHROME1 "
                    "handling, percentile normalisation, resize to 320×320.\n"
                    "2. **DenseNet121 classifier** (ImageNet transfer learning) → "
                    "pneumonia probability.\n"
                    "3. **Grad-CAM** from the last dense block visualises influential "
                    "regions (explanatory only).\n"
                    "4. **Structured summary**, optionally rewritten by a **free** LLM. "
                    "Clinical context never overrides the image model.")
    with st.container(border=True):
        st.subheader(f"{ic('llm')} Free LLM options (no paid key needed)")
        st.code("brew install ollama && ollama serve && ollama pull llama3.2",
                language="bash")
        st.caption("Then set `OLLAMA_BASE_URL=http://localhost:11434` and "
                   "`OLLAMA_MODEL=llama3.2`. Alternatives (free tiers): "
                   "Groq (`GROQ_API_KEY`), Gemini (`GEMINI_API_KEY`), "
                   "Hugging Face (`HF_API_KEY`). Without any of these the API safely "
                   "falls back to the deterministic template summary.")

st.markdown('<div class="footer">ClinVision AI - AI-assisted prototype · not a medical device · © 2026</div>',
            unsafe_allow_html=True)
