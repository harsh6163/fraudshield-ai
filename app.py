from pathlib import Path
import hashlib
import json
import uuid

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from database import initialize_database, save_cases, update_case_decision, get_all_cases, get_audit_logs

st.set_page_config(
    page_title="FraudShield AI | Transaction Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Support both the current Random Forest filenames and the filenames produced
# by train_model.py. This prevents the app and training script from drifting.
MODEL_CANDIDATES = [
    MODEL_DIR / "random_forest_model.joblib",
    MODEL_DIR / "fraud_model.joblib",
]
METRICS_CANDIDATES = [
    MODEL_DIR / "random_forest_metrics.json",
    MODEL_DIR / "metrics.json",
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])
METRICS_PATH = next((path for path in METRICS_CANDIDATES if path.exists()), METRICS_CANDIDATES[0])

# Persistent case storage
initialize_database()

# -------------------- Premium, high-contrast UI --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{color-scheme:light;--ink:#17243A;--soft:#34445E;--muted:#687991;--line:#E1E8F1;--canvas:#F4F7FB;--blue:#315CE8;--teal:#078C83}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{background:#F4F7FB!important;color:#17243A!important;color-scheme:light!important}
[data-testid="stHeader"]{background:rgba(244,247,251,.98)!important}
#MainMenu,footer{visibility:hidden}
[data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:#fff!important;color:#34445E!important;border-right:1px solid #E1E8F1!important}
[data-testid="stSidebar"]>div:first-child{padding-top:1.2rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] label p,[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#34445E!important;opacity:1!important}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] blockquote,[data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"] p,[data-testid="stText"]{color:#34445E!important}
[data-testid="stMarkdownContainer"] strong{color:#17243A!important}
.brand{font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;letter-spacing:-1px;color:#17243A!important;margin-bottom:2px}
.brand span{color:#315CE8!important}
.brand-caption{font-size:10px;font-weight:800;color:#71809A!important;letter-spacing:1.5px;margin-bottom:24px}
.sidebar-section{font-size:10px;font-weight:800;color:#8793A7!important;letter-spacing:1.5px;margin:15px 0 9px}
[data-testid="stSidebar"] [role="radiogroup"]{gap:5px!important}
[data-testid="stSidebar"] [role="radiogroup"] label{background:transparent!important;border:1px solid transparent!important;border-radius:10px!important;padding:8px 10px!important}
[data-testid="stSidebar"] [role="radiogroup"] label p{color:#40516B!important;font-size:13px!important;font-weight:600!important;opacity:1!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#EDF2FF!important;border-color:#DDE6FF!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#244BC8!important;font-weight:800!important}
.page-eyebrow{color:#315CE8!important;font-size:10px;font-weight:800;letter-spacing:1.8px;text-transform:uppercase;margin:4px 0 8px}
.page-title{font-family:'Manrope',sans-serif;font-size:clamp(28px,3vw,37px);font-weight:800;letter-spacing:-1.3px;color:#17243A!important;line-height:1.2;margin:0 0 9px}
.page-subtitle{color:#687991!important;font-size:14px;line-height:1.75;margin-bottom:23px}
.hero{background:linear-gradient(120deg,#172B50 0%,#254B91 66%,#147F88 100%);border-radius:19px;padding:29px 30px;color:#fff!important;margin:7px 0 23px;box-shadow:0 12px 30px rgba(32,61,111,.12)}
.hero-kicker{font-size:10px;letter-spacing:1.7px;text-transform:uppercase;font-weight:800;color:#C8D8FF!important;margin-bottom:10px}
.hero-title{font-family:'Manrope',sans-serif;font-size:27px;font-weight:800;letter-spacing:-.7px;line-height:1.3;color:#fff!important;margin-bottom:9px}
.hero-copy{color:#E3EDFF!important;font-size:13px;line-height:1.85;max-width:680px}
.metric-card{background:#fff!important;border:1px solid #E1E8F1;border-radius:15px;padding:18px 19px 17px;min-height:120px;box-shadow:0 4px 14px rgba(28,43,73,.025)}
.metric-label{color:#687991!important;font-size:12px;font-weight:700;margin-bottom:13px}
.metric-value{font-family:'Manrope',sans-serif;font-size:29px;font-weight:800;letter-spacing:-1px;color:#17243A!important;line-height:1.2;overflow-wrap:anywhere}
.metric-note{color:#748198!important;font-size:11px;margin-top:8px}
.section-title{font-family:'Manrope',sans-serif;color:#17243A!important;font-size:18px;font-weight:800;letter-spacing:-.4px;margin:25px 0 5px}
.section-copy{color:#687991!important;font-size:12px;margin-bottom:15px}
.panel{background:#fff!important;border:1px solid #E1E8F1;border-radius:15px;padding:19px;color:#17243A!important}
.muted-note{background:#EEF3FF!important;border:1px solid #DCE6FF;color:#294779!important;border-radius:12px;padding:13px 15px;font-size:12px;line-height:1.8;margin:14px 0}
.empty-state{background:#fff!important;border:1px dashed #C9D5E5;border-radius:17px;padding:34px 24px;text-align:center;margin:12px 0}
.empty-title{font-family:'Manrope',sans-serif;font-size:19px;font-weight:800;color:#17243A!important;margin-bottom:8px}
.empty-copy{color:#687991!important;font-size:13px;line-height:1.85}
.stApp div.stButton>button,.stApp div.stDownloadButton>button{background:#fff!important;color:#243653!important;border:1px solid #D3DDEB!important;border-radius:10px!important;font-weight:700!important;min-height:42px;opacity:1!important;box-shadow:none!important}
.stApp div.stButton>button:hover,.stApp div.stDownloadButton>button:hover{background:#F0F4FF!important;color:#244BC8!important;border-color:#9FB4F5!important}
.stApp div.stButton>button[kind="primary"],.stApp div.stDownloadButton>button[kind="primary"]{background:#315CE8!important;color:#fff!important;border-color:#315CE8!important}
.stApp div.stButton>button[kind="primary"]:hover,.stApp div.stDownloadButton>button[kind="primary"]:hover{background:#2549C7!important;color:#fff!important}
/* Light, readable form controls and dropdown menus on every page. */
.stApp input,.stApp textarea,.stApp [data-baseweb="input"] input,.stApp [data-baseweb="textarea"] textarea{background:#fff!important;color:#243653!important;-webkit-text-fill-color:#243653!important;caret-color:#315CE8!important;border-color:#D4DFEF!important;}
.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="input"]>div,.stApp [data-baseweb="textarea"]>div{background:#fff!important;color:#243653!important;border-color:#D4DFEF!important;box-shadow:none!important}
.stApp [data-baseweb="select"] span,.stApp [data-baseweb="select"] div,.stApp [data-baseweb="select"] input{color:#243653!important}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"],[role="option"]{background:#fff!important;color:#243653!important}
[role="option"]:hover,[role="option"][aria-selected="true"]{background:#EDF2FF!important;color:#244BC8!important}
[data-testid="stFileUploader"]{background:#fff!important;border:1px solid #DCE4F0!important;border-radius:14px!important;padding:12px!important}
[data-testid="stFileUploader"] section,[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{background:#F8FAFE!important;border:1px dashed #B9C9DF!important;border-radius:10px!important}
[data-testid="stFileUploader"] p,[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] button{color:#34445E!important;opacity:1!important}
[data-testid="stFileUploader"] button{background:#fff!important;border:1px solid #D4DFEF!important;border-radius:8px!important}
[data-testid="stDataFrame"],[data-testid="stTable"],[data-testid="stPlotlyChart"],[data-testid="stVegaLiteChart"]{background:#fff!important;border:1px solid #E1E8F1!important;border-radius:12px!important;overflow:hidden!important}
[data-testid="stAlert"]{border-radius:11px!important}
[data-testid="stExpander"]{background:#fff!important;border:1px solid #E1E8F1!important;border-radius:12px!important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{color:#34445E!important}
[data-testid="stCheckbox"] label,[data-testid="stRadio"] label{color:#34445E!important}
.footer-note{color:#77859A!important;font-size:11px;text-align:center;padding:25px 0 8px;line-height:1.9}
hr{border-color:#E1E8F1!important}
@media(max-width:768px){.hero{padding:23px 20px}.hero-title{font-size:23px}.metric-card{padding:15px}.metric-value{font-size:24px}}
</style>
""", unsafe_allow_html=True)


# -------------------- Model access (internal only) --------------------
@st.cache_resource
def load_model(path_string, modified_time):
    return joblib.load(path_string)

@st.cache_data
def load_settings(path_string, modified_time):
    with open(path_string, "r", encoding="utf-8") as file:
        return json.load(file)

def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

def get_threshold():
    if not METRICS_PATH.exists():
        return 0.25
    try:
        data = load_settings(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
        return min(max(float(data.get("threshold", 0.25)), 0.0), 1.0)
    except (ValueError, TypeError, OSError, json.JSONDecodeError):
        return 0.25

try:
    model = get_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)

threshold = get_threshold()


# -------------------- Session state --------------------
defaults = {
    "analysis_results": None,
    "analysis_hash": None,
    "analysis_filename": None,
    "analysis_has_labels": False,
    "selected_page": "Overview",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -------------------- UI and data helpers --------------------
def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>""", unsafe_allow_html=True)

def page_header(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="page-eyebrow">{eyebrow}</div>
    <div class="page-title">{title}</div>
    <div class="page-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)

def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-copy">{subtitle}</div>', unsafe_allow_html=True)

def empty_state(title, message):
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-title">{title}</div>
      <div class="empty-copy">{message}</div>
    </div>""", unsafe_allow_html=True)

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def template_bytes(loaded_model):
    if not hasattr(loaded_model, "feature_names_in_"):
        raise ValueError("Saved model is missing its original feature names.")
    return pd.DataFrame(columns=list(loaded_model.feature_names_in_)).to_csv(index=False).encode("utf-8-sig")

def prepare_csv(uploaded_file, loaded_model):
    raw_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("The uploaded CSV contains no transactions.")
    if df.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names. Rename them and try again.")
    if not hasattr(loaded_model, "feature_names_in_"):
        raise ValueError("The saved model is missing its expected input column names.")

    feature_names = list(loaded_model.feature_names_in_)
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        names = ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "")
        raise ValueError(f"This CSV does not match the template. Missing columns: {names}")

    X = df[feature_names].copy()
    for col in feature_names:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any():
        bad = ", ".join(X.columns[X.isna().any()].tolist()[:8])
        raise ValueError(f"Required fields contain blank or non-numeric values: {bad}")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("The CSV contains infinite or invalid numeric values.")

    actual = None
    if "Class" in df.columns:
        labels = pd.to_numeric(df["Class"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("If present, the optional Class column must contain only 0 or 1.")
        actual = labels.astype(int).to_numpy()
    return df, X, actual, file_hash

def screen_transactions(df, X, actual, loaded_model, active_threshold):
    probabilities = loaded_model.predict_proba(X)
    classes = list(loaded_model.classes_)
    if 1 not in classes:
        raise ValueError("The saved model cannot produce a fraud-class score.")
    scores = probabilities[:, classes.index(1)]
    results = df.copy()

    if "Transaction ID" not in results.columns:
        if "transaction_id" in results.columns:
            results["Transaction ID"] = results["transaction_id"].astype(str)
        elif "id" in results.columns:
            results["Transaction ID"] = results["id"].astype(str)
        else:
            results["Transaction ID"] = [f"TXN-{i:06d}" for i in range(1, len(results) + 1)]

    batch_id = uuid.uuid4().hex[:8].upper()
    results["Case Reference"] = [f"CASE-{batch_id}-{i:06d}" for i in range(1, len(results) + 1)]
    results["Fraud Risk Score"] = scores
    results["Monitoring Status"] = np.where(scores >= active_threshold, "Flagged for review", "Not flagged")
    results["Review Decision"] = np.where(scores >= active_threshold, "Pending review", "Not required")
    if actual is not None:
        results["Actual Class"] = actual
    return results

def get_results():
    value = st.session_state.analysis_results
    return None if value is None else value.copy()

def get_saved_cases():
    """Return persistent case records with customer-facing column names."""
    rows = get_all_cases()
    if not rows:
        return pd.DataFrame(columns=[
            "Case Reference", "Transaction ID", "Fraud Risk Score",
            "Monitoring Status", "Review Decision", "Analyst Notes",
            "Source File", "Created At", "Updated At"
        ])
    saved = pd.DataFrame(rows)
    saved = saved.rename(columns={
        "case_reference": "Case Reference",
        "transaction_id": "Transaction ID",
        "risk_score": "Fraud Risk Score",
        "monitoring_status": "Monitoring Status",
        "review_decision": "Review Decision",
        "analyst_notes": "Analyst Notes",
        "source_file": "Source File",
        "created_at": "Created At",
        "updated_at": "Updated At",
    })
    return saved

def display_results(df):
    view = df.copy()
    if "Fraud Risk Score" in view.columns:
        view["Fraud Risk Score"] = (view["Fraud Risk Score"] * 100).map(lambda x: f"{x:.2f}%")
    return view


def get_model_estimator(loaded_model):
    """Return the final estimator from a scikit-learn Pipeline, or the model itself."""
    if hasattr(loaded_model, "named_steps") and loaded_model.named_steps:
        return list(loaded_model.named_steps.values())[-1]
    return loaded_model


def get_feature_importance(loaded_model):
    """Build model-level importance from supported fitted estimators."""
    estimator = get_model_estimator(loaded_model)
    feature_names = list(getattr(loaded_model, "feature_names_in_", []))

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        method = "Built-in tree feature importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
        method = "Absolute model coefficients (global importance)"
    else:
        return None, "This model does not expose a supported built-in importance measure."

    if len(feature_names) != len(values):
        try:
            if hasattr(loaded_model, "named_steps") and "preprocessor" in loaded_model.named_steps:
                feature_names = list(loaded_model.named_steps["preprocessor"].get_feature_names_out())
        except Exception:
            feature_names = []

    if len(feature_names) != len(values):
        feature_names = [f"Feature {i + 1}" for i in range(len(values))]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": values,
    }).sort_values("Importance", ascending=False, kind="stable")
    return importance, method


# -------------------- Sidebar navigation --------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">FraudShield<span> AI</span></div>
    <div class="brand-caption">TRANSACTION INTELLIGENCE</div>
    <div class="sidebar-section">WORKSPACE</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Workspace navigation",
        ["Overview", "Analyze Transactions", "Model Explainability", "False Positive Analysis", "Review Queue", "Reports", "Audit Trail"],
        key="selected_page",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="sidebar-section">CURRENT SESSION</div>', unsafe_allow_html=True)
    current = get_results()
    if current is None:
        st.caption("No analysis completed yet.")
    else:
        st.caption(f"{len(current):,} transactions analyzed")
        st.caption(f"{int(current['Review Decision'].eq('Pending review').sum()):,} reviews pending")
    st.divider()
    st.markdown(
        '<div style="font-size:11px;color:#71809A;line-height:1.9;">Fraud monitoring workspace<br>Automated screening with human review</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":
    page_header(
        "TRANSACTION INTELLIGENCE",
        "Fraud monitoring overview",
        "Review transaction activity, prioritize suspicious patterns, and focus your team's attention where it matters.",
    )
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">YOUR FRAUD MONITORING WORKSPACE</div>
      <div class="hero-title">See the signals. Review with confidence.</div>
      <div class="hero-copy">Screen transaction records, prioritize activity that needs attention, and prepare clear reports for your team.</div>
    </div>""", unsafe_allow_html=True)

    results = get_results()
    if results is None:
        cols = st.columns(4)
        for col, vals in zip(cols, [
            ("Transactions analyzed", "—", "Awaiting first analysis"),
            ("Flagged for review", "—", "No screening results yet"),
            ("Pending reviews", "—", "No open cases yet"),
            ("Flag rate", "—", "Available after analysis"),
        ]):
            with col:
                metric_card(*vals)
        empty_state("Your workspace is ready", "Upload a transaction CSV to begin. This overview will display real screening results after your first analysis.")
        if st.button("Analyze transactions", type="primary"):
            st.session_state.selected_page = "Analyze Transactions"
            st.rerun()
    else:
        total = len(results)
        flagged_n = int(results["Monitoring Status"].eq("Flagged for review").sum())
        pending_n = int(results["Review Decision"].eq("Pending review").sum())
        cols = st.columns(4)
        metrics = [
            ("Transactions analyzed", f"{total:,}", "Current uploaded dataset"),
            ("Flagged for review", f"{flagged_n:,}", "Require closer inspection"),
            ("Pending reviews", f"{pending_n:,}", "Awaiting analyst decision"),
            ("Flag rate", f"{flagged_n / total:.1%}" if total else "0.0%", "Share of analyzed transactions"),
        ]
        for col, vals in zip(cols, metrics):
            with col:
                metric_card(*vals)

        section_header("Screening summary", "A snapshot of the current dataset.")
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown("**Transaction outcomes**")
            counts = results["Monitoring Status"].value_counts().reindex(["Not flagged", "Flagged for review"], fill_value=0)
            st.bar_chart(counts, color="#315CE8")
        with right:
            st.markdown("**Highest-risk transactions**")
            top = results.nlargest(5, "Fraud Risk Score")[["Transaction ID", "Fraud Risk Score", "Monitoring Status"]].copy()
            top["Fraud Risk Score"] = (top["Fraud Risk Score"] * 100).map(lambda x: f"{x:.2f}%")
            st.dataframe(top, hide_index=True, use_container_width=True)
        section_header("Recent screening activity")
        st.caption(f"Current dataset: {st.session_state.analysis_filename or 'Uploaded transactions'}")
        st.dataframe(display_results(results[["Transaction ID", "Fraud Risk Score", "Monitoring Status", "Review Decision"]].head(10)), hide_index=True, use_container_width=True)
        if st.button("Open review queue", type="primary"):
            st.session_state.selected_page = "Review Queue"
            st.rerun()


# =========================================================
# ANALYZE TRANSACTIONS
# =========================================================
elif page == "Analyze Transactions":
    page_header(
        "TRANSACTION SCREENING",
        "Analyze transactions",
        "Upload a transaction dataset to screen activity and create a prioritized list for review.",
    )

    if model is None:
        st.error("Transaction screening is currently unavailable. Please contact the workspace administrator.")
        if model_error:
            with st.expander("Administrator diagnostics"):
                st.code(model_error)
        st.stop()

    st.markdown("""
    <div class="panel">
      <strong>Before you upload</strong><br>
      <span style="color:#687991;font-size:13px;line-height:1.9;">Use a CSV that matches the supplied template. Do not upload card numbers, passwords, CVVs, or other unnecessary sensitive data.</span>
    </div>""", unsafe_allow_html=True)
    st.write("")
    st.download_button(
        "Download CSV template",
        data=template_bytes(model),
        file_name="fraudshield_transaction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload transaction data", type=["csv"], help="Upload a CSV containing the required transaction fields.")
    if uploaded is not None:
        st.caption(f"Selected file: {uploaded.name}")
        if st.button("Screen transactions", type="primary", use_container_width=True):
            try:
                with st.spinner("Screening transactions..."):
                    df, X, actual, file_hash = prepare_csv(uploaded, model)
                    results = screen_transactions(df, X, actual, model, threshold)
                st.session_state.analysis_results = results
                st.session_state.analysis_hash = file_hash
                st.session_state.analysis_filename = uploaded.name
                st.session_state.analysis_has_labels = actual is not None

                # Save flagged cases for follow-up review across app restarts.
                flagged_cases = results.loc[
                    results["Monitoring Status"].eq("Flagged for review")
                ].copy()
                if not flagged_cases.empty:
                    flagged_cases["Analyst Notes"] = ""
                    save_cases(flagged_cases, source_file=uploaded.name)

                st.success(
                    f"Screening complete — {len(results):,} transactions processed. "
                    f"{len(flagged_cases):,} flagged cases saved to the review queue."
                )
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))
            except Exception:
                st.error("Screening could not be completed. Check the file format and required fields, then try again.")

    results = get_results()
    if results is not None:
        st.divider()
        st.markdown("### Latest screening results")
        flagged_n = int(results["Monitoring Status"].eq("Flagged for review").sum())
        cols = st.columns(3)
        for col, vals in zip(cols, [
            ("Transactions processed", f"{len(results):,}", "Rows screened"),
            ("Flagged for review", f"{flagged_n:,}", "Require inspection"),
            ("Not flagged", f"{len(results) - flagged_n:,}", "Did not meet screening rule"),
        ]):
            with col:
                metric_card(*vals)
        st.download_button(
            "Download screening results",
            data=csv_bytes(results),
            file_name="fraudshield_screening_results.csv",
            mime="text/csv",
            type="primary",
        )
        st.dataframe(display_results(results.head(100)), hide_index=True, use_container_width=True)
        if len(results) > 100:
            st.caption("Showing the first 100 records. Download the report to access all results.")


# =========================================================
# REVIEW QUEUE
# =========================================================
elif page == "Review Queue":
    page_header(
        "INVESTIGATION WORKSPACE",
        "Review queue",
        "Inspect flagged transactions and record your team's review decisions.",
    )

    saved_cases = get_saved_cases()
    if saved_cases.empty:
        empty_state(
            "No transactions to review",
            "Run a screening first. Flagged transactions will be saved here for follow-up."
        )
    else:
        pending_mask = saved_cases["Review Decision"].eq("Pending review")
        pending_n = int(pending_mask.sum())
        escalated_n = int(saved_cases["Review Decision"].eq("Reviewed — escalate").sum())
        cleared_n = int(saved_cases["Review Decision"].eq("Reviewed — clear").sum())

        cols = st.columns(3)
        for col, vals in zip(cols, [
            ("Open cases", f"{pending_n:,}", "Awaiting a decision"),
            ("Escalated", f"{escalated_n:,}", "Marked for investigation"),
            ("Cleared after review", f"{cleared_n:,}", "Analyst decision recorded"),
        ]):
            with col:
                metric_card(*vals)

        section_header("Saved cases", "Search, filter, and prioritize cases across all screening sessions.")

        search_col, decision_col, sort_col = st.columns([2, 1, 1])
        with search_col:
            search_term = st.text_input(
                "Search cases",
                placeholder="Enter a case reference or transaction ID",
                key="review_queue_search",
            ).strip().lower()
        with decision_col:
            filter_choice = st.selectbox(
                "Review decision",
                ["All cases", "Pending review", "Reviewed — escalate", "Reviewed — clear"],
                key="review_queue_decision_filter",
            )
        with sort_col:
            sort_choice = st.selectbox(
                "Sort by",
                ["Most recently updated", "Highest risk score", "Lowest risk score"],
                key="review_queue_sort",
            )

        visible_cases = saved_cases.copy()
        if filter_choice != "All cases":
            visible_cases = visible_cases.loc[
                visible_cases["Review Decision"].eq(filter_choice)
            ]

        if search_term:
            reference_match = visible_cases["Case Reference"].astype(str).str.lower().str.contains(
                search_term, regex=False, na=False
            )
            transaction_match = visible_cases["Transaction ID"].astype(str).str.lower().str.contains(
                search_term, regex=False, na=False
            )
            visible_cases = visible_cases.loc[reference_match | transaction_match]

        if sort_choice == "Highest risk score":
            visible_cases = visible_cases.sort_values(
                "Fraud Risk Score", ascending=False, kind="stable"
            )
        elif sort_choice == "Lowest risk score":
            visible_cases = visible_cases.sort_values(
                "Fraud Risk Score", ascending=True, kind="stable"
            )
        elif "Updated At" in visible_cases.columns:
            visible_cases = visible_cases.sort_values(
                "Updated At", ascending=False, kind="stable"
            )

        st.caption(f"Showing {len(visible_cases):,} of {len(saved_cases):,} saved cases.")

        table_cols = [
            "Case Reference", "Transaction ID", "Fraud Risk Score",
            "Monitoring Status", "Review Decision", "Updated At"
        ]
        table_view = visible_cases[[c for c in table_cols if c in visible_cases.columns]].copy()
        if "Fraud Risk Score" in table_view.columns:
            table_view["Fraud Risk Score"] = (table_view["Fraud Risk Score"] * 100).map(lambda x: f"{x:.2f}%")
        st.dataframe(table_view, hide_index=True, use_container_width=True)

        if not visible_cases.empty:
            section_header("Review a case", "Choose a saved case to update its decision and notes.")
            options = visible_cases["Case Reference"].astype(str).tolist()

            def case_label(ref):
                match = visible_cases.loc[visible_cases["Case Reference"].astype(str).eq(ref)]
                txn = str(match.iloc[0]["Transaction ID"]) if not match.empty else "Unknown"
                return f"{ref} · {txn}"

            selected = st.selectbox("Select a case", options=options, format_func=case_label)
            case = saved_cases.loc[saved_cases["Case Reference"].astype(str).eq(selected)].iloc[0]

            left, right = st.columns(2)
            with left:
                st.markdown("**Case summary**")
                st.markdown(f"**Case reference:** {case['Case Reference']}")
                st.markdown(f"**Transaction:** {case['Transaction ID']}")
                st.markdown(f"**Status:** {case['Monitoring Status']}")
                st.markdown(f"**Risk score:** {float(case['Fraud Risk Score']):.2%}")
                st.markdown(f"**Review decision:** {case['Review Decision']}")
                if case.get("Source File"):
                    st.markdown(f"**Source file:** {case['Source File']}")
            with right:
                st.markdown("**Review notes**")
                st.caption("Record a concise rationale for the decision. Avoid entering unnecessary sensitive information.")
            decisions = ["Pending review", "Reviewed — escalate", "Reviewed — clear"]
            current_decision = str(case["Review Decision"])
            decision = st.selectbox(
                "Record review decision",
                decisions,
                index=decisions.index(current_decision) if current_decision in decisions else 0,
                key=f"persistent_decision_{selected}",
                help="This records an analyst assessment; it does not automatically confirm or rule out fraud.",
            )
            existing_notes = str(case.get("Analyst Notes", "") or "")
            notes = st.text_area(
                "Analyst notes",
                value=existing_notes,
                placeholder="Add the reason for escalation, review findings, or follow-up action...",
                key=f"persistent_notes_{selected}",
                max_chars=2000,
            )
            if st.button("Save review decision", type="primary", use_container_width=True):
                update_case_decision(selected, decision, notes)
                st.success("Review decision and notes saved.")
                st.rerun()

        st.markdown("""
        <div class="muted-note"><strong>Review responsibly.</strong> A screening alert is not proof of fraud. Review available evidence and follow your organization's investigation process before taking action.</div>
        """, unsafe_allow_html=True)


# =========================================================
# MODEL EXPLAINABILITY
# =========================================================
elif page == "Model Explainability":
    page_header(
        "MODEL TRANSPARENCY",
        "Model explainability",
        "Understand which input features influence the model overall and inspect the inputs for a selected transaction.",
    )

    if model is None:
        st.error("The trained model could not be loaded, so explainability is unavailable.")
        if model_error:
            with st.expander("Administrator diagnostics"):
                st.code(model_error)
    else:
        importance, importance_method = get_feature_importance(model)
        estimator = get_model_estimator(model)
        model_name = estimator.__class__.__name__

        info_left, info_right = st.columns(2)
        with info_left:
            metric_card("Model estimator", model_name, "Loaded from the models folder")
        with info_right:
            metric_card("Fraud threshold", f"{threshold:.0%}", "Scores at or above this level are flagged")

        st.markdown(
            '<div class="muted-note"><strong>How to read this page.</strong> Global feature importance describes which inputs the fitted model relies on most overall. It does not prove that a feature caused fraud or explain every individual prediction. A flagged transaction still requires human review.</div>',
            unsafe_allow_html=True,
        )

        section_header("Global feature importance", importance_method)
        if importance is None or importance.empty:
            st.info(importance_method)
        else:
            top_n = min(20, len(importance))
            chart_data = importance.head(top_n).sort_values("Importance", ascending=True)
            st.bar_chart(chart_data.set_index("Feature"), use_container_width=True)
            with st.expander("View importance values"):
                st.dataframe(importance, hide_index=True, use_container_width=True)
            st.download_button(
                "Download feature importance CSV",
                data=csv_bytes(importance),
                file_name="fraudshield_feature_importance.csv",
                mime="text/csv",
            )

        section_header("Inspect a screened transaction", "Select a transaction from the current analysis session to review its score and model input values.")
        current_results = get_results()
        if current_results is None or current_results.empty:
            empty_state(
                "No screening results in this session",
                "Open Analyze Transactions and screen a CSV first. Then return here to inspect a transaction.",
            )
        else:
            choices = list(range(len(current_results)))
            selected_row = st.selectbox(
                "Transaction",
                options=choices,
                format_func=lambda i: (
                    f"{current_results.iloc[i].get('Transaction ID', f'Row {i + 1}')} · "
                    f"{float(current_results.iloc[i]['Fraud Risk Score']):.2%} risk"
                ),
                key="explainability_transaction_row",
            )
            selected_transaction = current_results.iloc[selected_row]
            risk_score = float(selected_transaction["Fraud Risk Score"])
            a, b, c = st.columns(3)
            with a:
                metric_card("Transaction ID", str(selected_transaction.get("Transaction ID", "—")), "Selected transaction")
            with b:
                metric_card("Fraud risk score", f"{risk_score:.2%}", "Model-predicted score")
            with c:
                metric_card("Screening outcome", str(selected_transaction.get("Monitoring Status", "—")), "Based on configured threshold")

            model_features = list(getattr(model, "feature_names_in_", []))
            available_features = [name for name in model_features if name in current_results.columns]
            if available_features:
                feature_values = pd.DataFrame({
                    "Feature": available_features,
                    "Input Value": [selected_transaction[name] for name in available_features],
                })
                st.dataframe(feature_values, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download selected transaction inputs",
                    data=csv_bytes(feature_values),
                    file_name="fraudshield_selected_transaction_inputs.csv",
                    mime="text/csv",
                )
            else:
                st.info("The saved model's input columns could not be matched to the current screening results.")

        st.caption("Note: global importance is model-level context, not a causal explanation or a guarantee that a transaction is fraudulent.")


# =========================================================
# FALSE POSITIVE ANALYSIS
# =========================================================
elif page == "False Positive Analysis":
    page_header(
        "MODEL QUALITY & ERROR ANALYSIS",
        "False positive analysis",
        "Compare model flags with known labels to understand false alarms and missed fraud in a labelled dataset.",
    )

    current_results = get_results()
    if current_results is None or current_results.empty:
        empty_state(
            "No screening results available",
            "Go to Analyze Transactions and screen a CSV first. To calculate false positives and false negatives, the CSV must include a valid Class column with 0 = non-fraud and 1 = fraud.",
        )
    elif "Actual Class" not in current_results.columns:
        st.warning(
            "This screening session does not contain ground-truth labels. Upload a labelled evaluation CSV with a Class column (0 = non-fraud, 1 = fraud) to calculate false positives, false negatives, precision, and recall. Do not use model predictions as ground truth."
        )
        st.caption("For real-world unlabeled transactions, the model cannot determine which alerts are false positives without verified outcomes.")
    else:
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

        evaluated = current_results.copy()
        y_true = pd.to_numeric(evaluated["Actual Class"], errors="coerce").astype(int)
        y_pred = (pd.to_numeric(evaluated["Fraud Risk Score"], errors="coerce") >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        cols = st.columns(4)
        cards = [
            ("False positives", f"{fp:,}", "Non-fraud transactions flagged"),
            ("False negatives", f"{fn:,}", "Fraud transactions not flagged"),
            ("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.1%}", "Share of flags that are fraud"),
            ("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.1%}", "Share of fraud cases detected"),
        ]
        for col, vals in zip(cols, cards):
            with col:
                metric_card(*vals)

        st.markdown(
            '<div class="muted-note"><strong>Interpretation.</strong> A false positive is a known non-fraud transaction that was flagged. A false negative is a known fraud transaction that was not flagged. The counts are meaningful only when the uploaded Class labels are reliable and representative.</div>',
            unsafe_allow_html=True,
        )

        section_header("Confusion matrix", f"Evaluation threshold: {threshold:.2%}")
        matrix_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual non-fraud", "Actual fraud"],
            columns=["Predicted not flagged", "Predicted flagged"],
        )
        st.dataframe(matrix_df, use_container_width=True)

        other_cols = st.columns(2)
        with other_cols[0]:
            metric_card("True negatives", f"{tn:,}", "Non-fraud correctly not flagged")
        with other_cols[1]:
            metric_card("True positives", f"{tp:,}", "Fraud correctly flagged")

        section_header("Review model errors", "Use these rows to inspect patterns; they are not proof of why the model made an error.")
        evaluated["Evaluation Result"] = np.select(
            [
                (y_true == 0) & (y_pred == 1),
                (y_true == 1) & (y_pred == 0),
                (y_true == 1) & (y_pred == 1),
            ],
            ["False positive", "False negative", "True positive"],
            default="True negative",
        )
        error_type = st.selectbox(
            "Rows to inspect",
            ["False positives", "False negatives", "All incorrect predictions", "All evaluated rows"],
            key="error_analysis_filter",
        )
        if error_type == "False positives":
            detail = evaluated.loc[evaluated["Evaluation Result"].eq("False positive")].copy()
        elif error_type == "False negatives":
            detail = evaluated.loc[evaluated["Evaluation Result"].eq("False negative")].copy()
        elif error_type == "All incorrect predictions":
            detail = evaluated.loc[evaluated["Evaluation Result"].isin(["False positive", "False negative"])].copy()
        else:
            detail = evaluated.copy()

        detail = detail.sort_values("Fraud Risk Score", ascending=False)
        display_cols = [c for c in ["Transaction ID", "Actual Class", "Fraud Risk Score", "Monitoring Status", "Evaluation Result"] if c in detail.columns]
        shown = detail[display_cols].copy()
        if "Fraud Risk Score" in shown.columns:
            shown["Fraud Risk Score"] = (shown["Fraud Risk Score"] * 100).map(lambda x: f"{x:.2f}%")
        st.caption(f"Showing {len(shown):,} rows.")
        st.dataframe(shown.head(500), hide_index=True, use_container_width=True)
        if len(shown) > 500:
            st.caption("Showing the first 500 rows here; download the full selection below.")
        st.download_button(
            "Download selected evaluation rows",
            data=csv_bytes(detail),
            file_name="fraudshield_model_error_analysis.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(
            f"Accuracy: {accuracy_score(y_true, y_pred):.2%} · "
            f"F1 score: {f1_score(y_true, y_pred, zero_division=0):.2%} · "
            f"Evaluated rows: {len(evaluated):,}. Accuracy can be misleading for highly imbalanced fraud datasets; consider precision, recall, F1, and average precision together."
        )


# =========================================================
# REPORTS
# =========================================================
elif page == "Reports":
    page_header(
        "REPORTING CENTER",
        "Reports & exports",
        "Download screening results and review outcomes for your team.",
    )

    saved_cases = get_saved_cases()
    if saved_cases.empty:
        empty_state(
            "Your reports will appear here",
            "Complete a transaction screening to save flagged cases and generate review reports."
        )
    else:
        flagged = saved_cases["Monitoring Status"].eq("Flagged for review")
        escalated = saved_cases["Review Decision"].eq("Reviewed — escalate")
        cleared = saved_cases["Review Decision"].eq("Reviewed — clear")
        pending = saved_cases["Review Decision"].eq("Pending review")

        cols = st.columns(4)
        for col, vals in zip(cols, [
            ("Saved cases", f"{len(saved_cases):,}", "Across screening sessions"),
            ("Flagged cases", f"{int(flagged.sum()):,}", "Saved for review"),
            ("Escalated", f"{int(escalated.sum()):,}", "Marked for investigation"),
            ("Pending review", f"{int(pending.sum()):,}", "Awaiting a decision"),
        ]):
            with col:
                metric_card(*vals)

        section_header("Export your data", "These reports use saved case records and persist across app restarts.")
        export_cols = [
            "Case Reference", "Transaction ID", "Fraud Risk Score",
            "Monitoring Status", "Review Decision", "Analyst Notes",
            "Source File", "Created At", "Updated At"
        ]
        export_df = saved_cases[[c for c in export_cols if c in saved_cases.columns]].copy()
        left, right = st.columns(2)
        with left:
            st.markdown("**Complete case report**")
            st.caption("All saved flagged cases, screening outcomes, and analyst decisions.")
            st.download_button(
                "Download full case report",
                data=csv_bytes(export_df),
                file_name="fraudshield_saved_case_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with right:
            st.markdown("**Open cases report**")
            st.caption("Cases that are still waiting for an analyst decision.")
            st.download_button(
                "Download pending cases",
                data=csv_bytes(export_df.loc[pending].copy()),
                file_name="fraudshield_pending_cases.csv",
                mime="text/csv",
                use_container_width=True,
            )
        section_header("Review outcome summary")
        st.bar_chart(pd.Series({
            "Pending review": int(pending.sum()),
            "Reviewed — escalate": int(escalated.sum()),
            "Reviewed — clear": int(cleared.sum()),
        }, name="Cases"), color="#078C83")
        st.markdown("""
        <div class="muted-note">Reports summarize screening alerts and analyst decisions. A screening alert is not proof of fraud, and a case marked clear is not a guarantee that a transaction is legitimate.</div>
        """, unsafe_allow_html=True)



# =========================================================
# AUDIT TRAIL
# =========================================================
elif page == "Audit Trail":
    page_header(
        "GOVERNANCE & ACCOUNTABILITY",
        "Audit trail",
        "Review a timestamped history of changes to saved case decisions and analyst notes.",
    )
    audit_rows = get_audit_logs()
    if not audit_rows:
        empty_state(
            "No audit activity yet",
            "Changes to saved case decisions or analyst notes will appear here.",
        )
    else:
        audit_df = pd.DataFrame(audit_rows).rename(columns={
            "id": "Event ID",
            "case_reference": "Case Reference",
            "transaction_id": "Transaction ID",
            "previous_decision": "Previous Decision",
            "new_decision": "New Decision",
            "previous_notes": "Previous Notes",
            "new_notes": "New Notes",
            "changed_at": "Changed At",
        })
        cols = st.columns(3)
        with cols[0]:
            metric_card("Audit events", f"{len(audit_df):,}", "Recorded changes")
        with cols[1]:
            metric_card("Cases affected", f"{audit_df['Case Reference'].nunique():,}", "Unique cases")
        with cols[2]:
            metric_card("Latest change", str(audit_df.iloc[0]["Changed At"]), "Most recent event")

        section_header("Find audit events", "Search by case or transaction and filter by the new decision.")
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            search = st.text_input("Search events", placeholder="Case reference or transaction ID", key="audit_search").strip().lower()
        with c2:
            decision_filter = st.selectbox(
                "New decision",
                ["All decisions", "Pending review", "Reviewed — escalate", "Reviewed — clear"],
                key="audit_decision_filter",
            )
        with c3:
            include_notes = st.checkbox("Include notes", value=True, key="audit_include_notes")

        visible = audit_df.copy()
        if search:
            m1 = visible["Case Reference"].astype(str).str.lower().str.contains(search, regex=False, na=False)
            m2 = visible["Transaction ID"].astype(str).str.lower().str.contains(search, regex=False, na=False)
            visible = visible.loc[m1 | m2]
        if decision_filter != "All decisions":
            visible = visible.loc[visible["New Decision"].eq(decision_filter)]
        if not include_notes:
            visible = visible.drop(columns=["Previous Notes", "New Notes"])
        st.caption(f"Showing {len(visible):,} audit events.")
        st.dataframe(visible, hide_index=True, use_container_width=True)
        st.download_button(
            "Download audit trail CSV",
            data=csv_bytes(visible),
            file_name="fraudshield_audit_trail.csv",
            mime="text/csv",
            type="primary",
        )
        st.markdown(
            '<div class="muted-note"><strong>Audit integrity.</strong> This log records changes made through the review workflow; it does not independently establish whether a transaction is fraudulent.</div>',
            unsafe_allow_html=True,
        )


# -------------------- Footer --------------------
st.markdown("""
<div class="footer-note">
FraudShield AI · Transaction monitoring and investigation support<br>
Automated screening supports human review; it does not guarantee that a transaction is safe or fraudulent.
</div>
""", unsafe_allow_html=True)
