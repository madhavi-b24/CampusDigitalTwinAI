"""Streamlit app for Campus Digital Twin AI."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agents.profile_agent import (
    GeminiConfigurationError,
    ProfileAgent,
    ProfileAgentError,
)
from agents.twin_agent import InvalidStudentProfileError, TwinAgent
from utils.pdf_reader import InvalidPDFError, extract_text_from_pdf


APP_TITLE = "Campus Digital Twin AI"
APP_SUBTITLE = "Resume to student profile to career-readiness digital twin."
DEFAULT_MODEL = "gemini-2.5-flash"


def main() -> None:
    """Render the Campus Digital Twin AI Streamlit app."""
    load_dotenv()
    _configure_page()
    _inject_styles()
    _initialize_session_state()

    _render_header()
    uploaded_resume, model_name = _render_sidebar()

    if uploaded_resume is None:
        _render_empty_state()
        return

    _render_file_summary(uploaded_resume)

    generate_clicked = st.button(
        "Generate Digital Twin Report",
        type="primary",
        use_container_width=True,
    )

    if generate_clicked:
        _run_pipeline(uploaded_resume, model_name)

    if st.session_state.get("report"):
        _render_results(
            resume_text=st.session_state["resume_text"],
            profile=st.session_state["profile"],
            report=st.session_state["report"],
        )


def _configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .app-header {
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
        }
        .app-header h1 {
            font-size: 2.1rem;
            font-weight: 750;
            letter-spacing: 0;
            margin: 0;
        }
        .app-header p {
            color: #475569;
            font-size: 1rem;
            margin: 0.35rem 0 0 0;
        }
        .section-title {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.75rem 0;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session_state() -> None:
    defaults = {
        "resume_text": "",
        "profile": None,
        "report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> tuple[Any | None, str]:
    with st.sidebar:
        st.header("Workspace")
        uploaded_resume = st.file_uploader(
            "Resume PDF",
            type=["pdf"],
            accept_multiple_files=False,
        )
        model_name = st.text_input("Gemini model", value=DEFAULT_MODEL)

        api_ready = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if api_ready:
            st.success("Gemini API key detected.")
        else:
            st.warning("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env.")

        if st.button("Clear Results", use_container_width=True):
            st.session_state["resume_text"] = ""
            st.session_state["profile"] = None
            st.session_state["report"] = None

    return uploaded_resume, model_name.strip() or DEFAULT_MODEL


def _render_empty_state() -> None:
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.subheader("Upload a resume PDF to begin")
        st.write(
            "The app will extract resume text, create a structured student profile, "
            "and generate a career-readiness Digital Twin Report."
        )
    with right:
        st.info("Supported input: PDF resume")


def _render_file_summary(uploaded_resume: Any) -> None:
    file_size_kb = getattr(uploaded_resume, "size", 0) / 1024
    st.markdown('<div class="section-title">Selected Resume</div>', unsafe_allow_html=True)
    col_name, col_type, col_size = st.columns(3)
    col_name.metric("File", getattr(uploaded_resume, "name", "Uploaded PDF"))
    col_type.metric("Type", "PDF")
    col_size.metric("Size", f"{file_size_kb:.1f} KB")


def _run_pipeline(uploaded_resume: Any, model_name: str) -> None:
    st.session_state["resume_text"] = ""
    st.session_state["profile"] = None
    st.session_state["report"] = None

    try:
        with st.status("Generating Digital Twin Report", expanded=True) as status:
            st.write("Extracting resume text")
            resume_text = extract_text_from_pdf(uploaded_resume)
            if not resume_text:
                raise InvalidPDFError("No readable text was found in the PDF.")
            st.session_state["resume_text"] = resume_text

            st.write("Creating student profile")
            profile_agent = ProfileAgent(model_name=model_name)
            profile = profile_agent.run(resume_text)
            st.session_state["profile"] = profile

            st.write("Generating digital twin report")
            twin_agent = TwinAgent()
            report = twin_agent.run(profile)
            st.session_state["report"] = report

            status.update(label="Digital Twin Report ready", state="complete", expanded=False)
    except InvalidPDFError as exc:
        st.error(f"PDF processing failed: {exc}")
    except GeminiConfigurationError as exc:
        st.error(f"Gemini configuration error: {exc}")
    except ProfileAgentError as exc:
        st.error(f"Profile generation failed: {exc}")
    except InvalidStudentProfileError as exc:
        st.error(f"Digital twin generation failed: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")


def _render_results(
    *,
    resume_text: str,
    profile: dict[str, list[str]],
    report: dict[str, Any],
) -> None:
    st.divider()

    score = int(report.get("readiness_score", 0))
    score_label = _score_label(score)

    score_col, summary_col = st.columns([0.35, 0.65], gap="large")
    with score_col:
        st.metric("Readiness Score", f"{score}/100", score_label)
        st.progress(score / 100)
    with summary_col:
        st.markdown('<div class="section-title">Career Summary</div>', unsafe_allow_html=True)
        st.write(report.get("career_summary", "No career summary generated."))

    tab_report, tab_profile, tab_resume = st.tabs(
        ["Digital Twin Report", "Student Profile", "Extracted Resume Text"]
    )

    with tab_report:
        col_strengths, col_weaknesses, col_risks = st.columns(3, gap="large")
        with col_strengths:
            st.subheader("Strengths")
            _render_list(report.get("strengths", []))
        with col_weaknesses:
            st.subheader("Weaknesses")
            _render_list(report.get("weaknesses", []))
        with col_risks:
            st.subheader("Risk Areas")
            _render_list(report.get("risk_areas", []))

    with tab_profile:
        st.json(profile)

    with tab_resume:
        st.text_area(
            "Extracted text",
            value=resume_text,
            height=360,
            disabled=True,
            label_visibility="collapsed",
        )


def _render_list(items: list[str]) -> None:
    if not items:
        st.write("None")
        return

    for item in items:
        st.markdown(f"- {item}")


def _score_label(score: int) -> str:
    if score >= 80:
        return "High readiness"
    if score >= 65:
        return "Moderate readiness"
    if score >= 50:
        return "Partial readiness"
    return "Early stage"


if __name__ == "__main__":
    main()
