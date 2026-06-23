"""Career path simulation agent for Campus Digital Twin AI."""

from __future__ import annotations

import json
from typing import Any

from agents.profile_agent import DEFAULT_MODEL, _resolve_api_key


def simulate_career_path(
    student_profile: dict,
    career_goal: str,
    daily_hours: int,
) -> str:
    """Simulate a student's career path and return a clean Markdown report."""
    try:
        cleaned_profile = _validate_student_profile(student_profile)
        cleaned_goal = _validate_career_goal(career_goal)
        cleaned_daily_hours = _validate_daily_hours(daily_hours)
        api_key = _resolve_api_key()
        prompt = _build_simulation_prompt(
            student_profile=cleaned_profile,
            career_goal=cleaned_goal,
            daily_hours=cleaned_daily_hours,
        )

        report = _generate_markdown_with_gemini(
            prompt=prompt,
            model_name=DEFAULT_MODEL,
            api_key=api_key,
        )
        return _clean_markdown_response(report)
    except Exception as exc:
        return _build_error_report(career_goal, student_profile, daily_hours, exc)


def _validate_student_profile(student_profile: dict) -> dict[str, Any]:
    if not isinstance(student_profile, dict):
        raise ValueError("student_profile must be a dictionary.")
    return student_profile


def _validate_career_goal(career_goal: str) -> str:
    if not isinstance(career_goal, str):
        raise ValueError("career_goal must be a string.")

    cleaned = career_goal.strip()
    if not cleaned:
        raise ValueError("career_goal cannot be empty.")
    return cleaned


def _validate_daily_hours(daily_hours: int) -> int:
    if not isinstance(daily_hours, int):
        raise ValueError("daily_hours must be an integer.")
    if daily_hours <= 0:
        raise ValueError("daily_hours must be greater than 0.")
    return daily_hours


def _build_simulation_prompt(
    student_profile: dict[str, Any],
    career_goal: str,
    daily_hours: int,
) -> str:
    profile_json = json.dumps(student_profile, indent=2, ensure_ascii=True)

    return f"""
You are a senior career simulation agent for Campus Digital Twin AI.

Task:
Simulate whether this student can reach the stated career goal based on the
current profile and available study time.

Return only clean Markdown text. Do not include JSON, code fences, comments, or
extra sections.

Use exactly these Markdown section headings in this order:

# Career Simulation Report
## Career Goal
## Current Student Status
## Skill Gap Analysis
## 3 Month Forecast
## 6 Month Forecast
## 12 Month Forecast
## Success Probability (0-100)
## Recommended Projects
## Recommended Certifications
## Risks
## Final Advice

Rules:
- Keep the report practical, specific, and student-friendly.
- Base the analysis on the provided profile and career goal.
- Account for the student's available study time of {daily_hours} hour(s) per day.
- Make the success probability a single number from 0 to 100 with a short reason.
- Use bullets where they improve readability.
- Do not invent credentials, experience, or completed projects.
- If profile evidence is missing, state the uncertainty clearly.

Student profile:
{profile_json}

Career goal:
{career_goal}
""".strip()


def _generate_markdown_with_gemini(prompt: str, model_name: str, api_key: str) -> str:
    """Generate Markdown with the current Gemini SDK, falling back to legacy SDK."""
    try:
        return _generate_with_google_genai(prompt, model_name, api_key)
    except ModuleNotFoundError:
        return _generate_with_google_generativeai(prompt, model_name, api_key)
    except ImportError:
        return _generate_with_google_generativeai(prompt, model_name, api_key)


def _generate_with_google_genai(prompt: str, model_name: str, api_key: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except (ModuleNotFoundError, ImportError) as exc:
        raise exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _generate_with_google_generativeai(prompt: str, model_name: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Gemini SDK not installed. Install google-genai or google-generativeai."
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
        },
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _clean_markdown_response(report: str) -> str:
    cleaned = report.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _build_error_report(
    career_goal: Any,
    student_profile: Any,
    daily_hours: Any,
    error: Exception,
) -> str:
    goal = career_goal.strip() if isinstance(career_goal, str) and career_goal.strip() else "Not provided"
    hours = daily_hours if isinstance(daily_hours, int) and daily_hours > 0 else "Not provided"
    profile_status = (
        "Profile data was provided."
        if isinstance(student_profile, dict) and student_profile
        else "No usable profile data was provided."
    )

    return "\n\n".join(
        [
            "# Career Simulation Report",
            f"## Career Goal\n{goal}",
            f"## Current Student Status\n{profile_status}",
            "## Skill Gap Analysis\nUnable to generate analysis because Gemini did not complete successfully.",
            f"## 3 Month Forecast\nUnavailable. Planned daily study time: {hours} hour(s).",
            "## 6 Month Forecast\nUnavailable.",
            "## 12 Month Forecast\nUnavailable.",
            "## Success Probability (0-100)\nUnavailable.",
            "## Recommended Projects\n- Unable to recommend projects until the simulation can run.",
            "## Recommended Certifications\n- Unable to recommend certifications until the simulation can run.",
            f"## Risks\n- Gemini error: {error}",
            "## Final Advice\nCheck the Gemini API key, SDK installation, and input values, then run the simulation again.",
        ]
    )
