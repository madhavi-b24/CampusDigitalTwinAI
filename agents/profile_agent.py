"""Student profile extraction agent for Campus Digital Twin AI.

This module converts raw resume text into a structured digital student profile.
It intentionally keeps the public API small so it can be called from Streamlit,
tests, or a future LangGraph node without coupling the agent to any UI layer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


PROFILE_KEYS = ("skills", "projects", "education", "strengths", "weaknesses")
DEFAULT_MODEL = "gemini-2.5-flash"


class ProfileAgentError(Exception):
    """Base exception for profile agent failures."""


class GeminiConfigurationError(ProfileAgentError):
    """Raised when Gemini credentials or SDK dependencies are missing."""


class ProfileExtractionError(ProfileAgentError):
    """Raised when Gemini returns invalid or unusable profile data."""


@dataclass(frozen=True)
class StudentProfile:
    """Normalized student profile returned by the profile agent."""

    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StudentProfile":
        """Create a profile from a dictionary while enforcing the output shape."""
        if not isinstance(data, dict):
            raise ProfileExtractionError("Profile response must be a JSON object.")

        normalized: dict[str, list[str]] = {}
        for key in PROFILE_KEYS:
            value = data.get(key, [])
            if value is None:
                normalized[key] = []
                continue
            if not isinstance(value, list):
                raise ProfileExtractionError(f"Profile field '{key}' must be a list.")
            normalized[key] = [_clean_list_item(item) for item in value if _clean_list_item(item)]

        return cls(**normalized)

    def to_dict(self) -> dict[str, list[str]]:
        """Return the profile as JSON-serializable data."""
        return {
            "skills": self.skills,
            "projects": self.projects,
            "education": self.education,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


def create_student_profile(
    resume_text: str,
    *,
    model_name: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[str, list[str]]:
    """Extract a structured student profile from resume text.

    Args:
        resume_text: Raw text extracted from a student's resume.
        model_name: Gemini model ID to use.
        api_key: Optional Gemini API key. If omitted, GEMINI_API_KEY or
            GOOGLE_API_KEY is read from the environment.

    Returns:
        A dictionary with skills, projects, education, strengths, and weaknesses.

    Raises:
        ValueError: If resume text is empty or invalid.
        GeminiConfigurationError: If credentials or SDK setup is missing.
        ProfileExtractionError: If the model response cannot be parsed.
    """
    cleaned_resume = _validate_resume_text(resume_text)
    resolved_api_key = _resolve_api_key(api_key)
    prompt = _build_profile_prompt(cleaned_resume)

    raw_response = _generate_with_gemini(
        prompt=prompt,
        model_name=model_name,
        api_key=resolved_api_key,
    )
    profile_data = _parse_json_response(raw_response)
    return StudentProfile.from_mapping(profile_data).to_dict()


class ProfileAgent:
    """Small wrapper class for dependency injection and future LangGraph use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key

    def run(self, resume_text: str) -> dict[str, list[str]]:
        """Create a student profile from resume text."""
        return create_student_profile(
            resume_text,
            model_name=self.model_name,
            api_key=self.api_key,
        )


def _validate_resume_text(resume_text: str) -> str:
    if not isinstance(resume_text, str):
        raise ValueError("resume_text must be a string.")

    cleaned = resume_text.strip()
    if not cleaned:
        raise ValueError("resume_text cannot be empty.")

    if len(cleaned) < 50:
        raise ValueError("resume_text is too short to extract a reliable profile.")

    return cleaned


def _resolve_api_key(api_key: str | None = None) -> str:
    resolved = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved:
        raise GeminiConfigurationError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return resolved


def _build_profile_prompt(resume_text: str) -> str:
    return f"""
You are a senior career intelligence agent for Campus Digital Twin AI.

Task:
Extract a student's digital career profile from the resume text.

Return only valid JSON with exactly these keys:
{{
  "skills": [],
  "projects": [],
  "education": [],
  "strengths": [],
  "weaknesses": []
}}

Rules:
- Each value must be an array of concise strings.
- Include only information supported by the resume.
- Infer strengths from evidence in the resume.
- Infer weaknesses conservatively from missing or shallow evidence.
- Do not include markdown, comments, explanations, or extra keys.
- If a field has no evidence, return an empty array.

Resume text:
\"\"\"{resume_text}\"\"\"
""".strip()


def _generate_with_gemini(prompt: str, model_name: str, api_key: str) -> str:
    """Generate JSON with the current Gemini SDK, falling back to the legacy SDK."""
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

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
    except Exception as exc:
        raise ProfileExtractionError(f"Gemini request failed: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise ProfileExtractionError("Gemini returned an empty response.")
    return text


def _generate_with_google_generativeai(prompt: str, model_name: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except (ModuleNotFoundError, ImportError) as exc:
        raise GeminiConfigurationError(
            "Gemini SDK not installed. Install google-genai or google-generativeai."
        ) from exc

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        )
    except Exception as exc:
        raise ProfileExtractionError(f"Gemini request failed: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise ProfileExtractionError("Gemini returned an empty response.")
    return text


def _parse_json_response(raw_response: str) -> dict[str, Any]:
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ProfileExtractionError("Cannot parse an empty Gemini response.")

    cleaned = _strip_markdown_fence(raw_response.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not json_match:
            raise ProfileExtractionError("Gemini response did not contain JSON.")
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError as exc:
            raise ProfileExtractionError("Gemini response contained invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ProfileExtractionError("Gemini JSON response must be an object.")

    return parsed


def _strip_markdown_fence(text: str) -> str:
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _clean_list_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return " ".join(item.split())
    return " ".join(str(item).split())
