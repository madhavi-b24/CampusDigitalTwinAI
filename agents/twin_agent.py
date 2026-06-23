"""Digital twin report generation for Campus Digital Twin AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PROFILE_KEYS = ("skills", "projects", "education", "strengths", "weaknesses")


class TwinAgentError(Exception):
    """Base exception for digital twin report failures."""


class InvalidStudentProfileError(TwinAgentError):
    """Raised when the student profile JSON is malformed."""


@dataclass(frozen=True)
class StudentProfile:
    """Validated structured profile used to generate a digital twin report."""

    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, profile_json: dict[str, Any]) -> "StudentProfile":
        """Build a normalized profile from structured JSON."""
        if not isinstance(profile_json, dict):
            raise InvalidStudentProfileError("Student profile must be a JSON object.")

        normalized: dict[str, list[str]] = {}
        for key in PROFILE_KEYS:
            value = profile_json.get(key, [])
            if value is None:
                normalized[key] = []
                continue
            if not isinstance(value, list):
                raise InvalidStudentProfileError(f"Profile field '{key}' must be a list.")
            normalized[key] = [_clean_text_item(item) for item in value if _clean_text_item(item)]

        return cls(**normalized)


@dataclass(frozen=True)
class DigitalTwinReport:
    """Career readiness report produced from a student profile."""

    readiness_score: int
    strengths: list[str]
    weaknesses: list[str]
    career_summary: str
    risk_areas: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable data."""
        return {
            "readiness_score": self.readiness_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "career_summary": self.career_summary,
            "risk_areas": self.risk_areas,
        }

    def to_markdown(self) -> str:
        """Return a readable Digital Twin Report."""
        return "\n\n".join(
            [
                "# Digital Twin Report",
                f"## Readiness Score\n{self.readiness_score}/100",
                f"## Strengths\n{_format_bullets(self.strengths)}",
                f"## Weaknesses\n{_format_bullets(self.weaknesses)}",
                f"## Career Summary\n{self.career_summary}",
                f"## Risk Areas\n{_format_bullets(self.risk_areas)}",
            ]
        )


def create_digital_twin_report(profile_json: dict[str, Any]) -> dict[str, Any]:
    """Create a structured digital twin report from a student profile JSON."""
    profile = StudentProfile.from_json(profile_json)
    report = _build_report(profile)
    return report.to_dict()


def create_digital_twin_report_markdown(profile_json: dict[str, Any]) -> str:
    """Create a Markdown digital twin report from a student profile JSON."""
    profile = StudentProfile.from_json(profile_json)
    report = _build_report(profile)
    return report.to_markdown()


class TwinAgent:
    """Digital twin agent wrapper for Streamlit and future LangGraph integration."""

    def run(self, profile_json: dict[str, Any]) -> dict[str, Any]:
        """Create a structured Digital Twin Report."""
        return create_digital_twin_report(profile_json)

    def run_markdown(self, profile_json: dict[str, Any]) -> str:
        """Create a Markdown Digital Twin Report."""
        return create_digital_twin_report_markdown(profile_json)


def _build_report(profile: StudentProfile) -> DigitalTwinReport:
    readiness_score = _calculate_readiness_score(profile)
    strengths = _derive_strengths(profile)
    weaknesses = _derive_weaknesses(profile)
    risk_areas = _derive_risk_areas(profile, readiness_score)
    career_summary = _build_career_summary(profile, readiness_score)

    return DigitalTwinReport(
        readiness_score=readiness_score,
        strengths=strengths,
        weaknesses=weaknesses,
        career_summary=career_summary,
        risk_areas=risk_areas,
    )


def _calculate_readiness_score(profile: StudentProfile) -> int:
    score = 20

    skill_score = min(len(profile.skills), 8) * 5
    project_score = min(len(profile.projects), 4) * 8
    education_score = min(len(profile.education), 2) * 6
    strength_score = min(len(profile.strengths), 4) * 3
    weakness_penalty = min(len(profile.weaknesses), 6) * 4

    score += skill_score + project_score + education_score + strength_score - weakness_penalty

    if profile.skills and profile.projects:
        score += 10
    if profile.education:
        score += 5
    if not profile.projects:
        score -= 10
    if not profile.skills:
        score -= 15

    return max(0, min(100, score))


def _derive_strengths(profile: StudentProfile) -> list[str]:
    strengths = list(profile.strengths)

    if profile.skills:
        strengths.append(f"Demonstrates technical capability across {len(profile.skills)} skill areas.")
    if profile.projects:
        strengths.append(f"Shows applied learning through {len(profile.projects)} project(s).")
    if profile.education:
        strengths.append("Has a documented academic foundation.")

    return _dedupe_preserve_order(strengths) or ["No clear strengths were identified from the profile."]


def _derive_weaknesses(profile: StudentProfile) -> list[str]:
    weaknesses = list(profile.weaknesses)

    if not profile.skills:
        weaknesses.append("No skills are listed in the profile.")
    if not profile.projects:
        weaknesses.append("No project experience is listed.")
    if not profile.education:
        weaknesses.append("Education details are missing.")
    if len(profile.skills) < 3:
        weaknesses.append("Skill breadth appears limited.")

    return _dedupe_preserve_order(weaknesses) or ["No major weaknesses were identified from the profile."]


def _derive_risk_areas(profile: StudentProfile, readiness_score: int) -> list[str]:
    risk_areas: list[str] = []

    if readiness_score < 50:
        risk_areas.append("Low career readiness score may reduce placement competitiveness.")
    if not profile.projects:
        risk_areas.append("Limited evidence of practical implementation experience.")
    if len(profile.skills) < 3:
        risk_areas.append("Insufficient technical skill depth or breadth.")
    if profile.weaknesses:
        risk_areas.append("Identified weaknesses should be addressed before applications.")
    if not profile.education:
        risk_areas.append("Missing academic context may weaken profile evaluation.")

    return _dedupe_preserve_order(risk_areas) or ["No critical risk areas were identified."]


def _build_career_summary(profile: StudentProfile, readiness_score: int) -> str:
    level = _readiness_level(readiness_score)
    skill_summary = _join_preview(profile.skills, "skills")
    project_summary = _join_preview(profile.projects, "projects")
    education_summary = _join_preview(profile.education, "education")

    return (
        f"The student currently appears {level} for career opportunities with a "
        f"readiness score of {readiness_score}/100. The profile shows {skill_summary}, "
        f"{project_summary}, and {education_summary}. The next best focus is to improve "
        "the weakest evidence areas and strengthen the profile with measurable outcomes."
    )


def _readiness_level(score: int) -> str:
    if score >= 80:
        return "highly ready"
    if score >= 65:
        return "moderately ready"
    if score >= 50:
        return "partially ready"
    return "early-stage"


def _join_preview(items: list[str], label: str) -> str:
    if not items:
        return f"no documented {label}"

    preview = ", ".join(items[:3])
    if len(items) > 3:
        preview = f"{preview}, and more"
    return f"{label} including {preview}"


def _clean_text_item(item: Any) -> str:
    if item is None:
        return ""
    return " ".join(str(item).split())


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []

    for item in items:
        cleaned = _clean_text_item(item)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)

    return deduped


def _format_bullets(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)
