"""
NextGen Tutor AI – Personalization Engine
==========================================
Maintains a lightweight learner model (profile + session memory) and
produces dynamic instructions injected into the tutor system prompt.

Educators can customise:
  - skill-level thresholds in ``get_personalization_instructions``
  - reinforcement logic in ``_reinforcement_directive``
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LearnerProfile:
    """Static learner information set during onboarding."""
    name: str = "Learner"
    course: str = "General"
    skill_level: str = "Intermediate"  # Beginner | Intermediate | Advanced
    goals: str = "Understand the material"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "course": self.course,
            "skill_level": self.skill_level,
            "goals": self.goals,
        }


@dataclass
class SessionMemory:
    """Tracks in-session learner behaviour for adaptive tutoring."""
    concepts_asked: List[str] = field(default_factory=list)
    misunderstanding_flags: Dict[str, int] = field(default_factory=dict)
    quiz_scores: List[Dict] = field(default_factory=list)  # {topic, score, timestamp}
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    def record_concept(self, concept: str) -> None:
        """Note that the learner asked about a concept."""
        if concept not in self.concepts_asked:
            self.concepts_asked.append(concept)

    def flag_misunderstanding(self, concept: str) -> None:
        """Increment misunderstanding counter for reinforcement logic."""
        self.misunderstanding_flags[concept] = (
            self.misunderstanding_flags.get(concept, 0) + 1
        )

    def record_quiz(self, topic: str, score: float, timestamp: str) -> None:
        self.quiz_scores.append(
            {"topic": topic, "score": score, "timestamp": timestamp}
        )

    def get_weak_concepts(self, threshold: int = 2) -> List[str]:
        """Return concepts the learner has struggled with repeatedly."""
        return [
            c for c, count in self.misunderstanding_flags.items()
            if count >= threshold
        ]

    def add_message(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def get_recent_topic(self) -> Optional[str]:
        """Return the most recently discussed concept, if any."""
        return self.concepts_asked[-1] if self.concepts_asked else None


# ── Personalisation Directives ───────────────────────────────────────────────

def get_personalization_instructions(
    profile: LearnerProfile,
    memory: SessionMemory,
) -> str:
    """Generate dynamic prompt instructions based on learner state."""
    parts: list[str] = []

    # Skill-level adaptation
    level = profile.skill_level.lower()
    if level == "beginner":
        parts.append(
            "The learner is a BEGINNER. Use simple language, real-world analogies, "
            "and short illustrative examples. Avoid jargon unless you define it first. "
            "Use bullet points and step-by-step walkthroughs."
        )
    elif level == "advanced":
        parts.append(
            "The learner is ADVANCED. Be concise and formal. Include edge cases, "
            "deeper reasoning, caveats, and references to related advanced topics. "
            "Skip overly basic definitions."
        )
    else:
        parts.append(
            "The learner is at an INTERMEDIATE level. Balance clarity with depth. "
            "Provide explanations with moderate detail and one or two examples."
        )

    # Reinforcement for weak concepts
    weak = memory.get_weak_concepts()
    if weak:
        parts.append(
            f"The learner has struggled with: {', '.join(weak)}. "
            "Provide a brief, targeted recap of these concepts at the end of your "
            "answer if they are related to the current question."
        )

    # Goal awareness
    if profile.goals and profile.goals != "Understand the material":
        parts.append(f"The learner's stated goal is: {profile.goals}.")

    return "\n".join(parts)
