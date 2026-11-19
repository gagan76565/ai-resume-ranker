"""TalentLens scoring: explainable, lightweight resume/job-description matching.

Public API::

    from scoring import ScoringConfig, rank_candidates, score_resume

The package is pure Python plus scikit-learn; it has no Flask dependency and
can be used from scripts or notebooks.
"""

from .engine import (
    DEFAULT_WEIGHTS,
    SCORING_VERSION,
    ScoringConfig,
    config_summary,
    explanation_text,
    label_for,
    rank_candidates,
    required_skills,
    score_resume,
)
from .skills import load_taxonomy

__all__ = [
    "DEFAULT_WEIGHTS",
    "SCORING_VERSION",
    "ScoringConfig",
    "config_summary",
    "explanation_text",
    "label_for",
    "load_taxonomy",
    "rank_candidates",
    "required_skills",
    "score_resume",
]
