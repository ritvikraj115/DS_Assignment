from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    generation_model: str = "gemini-3.1-flash-lite"
    judge_model: str = "gemini-3.5-flash-lite"
    generation_fallback_models: tuple[str, ...] = ("gemini-3.5-flash-lite",)
    judge_fallback_models: tuple[str, ...] = ("gemini-3.1-flash-lite",)
    gemini_api_key: str | None = None
    gemini_timeout_ms: int = 30000
    top_k: int = 5
    max_retrieval_retries: int = 2
    minimum_dimension_score: int = 70
    minimum_overall_score: int = 75
    max_history_turns: int = 8
    feedback_positive_similarity_threshold: float = 0.72
    feedback_negative_similarity_threshold: float = 0.80

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        def model_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            configured = os.getenv(name)
            values = configured.split(",") if configured is not None else default
            return tuple(dict.fromkeys(item.strip() for item in values if item.strip()))

        return cls(
            generation_model=os.getenv("GENERATION_MODEL", cls.generation_model),
            judge_model=os.getenv("JUDGE_MODEL", cls.judge_model),
            generation_fallback_models=model_list("GENERATION_FALLBACK_MODELS", cls.generation_fallback_models),
            judge_fallback_models=model_list("JUDGE_FALLBACK_MODELS", cls.judge_fallback_models),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
            gemini_timeout_ms=int(os.getenv("GEMINI_TIMEOUT_MS", "30000")),
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            max_retrieval_retries=int(os.getenv("MAX_RETRIEVAL_RETRIES", "2")),
            minimum_dimension_score=int(os.getenv("MINIMUM_DIMENSION_SCORE", "70")),
            minimum_overall_score=int(os.getenv("MINIMUM_OVERALL_SCORE", "75")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "8")),
            feedback_positive_similarity_threshold=float(os.getenv("FEEDBACK_POSITIVE_SIMILARITY_THRESHOLD", "0.72")),
            feedback_negative_similarity_threshold=float(os.getenv("FEEDBACK_NEGATIVE_SIMILARITY_THRESHOLD", "0.80")),
        )

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K must be at least 1")
        if self.gemini_timeout_ms < 1:
            raise ValueError("GEMINI_TIMEOUT_MS must be positive")
        if not 0 <= self.max_retrieval_retries <= 2:
            raise ValueError("MAX_RETRIEVAL_RETRIES must be between 0 and 2")
        if not 0 <= self.minimum_dimension_score <= 100 or not 0 <= self.minimum_overall_score <= 100:
            raise ValueError("Evaluation thresholds must be between 0 and 100")
        if not 0.0 <= self.feedback_positive_similarity_threshold <= 1.0:
            raise ValueError("FEEDBACK_POSITIVE_SIMILARITY_THRESHOLD must be between 0 and 1")
        if not 0.0 <= self.feedback_negative_similarity_threshold <= 1.0:
            raise ValueError("FEEDBACK_NEGATIVE_SIMILARITY_THRESHOLD must be between 0 and 1")
