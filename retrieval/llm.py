from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Protocol
from collections.abc import Iterator


class ModelError(RuntimeError):
    """A Gemini request failed or returned invalid structured output."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


log = logging.getLogger(__name__)


def _provider_error_category(exc: Exception) -> tuple[str, int | None]:
    message = str(exc)
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    if status is None:
        match = re.search(r"\b(429|500|502|503|504)\b", message)
        status = int(match.group(1)) if match else None
    if status == 429 or "resource_exhausted" in message.casefold():
        return "rate limit", status
    if status in {500, 502, 503, 504} or "unavailable" in message.casefold():
        return "temporary service outage", status
    return "request failure", status


class ModelClient(Protocol):
    def generate(self, model: str, prompt: str, *, json_mode: bool = False) -> str: ...
    def generate_stream(self, model: str, prompt: str, *, on_fallback=None) -> Iterator[str]: ...


class GeminiClient:
    def __init__(self, api_key: str | None = None, timeout_ms: int = 30000,
                 model_fallbacks: dict[str, tuple[str, ...]] | None = None):
        if not api_key:
            raise ModelError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ModelError("Gemini calls require google-genai; install requirements.txt") from exc
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
        self.types = types
        self.model_fallbacks = model_fallbacks or {}

    def _candidate_models(self, model: str) -> list[str]:
        return list(dict.fromkeys([model, *self.model_fallbacks.get(model, ())]))

    @staticmethod
    def _fallback_allowed(error: ModelError, current: str, candidates: list[str]) -> bool:
        return error.retryable and current != candidates[-1]

    def generate(self, model: str, prompt: str, *, json_mode: bool = False) -> str:
        candidates = self._candidate_models(model)
        for index, candidate in enumerate(candidates):
            try:
                response = self._generate_once(candidate, prompt, json_mode=json_mode)
                if candidate != model:
                    log.info("gemini_model_fallback_succeeded primary=%s fallback=%s", model, candidate)
                return response
            except ModelError as exc:
                if not self._fallback_allowed(exc, candidate, candidates):
                    raise
                next_model = candidates[index + 1]
                log.warning("gemini_model_fallback primary=%s failed=%s fallback=%s category=%s",
                            model, candidate, next_model, str(exc))
        raise ModelError("Gemini model chain failed")

    def _generate_once(self, model: str, prompt: str, *, json_mode: bool = False) -> str:
        started = perf_counter()
        log.info("gemini_request_started model=%s json_mode=%s prompt_chars=%s", model, json_mode, len(prompt))
        try:
            config = self.types.GenerateContentConfig(
                response_mime_type="application/json" if json_mode else "text/plain",
                temperature=0.1 if json_mode else 0.2,
            )
            response = self.client.models.generate_content(model=model, contents=prompt, config=config)
            text = getattr(response, "text", None)
            if not text:
                raise ModelError("Gemini returned an empty response", retryable=True)
            if json_mode:
                try:
                    parse_json_response(text)
                except ModelError as exc:
                    raise ModelError("Gemini returned invalid structured output", retryable=True) from exc
            log.info("gemini_request_completed model=%s json_mode=%s response_chars=%s duration_ms=%.1f",
                     model, json_mode, len(text), (perf_counter() - started) * 1000)
            return text.strip()
        except ModelError:
            log.exception("gemini_request_failed model=%s duration_ms=%.1f", model, (perf_counter() - started) * 1000)
            raise
        except Exception as exc:
            # Keep provider details out of client responses; transient outages need no stack dump.
            category, status = _provider_error_category(exc)
            if category == "temporary service outage":
                log.warning("gemini_request_failed model=%s category=%s provider_status=%s duration_ms=%.1f",
                            model, category, status or "unknown", (perf_counter() - started) * 1000)
            else:
                log.exception("gemini_request_failed model=%s category=%s provider_status=%s duration_ms=%.1f",
                              model, category, status or "unknown", (perf_counter() - started) * 1000)
            raise ModelError(f"Gemini {category}: {type(exc).__name__}",
                             retryable=category in {"rate limit", "temporary service outage"}) from exc

    def generate_stream(self, model: str, prompt: str, *, on_fallback=None) -> Iterator[str]:
        candidates = self._candidate_models(model)
        for index, candidate in enumerate(candidates):
            try:
                yield from self._generate_stream_once(candidate, prompt)
                if candidate != model:
                    log.info("gemini_stream_fallback_succeeded primary=%s fallback=%s", model, candidate)
                return
            except ModelError as exc:
                if not self._fallback_allowed(exc, candidate, candidates):
                    raise
                if on_fallback:
                    on_fallback()
                log.warning("gemini_stream_fallback primary=%s failed=%s fallback=%s category=%s",
                            model, candidate, candidates[index + 1], str(exc))

    def _generate_stream_once(self, model: str, prompt: str) -> Iterator[str]:
        started = perf_counter()
        log.info("gemini_stream_started model=%s prompt_chars=%s", model, len(prompt))
        try:
            config = self.types.GenerateContentConfig(response_mime_type="text/plain", temperature=0.2)
            received_text = False
            for response in self.client.models.generate_content_stream(model=model, contents=prompt, config=config):
                text = getattr(response, "text", None)
                if text:
                    received_text = True
                    yield text
            if not received_text:
                raise ModelError("Gemini returned an empty stream", retryable=True)
            log.info("gemini_stream_completed model=%s duration_ms=%.1f", model, (perf_counter() - started) * 1000)
        except ModelError:
            raise
        except Exception as exc:
            category, status = _provider_error_category(exc)
            if category == "temporary service outage":
                log.warning("gemini_stream_failed model=%s category=%s provider_status=%s duration_ms=%.1f",
                            model, category, status or "unknown", (perf_counter() - started) * 1000)
            else:
                log.exception("gemini_stream_failed model=%s category=%s provider_status=%s duration_ms=%.1f",
                              model, category, status or "unknown", (perf_counter() - started) * 1000)
            raise ModelError(f"Gemini {category}: {type(exc).__name__}",
                             retryable=category in {"rate limit", "temporary service outage"}) from exc


def parse_json_response(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ModelError("Gemini returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise ModelError("Gemini JSON response must be an object")
    return value
