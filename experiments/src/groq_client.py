"""
experiments/src/groq_client.py
Thin wrapper around the Groq SDK:
  - JSON-mode calls with robust parsing
  - Exponential backoff via tenacity
  - Token counting, cost calculation, latency
  - Global in-process rate limiter (token bucket, per-model)

Public API:
    from src.groq_client import call_model, CallResult
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from groq import Groq, RateLimitError, APIStatusError
except ImportError as e:
    raise ImportError("groq package not installed. Run: pip install groq") from e

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import CFG, groq_api_key

# ─────────────────────────────────────────────────────────────────────────────
# Cost calculation
# ─────────────────────────────────────────────────────────────────────────────

_MTOK = 1_000_000


def _get_prices(model_alias: str) -> tuple[float | None, float | None]:
    models = CFG.get("models", {})
    mcfg = models.get(model_alias, {})
    return mcfg.get("price_in_per_mtok"), mcfg.get("price_out_per_mtok")


def _compute_cost(alias: str, in_tok: int, out_tok: int) -> float | None:
    pin, pout = _get_prices(alias)
    if pin is None or pout is None:
        return None
    return (in_tok * pin + out_tok * pout) / _MTOK


# ─────────────────────────────────────────────────────────────────────────────
# Simple token-bucket rate limiter (per model, in-process only)
# ─────────────────────────────────────────────────────────────────────────────

class _TokenBucket:
    """Leaky-bucket rate limiter. Thread-safe."""

    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


_rpm = CFG.get("rate_limit", {}).get("requests_per_minute", 30)
_buckets: dict[str, _TokenBucket] = {}
_bucket_lock = threading.Lock()


def _get_bucket(model_id: str) -> _TokenBucket:
    with _bucket_lock:
        if model_id not in _buckets:
            _buckets[model_id] = _TokenBucket(_rpm)
        return _buckets[model_id]


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallResult:
    model_alias: str
    model_id: str
    raw_content: str
    parsed: dict[str, Any] | None          # None if parse failed
    parse_ok: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    latency_s: float
    error: str | None = None               # set on API error (after retries)


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing (robust — never crash)
# ─────────────────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_json(text: str) -> tuple[dict | None, bool]:
    """Try to extract the first JSON object from text. Returns (dict|None, ok)."""
    # Direct parse
    try:
        return json.loads(text.strip()), True
    except json.JSONDecodeError:
        pass
    # Extract first {...} block (handles markdown code fences etc.)
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group()), True
        except json.JSONDecodeError:
            pass
    return None, False


# ─────────────────────────────────────────────────────────────────────────────
# Core call (with retry)
# ─────────────────────────────────────────────────────────────────────────────

_rl_cfg = CFG.get("rate_limit", {})
_MAX_ATTEMPTS = _rl_cfg.get("retry_max_attempts", 6)
_WAIT_MIN = _rl_cfg.get("retry_wait_min_s", 1)
_WAIT_MAX = _rl_cfg.get("retry_wait_max_s", 60)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=groq_api_key())
    return _client


def _alias_to_id(alias: str) -> str:
    models = CFG.get("models", {})
    if alias in models:
        return models[alias]["id"]
    # Assume it's already a model ID
    return alias


@retry(
    retry=retry_if_exception_type((RateLimitError, Exception)),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential(min=_WAIT_MIN, max=_WAIT_MAX),
    reraise=True,
)
def _api_call(client: Groq, model_id: str, system: str, user: str, max_tokens: int) -> Any:
    _get_bucket(model_id).acquire()
    return client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )


def call_model(
    alias: str,
    system: str,
    user: str,
    max_tokens: int | None = None,
) -> CallResult:
    """
    Make one JSON-mode call to the model identified by `alias`.
    Never raises — errors are captured in CallResult.error.
    """
    model_id = _alias_to_id(alias)
    if max_tokens is None:
        max_tokens = CFG.get("llm", {}).get("max_tokens", 256)

    client = _get_client()
    t0 = time.perf_counter()

    try:
        resp = _api_call(client, model_id, system, user, max_tokens)
        latency = time.perf_counter() - t0
        raw = resp.choices[0].message.content or ""
        parsed, ok = _parse_json(raw)
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        cost = _compute_cost(alias, in_tok, out_tok)
        return CallResult(
            model_alias=alias,
            model_id=model_id,
            raw_content=raw,
            parsed=parsed,
            parse_ok=ok,
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            cost_usd=cost,
            latency_s=latency,
            error=None,
        )
    except Exception as exc:
        latency = time.perf_counter() - t0
        return CallResult(
            model_alias=alias,
            model_id=model_id,
            raw_content="",
            parsed=None,
            parse_ok=False,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=None,
            latency_s=latency,
            error=str(exc),
        )
