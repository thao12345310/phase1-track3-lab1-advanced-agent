"""
Real Ollama LLM runtime — replaces mock_runtime.py.

Calls a local Ollama instance (default: http://localhost:11434) to generate
answers, evaluate them, and produce reflections for the Reflexion agent loop.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Failure-mode classification heuristics (post-hoc, not from mock)
FAILURE_MODE_BY_QID: dict[str, str] = {}  # populated dynamically

# ---------------------------------------------------------------------------
# Trace log — records every LLM call for observability
# ---------------------------------------------------------------------------
from datetime import datetime, timezone

_TRACE_LOG: list[dict[str, Any]] = []
_call_counter = 0


def get_trace_log() -> list[dict[str, Any]]:
    """Return the full trace log of all LLM calls."""
    return _TRACE_LOG


def clear_trace_log() -> None:
    """Clear the trace log."""
    global _call_counter
    _TRACE_LOG.clear()
    _call_counter = 0


def _chat(
    system: str,
    user: str,
    *,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Send a chat-completion request to the Ollama REST API.

    Returns a dict with keys:
        content   – the assistant's text reply
        tokens    – total tokens used (prompt + completion)
        latency_ms – wall-clock time in milliseconds
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    global _call_counter
    _call_counter += 1
    call_id = _call_counter

    t0 = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    resp.raise_for_status()
    data = resp.json()

    content = data.get("message", {}).get("content", "")

    # Real token counts from Ollama response
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    total_tokens = prompt_tokens + completion_tokens

    # ---- Record trace ----
    _TRACE_LOG.append({
        "call_id": call_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system[:200] + ("..." if len(system) > 200 else ""),
        "user_prompt": user,
        "response": content.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
    })

    return {
        "content": content.strip(),
        "tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Actor  – answers the multi-hop question
# ---------------------------------------------------------------------------
def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int]:
    """Return (answer_text, token_count, latency_ms)."""
    context_text = "\n\n".join(
        f"### {c.title}\n{c.text}" for c in example.context
    )

    reflection_block = ""
    if reflection_memory:
        reflection_block = (
            "\n\n--- Previous Reflections ---\n"
            + "\n".join(f"- {r}" for r in reflection_memory)
            + "\n--- End Reflections ---\n"
        )

    user_prompt = (
        f"Question: {example.question}\n\n"
        f"Context:\n{context_text}"
        f"{reflection_block}\n\n"
        f"Attempt: {attempt_id}/{3 if agent_type == 'reflexion' else 1}\n"
        "Provide ONLY the final answer — no explanation."
    )

    result = _chat(ACTOR_SYSTEM, user_prompt)
    answer = _clean_answer(result["content"])
    return answer, result["tokens"], result["latency_ms"]


def _clean_answer(raw: str) -> str:
    """Strip markdown, quotes, prefixes like 'Answer:' etc."""
    text = raw.strip().strip('"').strip("'")
    # Remove common prefixes
    for prefix in ("Answer:", "Final Answer:", "The answer is", "A:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip().strip('"').strip("'")
    # Remove trailing period
    text = text.rstrip(".")
    return text.strip()


# ---------------------------------------------------------------------------
# Evaluator – judges correctness (score 0 or 1)
# ---------------------------------------------------------------------------
def evaluator(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, int]:
    """Return (JudgeResult, token_count, latency_ms)."""
    # Quick exact-match shortcut (saves an LLM call if clearly correct)
    if normalize_answer(answer) == normalize_answer(example.gold_answer):
        return (
            JudgeResult(
                score=1,
                reason="Exact match after normalization.",
                missing_evidence=[],
                spurious_claims=[],
            ),
            0,
            0,
        )

    user_prompt = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n\n"
        "Respond ONLY with a JSON object — no markdown fences, no explanation:\n"
        '{"score": 0 or 1, "reason": "...", "missing_evidence": [...], "spurious_claims": [...]}'
    )

    result = _chat(EVALUATOR_SYSTEM, user_prompt, temperature=0.0)
    judge = _parse_judge(result["content"], answer)
    return judge, result["tokens"], result["latency_ms"]


def _parse_judge(raw: str, answer: str) -> JudgeResult:
    """Best-effort parse of the evaluator's JSON reply."""
    # Try to extract JSON from possible markdown fences
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return JudgeResult(
                score=int(data.get("score", 0)),
                reason=str(data.get("reason", "No reason provided.")),
                missing_evidence=data.get("missing_evidence", []),
                spurious_claims=data.get("spurious_claims", []),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: treat as incorrect
    return JudgeResult(
        score=0,
        reason=f"Evaluator returned unparseable response. Raw: {raw[:200]}",
        missing_evidence=[],
        spurious_claims=[answer],
    )


# ---------------------------------------------------------------------------
# Reflector – analyses the failure and proposes a new strategy
# ---------------------------------------------------------------------------
def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, int]:
    """Return (ReflectionEntry, token_count, latency_ms)."""
    context_text = "\n\n".join(
        f"### {c.title}\n{c.text}" for c in example.context
    )

    user_prompt = (
        f"Question: {example.question}\n"
        f"Context:\n{context_text}\n\n"
        f"Attempt #{attempt_id} was scored {judge.score}/1.\n"
        f"Evaluator reason: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n"
        f"Spurious claims: {judge.spurious_claims}\n\n"
        "Respond ONLY with a JSON object — no markdown fences:\n"
        '{"failure_reason": "...", "lesson": "...", "next_strategy": "..."}'
    )

    result = _chat(REFLECTOR_SYSTEM, user_prompt, temperature=0.4)
    entry = _parse_reflection(result["content"], attempt_id, judge)
    return entry, result["tokens"], result["latency_ms"]


def _parse_reflection(raw: str, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    """Best-effort parse of the reflector's JSON reply."""
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=str(data.get("failure_reason", judge.reason)),
                lesson=str(data.get("lesson", "Review multi-hop reasoning steps.")),
                next_strategy=str(data.get("next_strategy", "Re-read context carefully.")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback
    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="Could not parse reflector output — defaulting to generic lesson.",
        next_strategy="Re-read context and verify each reasoning hop.",
    )


# ---------------------------------------------------------------------------
# Failure-mode classifier (heuristic post-hoc)
# ---------------------------------------------------------------------------
def classify_failure(
    example: QAExample,
    predicted: str,
    traces: list,
) -> str:
    """Classify the failure mode based on the predicted answer and trace."""
    gold_norm = normalize_answer(example.gold_answer)
    pred_norm = normalize_answer(predicted)

    if gold_norm == pred_norm:
        return "none"

    # Check if the model kept giving the same wrong answer (looping)
    if len(traces) >= 3:
        answers = [normalize_answer(t.answer) for t in traces]
        if len(set(answers)) == 1:
            return "looping"

    # Check if reflection made things worse (overfit)
    if len(traces) >= 2:
        first_score = traces[0].score
        last_score = traces[-1].score
        if first_score == 0 and last_score == 0:
            first_ans = normalize_answer(traces[0].answer)
            last_ans = normalize_answer(traces[-1].answer)
            if first_ans != last_ans and first_ans in gold_norm and last_ans not in gold_norm:
                return "reflection_overfit"

    # Check for partial multi-hop
    context_entities = set()
    for c in example.context:
        for word in c.text.split():
            if word[0:1].isupper() and len(word) > 2:
                context_entities.add(word.lower().strip(".,!?"))

    if pred_norm in " ".join(c.text.lower() for c in example.context[:1]):
        return "incomplete_multi_hop"

    return "wrong_final_answer"
