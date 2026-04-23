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

from .prompts import (
    ACTOR_SYSTEM, ACTOR_SYSTEM_WITH_COT, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM,
    LATS_ACTOR_SYSTEM, LATS_SELECTOR_SYSTEM,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .structured_evaluator import StructuredEvalResult, structured_evaluate
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
    previous_answers: list[str] | None = None,
) -> tuple[str, int, int]:
    """Return (answer_text, token_count, latency_ms).

    Improvements over v1:
    - Uses CoT prompt for retry attempts (attempt >= 2) to force reasoning
    - Includes previous wrong answers to prevent looping
    - Escalates temperature on retries for diversity
    """
    context_text = "\n\n".join(
        f"### {c.title}\n{c.text}" for c in example.context
    )

    # Build reflection block
    reflection_block = ""
    if reflection_memory:
        reflection_block = (
            "\n\n--- Previous Reflections ---\n"
            + "\n".join(f"- {r}" for r in reflection_memory)
            + "\n--- End Reflections ---\n"
        )

    # Build previous wrong answers block (anti-looping)
    wrong_answers_block = ""
    if previous_answers:
        wrong_answers_block = (
            "\n--- Previous Wrong Answers (DO NOT repeat these) ---\n"
            + "\n".join(f"- {a}" for a in previous_answers)
            + "\n--- End Wrong Answers ---\n"
        )

    # Choose prompt strategy based on attempt
    use_cot = (attempt_id >= 2 and agent_type == "reflexion")

    if use_cot:
        # CoT prompt: force reasoning before answering
        user_prompt = (
            f"Question: {example.question}\n\n"
            f"Context:\n{context_text}"
            f"{reflection_block}"
            f"{wrong_answers_block}\n\n"
            f"Attempt: {attempt_id}/{3 if agent_type == 'reflexion' else 1}\n"
            "IMPORTANT: Think step by step. First reason about each hop, "
            "then give your answer.\n"
            "Respond in this format:\n"
            "Reasoning: <your chain of reasoning>\n"
            "Answer: <the final answer>"
        )
        system_prompt = ACTOR_SYSTEM_WITH_COT
    else:
        # Standard prompt for first attempt
        user_prompt = (
            f"Question: {example.question}\n\n"
            f"Context:\n{context_text}"
            f"{reflection_block}\n\n"
            f"Attempt: {attempt_id}/{3 if agent_type == 'reflexion' else 1}\n"
            "Provide ONLY the final answer — no explanation."
        )
        system_prompt = ACTOR_SYSTEM

    # Escalate temperature for retries to encourage diversity
    temp = 0.3 if attempt_id == 1 else min(0.3 + (attempt_id - 1) * 0.2, 0.7)

    result = _chat(system_prompt, user_prompt, temperature=temp)
    answer = _clean_answer(result["content"], use_cot=use_cot)
    return answer, result["tokens"], result["latency_ms"]


def _clean_answer(raw: str, use_cot: bool = False) -> str:
    """Strip markdown, quotes, prefixes like 'Answer:' etc."""
    text = raw.strip()

    # If CoT mode, extract only the Answer part
    if use_cot:
        # Look for "Answer:" line
        answer_match = re.search(r"(?:^|\n)\s*Answer:\s*(.+)", text, re.IGNORECASE)
        if answer_match:
            text = answer_match.group(1).strip()
        else:
            # Fallback: take the last non-empty line
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                text = lines[-1]

    text = text.strip('"').strip("'")
    # Remove common prefixes
    for prefix in ("Answer:", "Final Answer:", "The answer is", "A:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip().strip('"').strip("'")
    # Remove trailing period
    text = text.rstrip(".")
    return text.strip()


# ---------------------------------------------------------------------------
# Evaluator – uses the Structured Evaluator (bonus extension)
# ---------------------------------------------------------------------------

# Storage for structured evaluation results (for reporting)
_structured_eval_results: list[StructuredEvalResult] = []


def get_structured_eval_results() -> list[StructuredEvalResult]:
    """Return all structured evaluation results for report generation."""
    return _structured_eval_results


def clear_structured_eval_results() -> None:
    """Clear the structured evaluation results."""
    _structured_eval_results.clear()


def evaluator(
    example: QAExample,
    answer: str,
) -> tuple[JudgeResult, int, int]:
    """Evaluate using multi-dimensional structured evaluator.

    Delegates to the structured_evaluator module which applies a cascading
    pipeline: exact match → fuzzy containment → LLM multi-dimensional judge.

    Returns (JudgeResult, token_count, latency_ms) for backward compatibility.
    The richer StructuredEvalResult is stored internally and can be retrieved
    via get_structured_eval_results() for inclusion in reports.
    """
    structured_result, tokens, latency = structured_evaluate(
        example, answer, chat_fn=_chat
    )

    # Store for later aggregation in the report
    _structured_eval_results.append(structured_result)

    # Convert to JudgeResult for backward compatibility with the agent loop
    return structured_result.to_judge_result(), tokens, latency


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
    predicted_answer: str = "",
    previous_answers: list[str] | None = None,
) -> tuple[ReflectionEntry, int, int]:
    """Return (ReflectionEntry, token_count, latency_ms).

    Improvements over v1:
    - Includes the wrong answer in the prompt so reflector knows what to avoid
    - Includes all previous wrong answers to prevent suggestion loops
    - Asks reflector to propose a specific candidate answer
    """
    context_text = "\n\n".join(
        f"### {c.title}\n{c.text}" for c in example.context
    )

    # Build previous answers section
    prev_answers_text = ""
    if previous_answers:
        prev_answers_text = (
            f"\nAll previous wrong answers: {', '.join(previous_answers)}\n"
            "You MUST suggest a DIFFERENT answer than any of these.\n"
        )

    user_prompt = (
        f"Question: {example.question}\n"
        f"Context:\n{context_text}\n\n"
        f"Attempt #{attempt_id} was scored {judge.score}/1.\n"
        f"Your answer was: \"{predicted_answer}\"\n"
        f"Evaluator reason: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n"
        f"Spurious claims: {judge.spurious_claims}\n"
        f"{prev_answers_text}\n"
        "CRITICAL: Analyse why the answer above is wrong. "
        "Look at what the question is REALLY asking — it may ask about a "
        "different aspect than you focused on. Check if the question text "
        "itself contains the answer clue.\n\n"
        "Respond ONLY with a JSON object — no markdown fences:\n"
        '{"failure_reason": "...", "wrong_answer": "...", "lesson": "...", '
        '"next_strategy": "...", "candidate_answer": "your best guess"}'
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
            # Build next_strategy that includes the candidate answer
            next_strategy = str(data.get("next_strategy", "Re-read context carefully."))
            candidate = data.get("candidate_answer", "")
            if candidate and candidate.lower() not in next_strategy.lower():
                next_strategy += f" Consider answering: \"{candidate}\"."

            return ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=str(data.get("failure_reason", judge.reason)),
                lesson=str(data.get("lesson", "Review multi-hop reasoning steps.")),
                next_strategy=next_strategy,
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


# ---------------------------------------------------------------------------
# LATS branching actor — generates multiple diverse candidate answers
# ---------------------------------------------------------------------------
def lats_branch_actor(
    example: QAExample,
    attempt_id: int,
    reflection_memory: list[str],
    previous_answers: list[str] | None = None,
    num_branches: int = 3,
) -> tuple[list[dict[str, str]], int, int]:
    """Generate multiple candidate answers for tree-search.

    Returns (candidates_list, total_tokens, total_latency_ms).
    Each candidate is {"answer": "...", "reasoning": "..."}.
    """
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

    wrong_answers_block = ""
    if previous_answers:
        wrong_answers_block = (
            "\n--- Previous Wrong Answers (DO NOT repeat these) ---\n"
            + "\n".join(f"- {a}" for a in previous_answers)
            + "\n--- End Wrong Answers ---\n"
        )

    system_prompt = LATS_ACTOR_SYSTEM.format(num_branches=num_branches)
    user_prompt = (
        f"Question: {example.question}\n\n"
        f"Context:\n{context_text}"
        f"{reflection_block}"
        f"{wrong_answers_block}\n\n"
        f"Attempt: {attempt_id}\n"
        f"Generate exactly {num_branches} diverse candidate answers.\n"
        "Respond ONLY with a JSON array."
    )

    # Use higher temperature for diversity in branching
    temp = 0.5 + (attempt_id - 1) * 0.15
    result = _chat(system_prompt, user_prompt, temperature=min(temp, 0.8))

    candidates = _parse_lats_candidates(result["content"], num_branches)
    return candidates, result["tokens"], result["latency_ms"]


def _parse_lats_candidates(raw: str, num_branches: int) -> list[dict[str, str]]:
    """Parse the LATS actor's JSON array of candidates."""
    # Try to find JSON array in the response
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                candidates = []
                for item in data[:num_branches]:
                    if isinstance(item, dict):
                        answer = _clean_answer(str(item.get("answer", "")))
                        reasoning = str(item.get("reasoning", ""))
                        if answer:
                            candidates.append({
                                "answer": answer,
                                "reasoning": reasoning,
                            })
                if candidates:
                    return candidates
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: try to extract individual answers from text
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    candidates = []
    for line in lines:
        cleaned = _clean_answer(line)
        if cleaned and len(cleaned) < 100:
            candidates.append({"answer": cleaned, "reasoning": "extracted from text"})
    return candidates[:num_branches] if candidates else [{"answer": raw.strip()[:50], "reasoning": "fallback"}]
