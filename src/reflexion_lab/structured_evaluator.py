"""
Structured Evaluator — Bonus Extension for Reflexion Agent Lab.

Provides multi-dimensional, rubric-based evaluation of predicted answers
against gold answers. Unlike the basic binary evaluator, this module scores
answers across multiple quality dimensions and produces rich, structured
feedback that the Reflector can use for more targeted self-correction.

Evaluation Dimensions:
    1. Factual Accuracy  — Is the core fact correct?
    2. Completeness      — Does the answer capture all required information?
    3. Precision          — Is the answer free of extraneous/wrong info?
    4. Reasoning Quality  — Did the agent follow the right chain of hops?

Evaluation Strategies (applied in order):
    1. Exact match (normalized string comparison)
    2. Fuzzy containment (substring / token overlap)
    3. LLM-based structured judge (multi-dimensional rubric)

Author: Dương Phương Thảo
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .schemas import JudgeResult, QAExample
from .utils import normalize_answer


# ---------------------------------------------------------------------------
# Structured evaluation result (richer than basic JudgeResult)
# ---------------------------------------------------------------------------
@dataclass
class StructuredEvalResult:
    """Multi-dimensional evaluation result."""

    # Overall binary correctness (backward-compatible with JudgeResult)
    score: int = 0  # 1 = correct, 0 = incorrect

    # Dimensional scores (0.0 – 1.0 each)
    factual_accuracy: float = 0.0
    completeness: float = 0.0
    precision: float = 0.0
    reasoning_quality: float = 0.0

    # Weighted composite (0.0 – 1.0)
    composite_score: float = 0.0

    # Confidence in the evaluation itself (0.0 – 1.0)
    confidence: float = 0.0

    # Evaluation strategy that produced this result
    strategy: str = "unknown"

    # Textual feedback
    reason: str = ""
    missing_evidence: list[str] = field(default_factory=list)
    spurious_claims: list[str] = field(default_factory=list)
    improvement_hints: list[str] = field(default_factory=list)

    # Token / latency accounting
    eval_tokens: int = 0
    eval_latency_ms: int = 0

    def to_judge_result(self) -> JudgeResult:
        """Convert to basic JudgeResult for backward compatibility."""
        return JudgeResult(
            score=self.score,
            reason=self.reason,
            missing_evidence=self.missing_evidence,
            spurious_claims=self.spurious_claims,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for report inclusion."""
        return {
            "score": self.score,
            "factual_accuracy": self.factual_accuracy,
            "completeness": self.completeness,
            "precision": self.precision,
            "reasoning_quality": self.reasoning_quality,
            "composite_score": round(self.composite_score, 4),
            "confidence": round(self.confidence, 4),
            "strategy": self.strategy,
            "reason": self.reason,
            "missing_evidence": self.missing_evidence,
            "spurious_claims": self.spurious_claims,
            "improvement_hints": self.improvement_hints,
        }


# ---------------------------------------------------------------------------
# Dimension weights for composite score
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS = {
    "factual_accuracy": 0.50,
    "completeness": 0.20,
    "precision": 0.20,
    "reasoning_quality": 0.10,
}

# Threshold: composite >= this value → score = 1
CORRECTNESS_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# LLM-based structured evaluation prompt
# ---------------------------------------------------------------------------
STRUCTURED_EVAL_SYSTEM = """\
You are a rigorous, multi-dimensional answer evaluator. Given a question, the
gold (correct) answer, and a predicted answer, you must evaluate the prediction
across four quality dimensions.

## Scoring Rubric

### 1. Factual Accuracy (0.0 – 1.0)
- 1.0: Predicted answer is semantically identical to gold answer.
- 0.7: Core entity is correct but minor details differ (e.g., spelling variant).
- 0.3: Partially correct — related but not the exact answer.
- 0.0: Completely wrong entity/fact.

### 2. Completeness (0.0 – 1.0)
- 1.0: All parts of the gold answer are present.
- 0.5: Some parts are present, others missing.
- 0.0: None of the required information is present.

### 3. Precision (0.0 – 1.0)
- 1.0: Answer contains only relevant information, no hallucinations.
- 0.5: Answer is mostly relevant but includes some extraneous info.
- 0.0: Answer is mostly irrelevant or contains significant hallucinations.

### 4. Reasoning Quality (0.0 – 1.0)
- 1.0: The answer demonstrates correct multi-hop reasoning.
- 0.5: Reasoning was partially correct (some hops right, some wrong).
- 0.0: No evidence of correct reasoning chain.

## Rules
- Minor spelling, casing, or article differences are acceptable for full marks.
- "Sacramento" vs "Sacramento, California" — both acceptable if context is clear.
- Abbreviations are acceptable (e.g., "NYC" = "New York City").

Respond ONLY with a JSON object (no markdown fences, no extra text):
{
  "factual_accuracy": 0.0,
  "completeness": 0.0,
  "precision": 0.0,
  "reasoning_quality": 0.0,
  "reason": "brief explanation of the evaluation",
  "missing_evidence": ["list of missing pieces"],
  "spurious_claims": ["list of incorrect claims"],
  "improvement_hints": ["actionable suggestions for the agent"]
}
"""


# ---------------------------------------------------------------------------
# Token-overlap similarity (no LLM needed)
# ---------------------------------------------------------------------------
def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on normalized token sets."""
    tokens_a = set(normalize_answer(a).split())
    tokens_b = set(normalize_answer(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _containment_score(gold: str, predicted: str) -> float:
    """How much of the gold answer is contained in the prediction."""
    gold_tokens = set(normalize_answer(gold).split())
    pred_tokens = set(normalize_answer(predicted).split())
    if not gold_tokens:
        return 0.0
    return len(gold_tokens & pred_tokens) / len(gold_tokens)


# ---------------------------------------------------------------------------
# Strategy 1: Exact match (fast, no LLM call)
# ---------------------------------------------------------------------------
def _eval_exact_match(
    example: QAExample, answer: str
) -> StructuredEvalResult | None:
    """Return a perfect-score result if exact match, else None."""
    if normalize_answer(answer) == normalize_answer(example.gold_answer):
        return StructuredEvalResult(
            score=1,
            factual_accuracy=1.0,
            completeness=1.0,
            precision=1.0,
            reasoning_quality=1.0,
            composite_score=1.0,
            confidence=1.0,
            strategy="exact_match",
            reason="Exact match after normalization.",
        )
    return None


# ---------------------------------------------------------------------------
# Strategy 2: Fuzzy / containment heuristic (fast, no LLM call)
# ---------------------------------------------------------------------------
def _eval_fuzzy(
    example: QAExample, answer: str
) -> StructuredEvalResult | None:
    """
    Return a result if the answer is clearly correct via containment
    heuristics (e.g., "Imi Lichtenfeld" matches 'Emrich "Imi" Lichtenfeld').
    Returns None if uncertain — falls through to LLM judge.
    """
    gold_norm = normalize_answer(example.gold_answer)
    pred_norm = normalize_answer(answer)

    overlap = _token_overlap(example.gold_answer, answer)
    containment = _containment_score(example.gold_answer, answer)

    # Check normalized string containment (handles partial name matches)
    # e.g. "imi lichtenfeld" is a substring of "emrich imi lichtenfeld"
    string_contained = gold_norm in pred_norm or pred_norm in gold_norm

    if string_contained:
        # Determine match quality based on relative lengths
        len_ratio = min(len(pred_norm), len(gold_norm)) / max(
            len(pred_norm), len(gold_norm), 1
        )

        if len_ratio >= 0.5 and overlap >= 0.4:
            # Good substring match with reasonable overlap
            adjusted_containment = max(containment, len_ratio)
            return StructuredEvalResult(
                score=1,
                factual_accuracy=0.9,
                completeness=adjusted_containment,
                precision=min(1.0, overlap + 0.2),
                reasoning_quality=0.8,
                composite_score=0.9,
                confidence=0.85,
                strategy="fuzzy_containment",
                reason=(
                    f"Fuzzy match: predicted '{answer}' contains/is contained "
                    f"by gold '{example.gold_answer}' "
                    f"(overlap={overlap:.2f}, containment={containment:.2f}, "
                    f"len_ratio={len_ratio:.2f})."
                ),
            )

    # Token-level containment (e.g. gold tokens are mostly in prediction)
    if containment >= 0.7 and overlap >= 0.5:
        return StructuredEvalResult(
            score=1,
            factual_accuracy=0.85,
            completeness=containment,
            precision=min(1.0, overlap + 0.1),
            reasoning_quality=0.75,
            composite_score=0.85,
            confidence=0.8,
            strategy="fuzzy_containment",
            reason=(
                f"Token containment match: {containment:.2f} of gold tokens "
                f"found in prediction (overlap={overlap:.2f})."
            ),
        )

    # Token overlap alone (e.g., "Sarah J. Maas" vs "Sarah Janet Maas")
    if overlap >= 0.7:
        return None  # Uncertain — let LLM decide

    return None


# ---------------------------------------------------------------------------
# Strategy 3: LLM-based structured judge (most expensive, most accurate)
# ---------------------------------------------------------------------------
def _eval_llm_judge(
    example: QAExample,
    answer: str,
    chat_fn: Any,
) -> StructuredEvalResult:
    """Use the LLM to produce a multi-dimensional evaluation."""
    user_prompt = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n\n"
        "Evaluate the predicted answer against the gold answer using the "
        "rubric. Respond ONLY with a JSON object — no markdown fences:\n"
        '{"factual_accuracy": 0.0, "completeness": 0.0, "precision": 0.0, '
        '"reasoning_quality": 0.0, "reason": "...", '
        '"missing_evidence": [...], "spurious_claims": [...], '
        '"improvement_hints": [...]}'
    )

    result = chat_fn(STRUCTURED_EVAL_SYSTEM, user_prompt, temperature=0.0)
    parsed = _parse_structured_response(result["content"], answer)
    parsed.eval_tokens = result.get("tokens", 0)
    parsed.eval_latency_ms = result.get("latency_ms", 0)
    return parsed


def _parse_structured_response(raw: str, answer: str) -> StructuredEvalResult:
    """Parse the LLM's JSON response into a StructuredEvalResult."""
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())

            factual = _clamp(float(data.get("factual_accuracy", 0.0)))
            completeness = _clamp(float(data.get("completeness", 0.0)))
            precision = _clamp(float(data.get("precision", 0.0)))
            reasoning = _clamp(float(data.get("reasoning_quality", 0.0)))

            composite = (
                DIMENSION_WEIGHTS["factual_accuracy"] * factual
                + DIMENSION_WEIGHTS["completeness"] * completeness
                + DIMENSION_WEIGHTS["precision"] * precision
                + DIMENSION_WEIGHTS["reasoning_quality"] * reasoning
            )

            binary_score = 1 if composite >= CORRECTNESS_THRESHOLD else 0

            return StructuredEvalResult(
                score=binary_score,
                factual_accuracy=factual,
                completeness=completeness,
                precision=precision,
                reasoning_quality=reasoning,
                composite_score=composite,
                confidence=0.9,
                strategy="llm_structured_judge",
                reason=str(data.get("reason", "No reason provided.")),
                missing_evidence=data.get("missing_evidence", []),
                spurious_claims=data.get("spurious_claims", []),
                improvement_hints=data.get("improvement_hints", []),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: treat as incorrect
    return StructuredEvalResult(
        score=0,
        confidence=0.3,
        strategy="llm_structured_judge_fallback",
        reason=f"Could not parse structured evaluation. Raw: {raw[:200]}",
        spurious_claims=[answer],
    )


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Main entry point — cascading evaluation pipeline
# ---------------------------------------------------------------------------
def structured_evaluate(
    example: QAExample,
    answer: str,
    chat_fn: Any = None,
) -> tuple[StructuredEvalResult, int, int]:
    """
    Evaluate a predicted answer using cascading strategies.

    Strategies are tried in order (cheapest first):
        1. Exact match (0 tokens, 0 latency)
        2. Fuzzy containment heuristic (0 tokens, 0 latency)
        3. LLM structured judge (uses tokens, has latency)

    Args:
        example:  The QA example with gold answer and context.
        answer:   The predicted answer to evaluate.
        chat_fn:  The LLM chat function (from ollama_runtime._chat).
                  If None, only strategies 1-2 are used.

    Returns:
        (StructuredEvalResult, total_tokens, latency_ms)
    """
    # Strategy 1: Exact match
    result = _eval_exact_match(example, answer)
    if result is not None:
        return result, 0, 0

    # Strategy 2: Fuzzy containment
    result = _eval_fuzzy(example, answer)
    if result is not None:
        return result, 0, 0

    # Strategy 3: LLM structured judge
    if chat_fn is not None:
        result = _eval_llm_judge(example, answer, chat_fn)
        return result, result.eval_tokens, result.eval_latency_ms

    # No LLM available — fall back to simple heuristic scoring
    overlap = _token_overlap(example.gold_answer, answer)
    containment = _containment_score(example.gold_answer, answer)
    composite = 0.5 * containment + 0.3 * overlap
    return (
        StructuredEvalResult(
            score=1 if composite >= CORRECTNESS_THRESHOLD else 0,
            factual_accuracy=containment,
            completeness=containment,
            precision=overlap,
            reasoning_quality=0.5,
            composite_score=composite,
            confidence=0.5,
            strategy="heuristic_fallback",
            reason=f"Heuristic: overlap={overlap:.2f}, containment={containment:.2f}",
        ),
        0,
        0,
    )


# ---------------------------------------------------------------------------
# Batch evaluation + aggregate metrics
# ---------------------------------------------------------------------------
@dataclass
class EvaluationSummary:
    """Aggregated metrics across multiple evaluations."""

    total: int = 0
    correct: int = 0
    accuracy: float = 0.0

    avg_factual_accuracy: float = 0.0
    avg_completeness: float = 0.0
    avg_precision: float = 0.0
    avg_reasoning_quality: float = 0.0
    avg_composite: float = 0.0
    avg_confidence: float = 0.0

    strategy_counts: dict[str, int] = field(default_factory=dict)
    total_eval_tokens: int = 0
    total_eval_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "avg_dimensions": {
                "factual_accuracy": round(self.avg_factual_accuracy, 4),
                "completeness": round(self.avg_completeness, 4),
                "precision": round(self.avg_precision, 4),
                "reasoning_quality": round(self.avg_reasoning_quality, 4),
            },
            "avg_composite": round(self.avg_composite, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "strategy_counts": self.strategy_counts,
            "total_eval_tokens": self.total_eval_tokens,
            "total_eval_latency_ms": self.total_eval_latency_ms,
        }


def summarize_evaluations(
    results: list[StructuredEvalResult],
) -> EvaluationSummary:
    """Compute aggregate metrics from a list of structured evaluations."""
    if not results:
        return EvaluationSummary()

    n = len(results)
    correct = sum(1 for r in results if r.score == 1)
    strategy_counts: dict[str, int] = {}
    for r in results:
        strategy_counts[r.strategy] = strategy_counts.get(r.strategy, 0) + 1

    return EvaluationSummary(
        total=n,
        correct=correct,
        accuracy=correct / n,
        avg_factual_accuracy=sum(r.factual_accuracy for r in results) / n,
        avg_completeness=sum(r.completeness for r in results) / n,
        avg_precision=sum(r.precision for r in results) / n,
        avg_reasoning_quality=sum(r.reasoning_quality for r in results) / n,
        avg_composite=sum(r.composite_score for r in results) / n,
        avg_confidence=sum(r.confidence for r in results) / n,
        strategy_counts=strategy_counts,
        total_eval_tokens=sum(r.eval_tokens for r in results),
        total_eval_latency_ms=sum(r.eval_latency_ms for r in results),
    )
