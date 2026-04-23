"""
Tests for the structured_evaluator bonus extension.

Validates:
    1. Exact match detection (fast path, no LLM)
    2. Fuzzy containment matching
    3. Heuristic fallback (no LLM available)
    4. Composite score calculation
    5. Backward compatibility with JudgeResult
    6. EvaluationSummary aggregation
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reflexion_lab.schemas import QAExample, ContextChunk
from src.reflexion_lab.structured_evaluator import (
    StructuredEvalResult,
    structured_evaluate,
    summarize_evaluations,
    _token_overlap,
    _containment_score,
    CORRECTNESS_THRESHOLD,
)


def _make_example(question: str, gold: str) -> QAExample:
    """Helper to create a minimal QAExample."""
    return QAExample(
        qid="test1",
        difficulty="easy",
        question=question,
        gold_answer=gold,
        context=[
            ContextChunk(title="Test", text="Some context about the question.")
        ],
    )


class TestTokenOverlap:
    def test_identical(self):
        assert _token_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _token_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        score = _token_overlap("hello world foo", "hello world bar")
        assert 0.3 < score < 0.8  # 2/4 overlap

    def test_empty(self):
        assert _token_overlap("", "hello") == 0.0


class TestContainmentScore:
    def test_full_containment(self):
        assert _containment_score("Australia", "Australia") == 1.0

    def test_partial_containment(self):
        score = _containment_score("New York City", "New York")
        assert 0.5 <= score <= 1.0

    def test_no_containment(self):
        score = _containment_score("Australia", "France")
        assert score == 0.0


class TestExactMatch:
    def test_exact_match(self):
        ex = _make_example("Where is it?", "Australia")
        result, tokens, latency = structured_evaluate(ex, "Australia")
        assert result.score == 1
        assert result.strategy == "exact_match"
        assert tokens == 0
        assert latency == 0

    def test_case_insensitive_match(self):
        ex = _make_example("Where is it?", "Australia")
        result, _, _ = structured_evaluate(ex, "australia")
        assert result.score == 1
        assert result.strategy == "exact_match"

    def test_no_match(self):
        ex = _make_example("Where is it?", "Australia")
        result, _, _ = structured_evaluate(ex, "France")
        assert result.score == 0
        assert result.strategy != "exact_match"


class TestFuzzyMatch:
    def test_substring_match(self):
        ex = _make_example("Who founded it?", 'Emrich "Imi" Lichtenfeld')
        result, tokens, _ = structured_evaluate(ex, "Imi Lichtenfeld")
        # Should be caught by fuzzy containment
        assert result.score == 1
        assert result.strategy in ("fuzzy_containment", "exact_match")
        assert tokens == 0

    def test_abbreviation_match(self):
        ex = _make_example("Who wrote it?", "Sarah Janet Maas")
        result, _, _ = structured_evaluate(ex, "Sarah J Maas")
        # Token overlap may not be high enough for fuzzy, but let's see
        # At minimum, it should go to heuristic fallback
        assert result.strategy in (
            "fuzzy_containment",
            "heuristic_fallback",
        )


class TestFallbackHeuristic:
    def test_no_llm_returns_heuristic(self):
        ex = _make_example("Where is it?", "California")
        result, tokens, latency = structured_evaluate(
            ex, "Florida", chat_fn=None
        )
        assert result.score == 0
        assert result.strategy == "heuristic_fallback"
        assert tokens == 0
        assert latency == 0


class TestStructuredEvalResult:
    def test_to_judge_result(self):
        r = StructuredEvalResult(
            score=1,
            reason="Test reason",
            missing_evidence=["a"],
            spurious_claims=["b"],
        )
        judge = r.to_judge_result()
        assert judge.score == 1
        assert judge.reason == "Test reason"
        assert judge.missing_evidence == ["a"]
        assert judge.spurious_claims == ["b"]

    def test_to_dict(self):
        r = StructuredEvalResult(
            score=1,
            factual_accuracy=0.9,
            composite_score=0.85,
            strategy="exact_match",
        )
        d = r.to_dict()
        assert d["score"] == 1
        assert d["factual_accuracy"] == 0.9
        assert d["composite_score"] == 0.85
        assert d["strategy"] == "exact_match"


class TestEvaluationSummary:
    def test_summary_correct(self):
        results = [
            StructuredEvalResult(
                score=1,
                factual_accuracy=1.0,
                completeness=1.0,
                precision=1.0,
                reasoning_quality=1.0,
                composite_score=1.0,
                strategy="exact_match",
            ),
            StructuredEvalResult(
                score=0,
                factual_accuracy=0.0,
                completeness=0.0,
                precision=0.0,
                reasoning_quality=0.0,
                composite_score=0.0,
                strategy="heuristic_fallback",
            ),
        ]
        summary = summarize_evaluations(results)
        assert summary.total == 2
        assert summary.correct == 1
        assert summary.accuracy == 0.5
        assert summary.avg_factual_accuracy == 0.5
        assert summary.strategy_counts["exact_match"] == 1
        assert summary.strategy_counts["heuristic_fallback"] == 1

    def test_empty_summary(self):
        summary = summarize_evaluations([])
        assert summary.total == 0
        assert summary.accuracy == 0.0

    def test_to_dict(self):
        results = [
            StructuredEvalResult(score=1, composite_score=0.9, strategy="exact_match"),
        ]
        summary = summarize_evaluations(results)
        d = summary.to_dict()
        assert "total" in d
        assert "avg_dimensions" in d
        assert "strategy_counts" in d


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
