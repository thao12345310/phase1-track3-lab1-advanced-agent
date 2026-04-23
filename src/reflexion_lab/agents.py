from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from rich import print as rprint

from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer

# Import runtime based on environment — default to real Ollama runtime
try:
    from .ollama_runtime import (
        actor_answer,
        evaluator,
        reflector,
        classify_failure,
    )
    _RUNTIME = "ollama"
except ImportError:
    from .mock_runtime import (  # type: ignore[assignment]
        actor_answer,
        evaluator,
        reflector,
        FAILURE_MODE_BY_QID,
    )
    _RUNTIME = "mock"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion", "lats"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        previous_answers: list[str] = []   # Track all wrong answers
        final_answer = ""
        final_score = 0
        total_tokens = 0
        total_latency = 0

        for attempt_id in range(1, self.max_attempts + 1):
            rprint(
                f"  [dim]attempt {attempt_id}/{self.max_attempts} "
                f"({self.agent_type})[/dim]"
            )

            # ---------- Actor ----------
            if _RUNTIME == "ollama":
                answer, actor_tokens, actor_lat = actor_answer(
                    example, attempt_id, self.agent_type, reflection_memory,
                    previous_answers=previous_answers if previous_answers else None,
                )
            else:
                answer = actor_answer(
                    example, attempt_id, self.agent_type, reflection_memory
                )
                actor_tokens = 320 + (attempt_id * 65) + (
                    120 if self.agent_type == "reflexion" else 0
                )
                actor_lat = 160 + (attempt_id * 40) + (
                    90 if self.agent_type == "reflexion" else 0
                )

            # ---------- Looping detection ----------
            # If agent gives same answer as before, log it but still evaluate
            answer_norm = normalize_answer(answer)
            is_repeat = any(
                normalize_answer(pa) == answer_norm for pa in previous_answers
            )
            if is_repeat and attempt_id < self.max_attempts:
                rprint(
                    f"    [yellow]⚠ repeated answer detected: {answer}[/yellow]"
                )

            # ---------- Evaluator ----------
            if _RUNTIME == "ollama":
                judge, eval_tokens, eval_lat = evaluator(example, answer)
            else:
                judge = evaluator(example, answer)
                eval_tokens = 0
                eval_lat = 0

            step_tokens = actor_tokens + eval_tokens
            step_latency = actor_lat + eval_lat

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=step_tokens,
                latency_ms=step_latency,
            )

            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                total_tokens += step_tokens
                total_latency += step_latency
                rprint(f"    [green]✓ correct[/green]")
                break

            # Track this wrong answer
            previous_answers.append(answer)

            # ---------- Reflector (only for reflexion agent with remaining attempts) ----------
            if (
                self.agent_type == "reflexion"
                and attempt_id < self.max_attempts
            ):
                if _RUNTIME == "ollama":
                    ref_entry, ref_tokens, ref_lat = reflector(
                        example, attempt_id, judge,
                        predicted_answer=answer,
                        previous_answers=previous_answers,
                    )
                    step_tokens += ref_tokens
                    step_latency += ref_lat
                    trace.token_estimate = step_tokens
                    trace.latency_ms = step_latency
                else:
                    ref_entry = reflector(example, attempt_id, judge)

                trace.reflection = ref_entry
                reflections.append(ref_entry)
                reflection_memory.append(ref_entry.next_strategy)
                rprint(
                    f"    [yellow]✗ wrong → reflecting: {ref_entry.next_strategy[:80]}...[/yellow]"
                )
            else:
                rprint(f"    [red]✗ wrong[/red]")

            traces.append(trace)
            total_tokens += step_tokens
            total_latency += step_latency

        # ---------- Failure-mode classification ----------
        if _RUNTIME == "ollama":
            failure_mode = classify_failure(example, final_answer, traces)
        else:
            failure_mode = (
                "none"
                if final_score == 1
                else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
            )

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)


class LATSAgent:
    """Language Agent Tree Search — generates multiple candidate answers per
    attempt, evaluates each, and picks the best. If none are correct, reflects
    and retries with a new branch set."""

    agent_type: str = "lats"

    def __init__(self, max_depth: int = 2, num_branches: int = 3) -> None:
        self.max_depth = max_depth
        self.num_branches = num_branches

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        previous_answers: list[str] = []
        final_answer = ""
        final_score = 0
        total_tokens = 0
        total_latency = 0
        found_correct = False

        for depth in range(1, self.max_depth + 1):
            rprint(
                f"  [dim]depth {depth}/{self.max_depth} "
                f"(lats, {self.num_branches} branches)[/dim]"
            )

            # ---------- Branch: generate multiple candidates ----------
            if _RUNTIME == "ollama":
                from .ollama_runtime import lats_branch_actor
                candidates, branch_tokens, branch_lat = lats_branch_actor(
                    example, depth, reflection_memory,
                    previous_answers=previous_answers if previous_answers else None,
                    num_branches=self.num_branches,
                )
            else:
                # Mock mode: generate simple candidates
                if depth == 2 or example.qid not in FAILURE_MODE_BY_QID:
                    candidates = [
                        {"answer": example.gold_answer, "reasoning": "direct match"}
                    ]
                else:
                    candidates = [
                        {"answer": f"wrong_{i}", "reasoning": f"candidate {i}"}
                        for i in range(self.num_branches)
                    ]
                branch_tokens = 500
                branch_lat = 300

            total_tokens += branch_tokens
            total_latency += branch_lat

            # ---------- Evaluate each candidate ----------
            best_answer = ""
            best_score = -1
            best_judge = None

            for ci, cand in enumerate(candidates):
                answer = cand["answer"]
                rprint(f"    [dim]branch {ci+1}: {answer[:50]}[/dim]")

                if _RUNTIME == "ollama":
                    judge, eval_tokens, eval_lat = evaluator(example, answer)
                else:
                    judge = evaluator(example, answer)
                    eval_tokens = 0
                    eval_lat = 0

                total_tokens += eval_tokens
                total_latency += eval_lat

                if judge.score > best_score:
                    best_score = judge.score
                    best_answer = answer
                    best_judge = judge

                if judge.score == 1:
                    rprint(f"    [green]✓ branch {ci+1} correct: {answer}[/green]")
                    found_correct = True
                    break

            final_answer = best_answer
            final_score = best_score if best_score >= 0 else 0

            trace = AttemptTrace(
                attempt_id=depth,
                answer=best_answer,
                score=final_score,
                reason=best_judge.reason if best_judge else "no evaluation",
                token_estimate=branch_tokens,
                latency_ms=branch_lat,
            )

            if found_correct:
                traces.append(trace)
                break

            # Track wrong answers from this depth
            for cand in candidates:
                ans = cand["answer"]
                if normalize_answer(ans) not in [normalize_answer(pa) for pa in previous_answers]:
                    previous_answers.append(ans)

            # ---------- Reflect before next depth ----------
            if depth < self.max_depth and best_judge:
                if _RUNTIME == "ollama":
                    ref_entry, ref_tokens, ref_lat = reflector(
                        example, depth, best_judge,
                        predicted_answer=best_answer,
                        previous_answers=previous_answers,
                    )
                    total_tokens += ref_tokens
                    total_latency += ref_lat
                else:
                    ref_entry = reflector(example, depth, best_judge)

                trace.reflection = ref_entry
                reflections.append(ref_entry)
                reflection_memory.append(ref_entry.next_strategy)
                rprint(
                    f"    [yellow]✗ best branch wrong → reflecting: "
                    f"{ref_entry.next_strategy[:80]}...[/yellow]"
                )
            else:
                rprint(f"    [red]✗ all branches wrong[/red]")

            traces.append(trace)

        # ---------- Failure-mode classification ----------
        if _RUNTIME == "ollama":
            failure_mode = classify_failure(example, final_answer, traces)
        else:
            failure_mode = (
                "none"
                if final_score == 1
                else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
            )

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type="lats",
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )
