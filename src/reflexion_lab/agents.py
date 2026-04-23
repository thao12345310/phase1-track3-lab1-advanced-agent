from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from rich import print as rprint

from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

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
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
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
                    example, attempt_id, self.agent_type, reflection_memory
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

            # ---------- Reflector (only for reflexion agent with remaining attempts) ----------
            if (
                self.agent_type == "reflexion"
                and attempt_id < self.max_attempts
            ):
                if _RUNTIME == "ollama":
                    ref_entry, ref_tokens, ref_lat = reflector(
                        example, attempt_id, judge
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
