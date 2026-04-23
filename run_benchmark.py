from __future__ import annotations
import json
import time
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent, LATSAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)

# Will be lazily imported when mode == 'ollama'
_clear_structured_eval: callable = None
_get_structured_eval: callable = None


@app.command()
def main(
    dataset: str = "data/hotpot_full.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    lats_depth: int = 2,
    lats_branches: int = 3,
    mode: str = "ollama",
) -> None:
    examples = load_dataset(dataset)
    print(f"[bold blue]Loaded {len(examples)} examples from {dataset}[/bold blue]")
    print(f"[bold blue]Mode: {mode} | Reflexion attempts: {reflexion_attempts}[/bold blue]")
    print(f"[bold blue]LATS depth: {lats_depth} | LATS branches: {lats_branches}[/bold blue]\n")

    # Import trace functions if using ollama
    trace_log_getter = None
    if mode == "ollama":
        try:
            from src.reflexion_lab.ollama_runtime import (
                get_trace_log, clear_trace_log,
                get_structured_eval_results, clear_structured_eval_results,
            )
            clear_trace_log()
            clear_structured_eval_results()
            trace_log_getter = get_trace_log
            global _clear_structured_eval, _get_structured_eval
            _clear_structured_eval = clear_structured_eval_results
            _get_structured_eval = get_structured_eval_results
            print("[dim]Tracing enabled — all LLM calls will be recorded[/dim]")
            print("[dim]Structured evaluator enabled — multi-dimensional scoring active[/dim]\n")
        except ImportError:
            pass

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    lats = LATSAgent(max_depth=lats_depth, num_branches=lats_branches)

    t_start = time.time()

    # --- ReAct pass ---
    print("[bold cyan]═══ ReAct Agent ═══[/bold cyan]")
    react_records = []
    for i, example in enumerate(examples, 1):
        print(f"[bold]({i}/{len(examples)}) {example.qid}:[/bold] {example.question[:60]}...")
        record = react.run(example)
        react_records.append(record)

    react_correct = sum(1 for r in react_records if r.is_correct)
    print(f"\n[bold green]ReAct: {react_correct}/{len(react_records)} correct ({react_correct/len(react_records)*100:.1f}%)[/bold green]\n")

    # --- Reflexion pass ---
    print(f"[bold cyan]═══ Reflexion Agent (max {reflexion_attempts} attempts) ═══[/bold cyan]")
    reflexion_records = []
    for i, example in enumerate(examples, 1):
        print(f"[bold]({i}/{len(examples)}) {example.qid}:[/bold] {example.question[:60]}...")
        record = reflexion.run(example)
        reflexion_records.append(record)

    reflexion_correct = sum(1 for r in reflexion_records if r.is_correct)
    print(f"\n[bold green]Reflexion: {reflexion_correct}/{len(reflexion_records)} correct ({reflexion_correct/len(reflexion_records)*100:.1f}%)[/bold green]\n")

    # --- LATS pass ---
    print(f"[bold cyan]═══ LATS Agent (depth={lats_depth}, branches={lats_branches}) ═══[/bold cyan]")
    lats_records = []
    for i, example in enumerate(examples, 1):
        print(f"[bold]({i}/{len(examples)}) {example.qid}:[/bold] {example.question[:60]}...")
        record = lats.run(example)
        lats_records.append(record)

    lats_correct = sum(1 for r in lats_records if r.is_correct)
    print(f"\n[bold green]LATS: {lats_correct}/{len(lats_records)} correct ({lats_correct/len(lats_records)*100:.1f}%)[/bold green]")

    t_total = time.time() - t_start
    print(f"[dim]Total wall time: {t_total:.1f}s[/dim]\n")

    # --- Save outputs ---
    all_records = react_records + reflexion_records + lats_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    save_jsonl(out_path / "lats_runs.jsonl", lats_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")

    # --- Save trace log ---
    if trace_log_getter:
        traces = trace_log_getter()
        trace_path = out_path / "trace_log.json"
        trace_path.write_text(
            json.dumps(traces, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        total_tokens = sum(t.get("total_tokens", 0) for t in traces)
        total_llm_calls = len(traces)
        avg_latency = (
            sum(t.get("latency_ms", 0) for t in traces) / total_llm_calls
            if total_llm_calls else 0
        )
        print(f"[green]Saved[/green] {trace_path}")
        print(f"\n[bold magenta]═══ Trace Summary ═══[/bold magenta]")
        print(f"  Total LLM calls: {total_llm_calls}")
        print(f"  Total tokens:    {total_tokens:,}")
        print(f"  Avg latency:     {avg_latency:.0f}ms/call")
        print(f"  Trace file:      {trace_path}")

    # --- Save structured evaluation summary ---
    if _get_structured_eval is not None:
        from src.reflexion_lab.structured_evaluator import summarize_evaluations
        eval_results = _get_structured_eval()
        if eval_results:
            eval_summary = summarize_evaluations(eval_results)
            eval_path = out_path / "structured_eval_summary.json"
            eval_detail_path = out_path / "structured_eval_details.json"
            eval_path.write_text(
                json.dumps(eval_summary.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            eval_details = [r.to_dict() for r in eval_results]
            eval_detail_path.write_text(
                json.dumps(eval_details, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[green]Saved[/green] {eval_path}")
            print(f"[green]Saved[/green] {eval_detail_path}")
            print(f"\n[bold magenta]═══ Structured Evaluation Summary ═══[/bold magenta]")
            print(f"  Total evaluations:    {eval_summary.total}")
            print(f"  Correct:              {eval_summary.correct}/{eval_summary.total} ({eval_summary.accuracy:.1%})")
            print(f"  Avg factual accuracy: {eval_summary.avg_factual_accuracy:.3f}")
            print(f"  Avg completeness:     {eval_summary.avg_completeness:.3f}")
            print(f"  Avg precision:        {eval_summary.avg_precision:.3f}")
            print(f"  Avg reasoning:        {eval_summary.avg_reasoning_quality:.3f}")
            print(f"  Avg composite:        {eval_summary.avg_composite:.3f}")
            print(f"  Strategy breakdown:   {eval_summary.strategy_counts}")

    print(f"\n{json.dumps(report.summary, indent=2)}")


if __name__ == "__main__":
    app()
