from __future__ import annotations
import json
import time
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "ollama",
) -> None:
    examples = load_dataset(dataset)
    print(f"[bold blue]Loaded {len(examples)} examples from {dataset}[/bold blue]")
    print(f"[bold blue]Mode: {mode} | Reflexion attempts: {reflexion_attempts}[/bold blue]\n")

    # Import trace functions if using ollama
    trace_log_getter = None
    if mode == "ollama":
        try:
            from src.reflexion_lab.ollama_runtime import get_trace_log, clear_trace_log
            clear_trace_log()
            trace_log_getter = get_trace_log
            print("[dim]Tracing enabled — all LLM calls will be recorded[/dim]\n")
        except ImportError:
            pass

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

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
    print(f"\n[bold green]Reflexion: {reflexion_correct}/{len(reflexion_records)} correct ({reflexion_correct/len(reflexion_records)*100:.1f}%)[/bold green]")

    t_total = time.time() - t_start
    print(f"[dim]Total wall time: {t_total:.1f}s[/dim]\n")

    # --- Save outputs ---
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
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

    print(f"\n{json.dumps(report.summary, indent=2)}")


if __name__ == "__main__":
    app()
