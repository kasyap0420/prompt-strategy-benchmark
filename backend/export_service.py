import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any


def export_json(data: dict[str, Any] | list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def export_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def benchmark_run_to_dict(run: Any) -> dict[str, Any]:
    if hasattr(run, "model_dump"):
        return run.model_dump()
    return dict(run)


def benchmark_run_to_json(run: Any) -> str:
    return json.dumps(benchmark_run_to_dict(run), indent=2)


def benchmark_run_to_csv(run: Any) -> str:
    run_data = benchmark_run_to_dict(run)
    winners = run_data["winners"]
    rows: list[dict[str, Any]] = []
    for result in run_data["results"]:
        metrics = result["metrics"]
        rows.append(
            {
                "run_id": run_data["run_id"],
                "created_at": run_data["created_at"],
                "user_input": run_data["user_input"],
                "strategy_name": result["strategy_name"],
                "prompt": result["prompt"],
                "response": result["response"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "status": metrics["status"],
                "latency_ms": metrics["latency_ms"],
                "response_length": metrics["response_length"],
                "word_count": metrics["word_count"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "total_tokens": metrics["total_tokens"],
            }
        )

    output = StringIO()
    metadata_writer = csv.writer(output)
    metadata_writer.writerow(["metadata_key", "metadata_value"])
    metadata_writer.writerow(["overall_winner", winners["overall_winner"]])
    metadata_writer.writerow(["fastest_strategy", winners["fastest_strategy"]])
    metadata_writer.writerow(["most_detailed_strategy", winners["most_detailed_strategy"]])
    metadata_writer.writerow(
        ["most_token_efficient_strategy", winners["most_token_efficient_strategy"]]
    )
    metadata_writer.writerow(["benchmark_summary", run_data["benchmark_summary"]])
    metadata_writer.writerow([])

    fieldnames = [
        "run_id",
        "created_at",
        "user_input",
        "strategy_name",
        "prompt",
        "response",
        "error_type",
        "error_message",
        "status",
        "latency_ms",
        "response_length",
        "word_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
