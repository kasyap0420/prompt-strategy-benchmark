import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse
from uuid import uuid4

from backend.config import BASE_DIR, settings
from backend.schemas import (
    BenchmarkHistoryItem,
    BenchmarkMetrics,
    BenchmarkRankingItem,
    BenchmarkResponse,
    BenchmarkResult,
    BenchmarkRunResponse,
    BenchmarkWinners,
)


def _database_path() -> Path:
    parsed = urlparse(settings.database_url)
    if parsed.scheme != "sqlite":
        raise ValueError("Only SQLite DATABASE_URL values are supported in Phase 1.")

    raw_path = unquote(parsed.path or "")
    if parsed.netloc:
        raw_path = f"//{parsed.netloc}{raw_path}"

    if not raw_path:
        raise ValueError("SQLite DATABASE_URL must include a database file path.")

    if raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
        raw_path = raw_path[1:]
    elif raw_path.startswith("/") and not raw_path.startswith("//"):
        raw_path = raw_path[1:]

    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


DATABASE_PATH = _database_path()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Provide a SQLite connection with row access by column name."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Initialize the SQLite database schema used by benchmark history."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                user_input TEXT NOT NULL,
                overall_winner TEXT,
                fastest_strategy TEXT,
                most_detailed_strategy TEXT,
                most_token_efficient_strategy TEXT,
                benchmark_summary TEXT
            )
            """
        )
        _ensure_column(connection, "benchmark_runs", "overall_winner", "TEXT")
        _ensure_column(connection, "benchmark_runs", "fastest_strategy", "TEXT")
        _ensure_column(connection, "benchmark_runs", "most_detailed_strategy", "TEXT")
        _ensure_column(
            connection,
            "benchmark_runs",
            "most_token_efficient_strategy",
            "TEXT",
        )
        _ensure_column(connection, "benchmark_runs", "benchmark_summary", "TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT,
                error_type TEXT,
                error_message TEXT,
                status TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                response_length INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs (run_id) ON DELETE CASCADE
            )
            """
        )
        _ensure_column(connection, "benchmark_results", "error_type", "TEXT")
        _ensure_column(connection, "benchmark_results", "error_message", "TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                strategy_name TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                response_length INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                total_tokens INTEGER,
                token_efficiency REAL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs (run_id) ON DELETE CASCADE
            )
            """
        )


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def create_benchmark_run(user_input: str, response: BenchmarkResponse) -> BenchmarkRunResponse:
    """Persist a benchmark response and return the stored run."""
    init_db()
    run_id = response.run_id or str(uuid4())
    created_at = response.created_at or datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO benchmark_runs (
                run_id,
                created_at,
                user_input,
                overall_winner,
                fastest_strategy,
                most_detailed_strategy,
                most_token_efficient_strategy,
                benchmark_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                user_input,
                response.winners.overall_winner,
                response.winners.fastest_strategy,
                response.winners.most_detailed_strategy,
                response.winners.most_token_efficient_strategy,
                response.benchmark_summary,
            ),
        )
        connection.executemany(
            """
            INSERT INTO benchmark_results (
                run_id,
                strategy_name,
                prompt,
                response,
                error_type,
                error_message,
                status,
                latency_ms,
                response_length,
                word_count,
                input_tokens,
                output_tokens,
                total_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    result.strategy_name,
                    result.prompt,
                    result.response,
                    result.error_type,
                    result.error_message,
                    result.metrics.status,
                    result.metrics.latency_ms,
                    result.metrics.response_length,
                    result.metrics.word_count,
                    result.metrics.input_tokens,
                    result.metrics.output_tokens,
                    result.metrics.total_tokens,
                )
                for result in response.results
            ],
        )
        connection.executemany(
            """
            INSERT INTO benchmark_rankings (
                run_id,
                rank,
                strategy_name,
                status,
                latency_ms,
                response_length,
                word_count,
                total_tokens,
                token_efficiency
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.rank,
                    item.strategy_name,
                    item.status,
                    item.latency_ms,
                    item.response_length,
                    item.word_count,
                    item.total_tokens,
                    item.token_efficiency,
                )
                for item in response.ranking
            ],
        )

    stored_run = get_benchmark_run(run_id)
    if stored_run is None:
        raise RuntimeError("Benchmark run was not persisted.")
    return stored_run


def list_benchmark_runs() -> list[BenchmarkHistoryItem]:
    """Return benchmark runs in newest-first order."""
    init_db()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                benchmark_runs.run_id,
                benchmark_runs.created_at,
                benchmark_runs.user_input,
                COUNT(benchmark_results.id) AS result_count
            FROM benchmark_runs
            LEFT JOIN benchmark_results ON benchmark_results.run_id = benchmark_runs.run_id
            GROUP BY benchmark_runs.run_id
            ORDER BY benchmark_runs.created_at DESC
            """
        ).fetchall()

    return [
        BenchmarkHistoryItem(
            run_id=row["run_id"],
            created_at=row["created_at"],
            user_input=row["user_input"],
            result_count=row["result_count"],
        )
        for row in rows
    ]


def get_benchmark_run(run_id: str) -> BenchmarkRunResponse | None:
    """Return a stored benchmark run and its results."""
    init_db()
    with get_connection() as connection:
        run_row = connection.execute(
            """
            SELECT
                run_id,
                created_at,
                user_input,
                overall_winner,
                fastest_strategy,
                most_detailed_strategy,
                most_token_efficient_strategy,
                benchmark_summary
            FROM benchmark_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if run_row is None:
            return None

        result_rows = connection.execute(
            """
            SELECT
                strategy_name,
                prompt,
                response,
                error_type,
                error_message,
                status,
                latency_ms,
                response_length,
                word_count,
                input_tokens,
                output_tokens,
                total_tokens
            FROM benchmark_results
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
        ranking_rows = connection.execute(
            """
            SELECT
                rank,
                strategy_name,
                status,
                latency_ms,
                response_length,
                word_count,
                total_tokens,
                token_efficiency
            FROM benchmark_rankings
            WHERE run_id = ?
            ORDER BY rank ASC
            """,
            (run_id,),
        ).fetchall()

    return BenchmarkRunResponse(
        run_id=run_row["run_id"],
        created_at=run_row["created_at"],
        user_input=run_row["user_input"],
        results=[
            BenchmarkResult(
                strategy_name=row["strategy_name"],
                prompt=row["prompt"],
                response=row["response"],
                error_type=row["error_type"],
                error_message=row["error_message"],
                metrics=BenchmarkMetrics(
                    status=row["status"],
                    latency_ms=row["latency_ms"],
                    response_length=row["response_length"],
                    word_count=row["word_count"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    total_tokens=row["total_tokens"],
                ),
            )
            for row in result_rows
        ],
        ranking=[
            BenchmarkRankingItem(
                rank=row["rank"],
                strategy_name=row["strategy_name"],
                status=row["status"],
                latency_ms=row["latency_ms"],
                response_length=row["response_length"],
                word_count=row["word_count"],
                total_tokens=row["total_tokens"],
                token_efficiency=row["token_efficiency"],
            )
            for row in ranking_rows
        ],
        winners=BenchmarkWinners(
            overall_winner=run_row["overall_winner"],
            fastest_strategy=run_row["fastest_strategy"],
            most_detailed_strategy=run_row["most_detailed_strategy"],
            most_token_efficient_strategy=run_row["most_token_efficient_strategy"],
        ),
        benchmark_summary=run_row["benchmark_summary"],
    )
