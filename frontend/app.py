import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("PROMPT_BENCHMARK_API_URL", "http://127.0.0.1:8000")


def api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def run_benchmark(user_input: str) -> dict[str, Any]:
    response = requests.post(
        api_url("/benchmark"),
        json={"user_input": user_input},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def fetch_export(run_id: str, export_format: str) -> bytes:
    response = requests.get(
        api_url(f"/benchmark/{run_id}/export/{export_format}"),
        timeout=30,
    )
    response.raise_for_status()
    return response.content


def fetch_history() -> list[dict[str, Any]]:
    response = requests.get(api_url("/benchmark/history"), timeout=30)
    response.raise_for_status()
    return response.json()["runs"]


def fetch_run(run_id: str) -> dict[str, Any]:
    response = requests.get(api_url(f"/benchmark/{run_id}"), timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_analytics_summary() -> dict[str, Any]:
    response = requests.get(api_url("/analytics/summary"), timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_strategy_performance() -> list[dict[str, Any]]:
    response = requests.get(api_url("/analytics/strategy-performance"), timeout=30)
    response.raise_for_status()
    return response.json()["strategies"]


def fetch_analytics_history() -> list[dict[str, Any]]:
    response = requests.get(api_url("/analytics/history"), timeout=30)
    response.raise_for_status()
    return response.json()["runs"]


def result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "Strategy": result["strategy_name"],
                "Status": metrics["status"],
                "Error type": result.get("error_type"),
                "Latency (ms)": metrics["latency_ms"],
                "Words": metrics["word_count"],
                "Characters": metrics["response_length"],
                "Input tokens": metrics["input_tokens"],
                "Output tokens": metrics["output_tokens"],
                "Total tokens": metrics["total_tokens"],
            }
        )
    return rows


def ranking_rows(ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Rank": item["rank"],
            "Strategy": item["strategy_name"],
            "Status": item["status"],
            "Latency": item["latency_ms"],
            "Word Count": item["word_count"],
            "Token Count": item["total_tokens"],
        }
        for item in ranking
    ]


def render_analytics_dashboard() -> None:
    st.divider()
    st.header("Analytics Dashboard")

    try:
        summary = fetch_analytics_summary()
        performance = fetch_strategy_performance()
        analytics_history = fetch_analytics_history()
    except requests.RequestException as exc:
        st.warning(f"Analytics unavailable: {exc}")
        return

    summary_cols = st.columns(5)
    summary_cols[0].metric("Runs", summary["total_runs"])
    summary_cols[1].metric("Results", summary["total_results"])
    summary_cols[2].metric("Successes", summary["total_successes"])
    summary_cols[3].metric("Failures", summary["total_failures"])
    summary_cols[4].metric("Success Rate", f"{summary['overall_success_rate'] * 100:.1f}%")

    if not performance:
        st.info("No benchmark analytics yet. Run a benchmark to populate this dashboard.")
        return

    performance_df = pd.DataFrame(performance).set_index("strategy_name")

    chart_cols_1 = st.columns(2)
    with chart_cols_1[0]:
        st.subheader("Strategy Win Distribution")
        st.bar_chart(performance_df["wins"])
    with chart_cols_1[1]:
        st.subheader("Average Latency By Strategy")
        st.bar_chart(performance_df["avg_latency_ms"])

    chart_cols_2 = st.columns(2)
    with chart_cols_2[0]:
        st.subheader("Success Rate By Strategy")
        st.bar_chart(performance_df["success_rate"])
    with chart_cols_2[1]:
        st.subheader("Average Response Length By Strategy")
        st.bar_chart(performance_df["avg_response_length"])

    st.subheader("Token Usage By Strategy")
    token_columns = ["avg_input_tokens", "avg_output_tokens", "avg_total_tokens"]
    st.bar_chart(performance_df[token_columns].fillna(0))

    st.subheader("Historical Run Overview")
    if analytics_history:
        history_df = pd.DataFrame(analytics_history)
        overview_df = history_df[["created_at", "success_rate", "successes", "failures"]].set_index(
            "created_at"
        )
        st.line_chart(overview_df[["success_rate"]])
        st.dataframe(
            history_df[
                [
                    "created_at",
                    "user_input",
                    "overall_winner",
                    "success_rate",
                    "successes",
                    "failures",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No historical runs are available yet.")


st.set_page_config(page_title="Prompt Strategy Benchmark", layout="wide")

st.title("Prompt Strategy Benchmark")

with st.sidebar:
    st.subheader("Backend")
    st.caption(API_BASE_URL)
    if st.button("Check health", use_container_width=True):
        try:
            health_response = requests.get(api_url("/health"), timeout=10)
            health_response.raise_for_status()
            st.success("Backend is healthy.")
        except requests.RequestException as exc:
            st.error(f"Backend health check failed: {exc}")

    st.subheader("Historical Run Explorer")
    try:
        history_runs = fetch_history()
    except requests.RequestException as exc:
        history_runs = []
        st.caption(f"History unavailable: {exc}")

    if history_runs:
        history_options = {
            f"{run['created_at']} - {run['user_input'][:40]}": run["run_id"]
            for run in history_runs
        }
        selected_history = st.selectbox(
            "Stored runs",
            options=list(history_options.keys()),
            index=None,
            placeholder="Select a historical run",
        )
        if selected_history and st.button("Open selected run", use_container_width=True):
            try:
                st.session_state.benchmark_run = fetch_run(history_options[selected_history])
                st.success("Historical run loaded.")
            except requests.RequestException as exc:
                st.error(f"Could not load run: {exc}")
    else:
        st.caption("No saved runs yet.")

user_input = st.text_area(
    "User input",
    height=180,
    placeholder="Enter the task or prompt to benchmark.",
)

if "benchmark_run" not in st.session_state:
    st.session_state.benchmark_run = None

if st.button("Run Benchmark", type="primary", use_container_width=True):
    cleaned_input = user_input.strip()
    if not cleaned_input:
        st.warning("Enter user input before running a benchmark.")
    else:
        with st.spinner("Running all prompt strategies..."):
            try:
                st.session_state.benchmark_run = run_benchmark(cleaned_input)
                st.success("Benchmark completed and saved.")
            except requests.RequestException as exc:
                st.error(f"Benchmark failed: {exc}")

benchmark_run = st.session_state.benchmark_run

if benchmark_run:
    run_id = benchmark_run["run_id"]
    st.divider()
    if benchmark_run.get("benchmark_summary"):
        st.subheader("Benchmark Summary")
        st.info(benchmark_run["benchmark_summary"])

    winners = benchmark_run.get("winners") or {}
    st.subheader("Winners")
    winner_cols = st.columns(4)
    winner_labels = [
        ("Overall Winner", winners.get("overall_winner")),
        ("Fastest", winners.get("fastest_strategy")),
        ("Most Detailed", winners.get("most_detailed_strategy")),
        ("Most Efficient", winners.get("most_token_efficient_strategy")),
    ]
    for column, (label, value) in zip(winner_cols, winner_labels):
        column.metric(label, value or "None")

    if benchmark_run.get("ranking"):
        st.subheader("Ranking")
        ranking_df = pd.DataFrame(ranking_rows(benchmark_run["ranking"]))
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    st.subheader("Results")
    st.caption(f"Run ID: {run_id}")

    metrics_df = pd.DataFrame(result_rows(benchmark_run["results"]))
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    export_col_1, export_col_2 = st.columns(2)
    with export_col_1:
        try:
            json_export = fetch_export(run_id, "json")
            st.download_button(
                "Download JSON",
                data=json_export,
                file_name=f"{run_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        except requests.RequestException as exc:
            st.error(f"JSON export unavailable: {exc}")
    with export_col_2:
        try:
            csv_export = fetch_export(run_id, "csv")
            st.download_button(
                "Download CSV",
                data=csv_export,
                file_name=f"{run_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except requests.RequestException as exc:
            st.error(f"CSV export unavailable: {exc}")

    for result in benchmark_run["results"]:
        metrics = result["metrics"]
        with st.expander(f"{result['strategy_name']} - {metrics['status']}", expanded=False):
            st.markdown("**Prompt**")
            st.code(result["prompt"], language="text")
            st.markdown("**Response**")
            if result["response"]:
                st.write(result["response"])
            else:
                st.error(result.get("error_message") or "No response was returned.")
                if result.get("error_type"):
                    st.caption(f"Error type: {result['error_type']}")
else:
    st.info("Run a benchmark to see saved results and exports.")

render_analytics_dashboard()
