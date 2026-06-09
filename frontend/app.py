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
