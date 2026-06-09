import streamlit as st


st.set_page_config(page_title="Prompt Strategy Benchmark", layout="wide")

st.title("Prompt Strategy Benchmark")
st.write(
    "A foundation for comparing prompt strategies against model responses, "
    "objective metrics, and exportable benchmark results."
)

st.header("Benchmark Setup")
st.text_area("User input", placeholder="Enter the task or prompt to benchmark.")
st.multiselect(
    "Prompt strategies",
    options=[
        "Zero-Shot Prompting",
        "Role Prompting",
        "Chain-of-Thought Prompting",
        "Structured Prompting",
        "Expert + Structured Prompting",
    ],
)

st.header("Results")
st.info("Benchmark execution will be added in the next phase.")
