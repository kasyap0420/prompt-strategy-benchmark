from backend.gemini_client import GeminiClient
from backend.schemas import BenchmarkRequest, BenchmarkResponse


class BenchmarkEngine:
    """Coordinates future prompt generation, model calls, metrics, and persistence."""

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        self.gemini_client = gemini_client or GeminiClient()

    def run(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """Future benchmark pipeline entry point."""
        raise NotImplementedError("Benchmark execution will be implemented in Phase 2.")

    def prepare_prompts(self, request: BenchmarkRequest) -> list[str]:
        """Future step for generating prompts from selected strategies."""
        raise NotImplementedError("Prompt preparation will be implemented in Phase 2.")

    def persist_results(self, response: BenchmarkResponse) -> None:
        """Future step for saving benchmark results."""
        raise NotImplementedError("Result persistence will be implemented in Phase 2.")
