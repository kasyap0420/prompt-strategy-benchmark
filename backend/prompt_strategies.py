from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class PromptVariant:
    strategy_name: str
    prompt: str


@dataclass(frozen=True, slots=True)
class PromptStrategy:
    name: str
    builder: Callable[[str], str]

    def build_prompt(self, user_input: str) -> str:
        return self.builder(user_input)


def _clean_user_input(user_input: str) -> str:
    cleaned_input = user_input.strip()
    if not cleaned_input:
        raise ValueError("User input cannot be empty.")
    return cleaned_input


def zero_shot_prompt(user_input: str) -> str:
    return f"Task:\n{user_input}"


def role_prompt(user_input: str) -> str:
    return (
        "You are a careful, practical assistant focused on accurate and useful answers.\n\n"
        f"Task:\n{user_input}"
    )


def structured_prompt(user_input: str) -> str:
    return (
        "Complete the task and organize the response in a clear structure.\n\n"
        f"Task:\n{user_input}\n\n"
        "Use this response format:\n"
        "1. Answer\n"
        "2. Key points\n"
        "3. Assumptions or constraints"
    )


def expert_prompt(user_input: str) -> str:
    return (
        "Act as a senior domain expert. Prioritize accuracy, relevant context, and practical judgment.\n\n"
        f"Task:\n{user_input}"
    )


def reasoning_oriented_prompt(user_input: str) -> str:
    return (
        "Analyze the task carefully before answering. Provide the final answer with a brief reasoning summary, "
        "without exposing private chain-of-thought.\n\n"
        f"Task:\n{user_input}"
    )


PROMPT_STRATEGIES: tuple[PromptStrategy, ...] = (
    PromptStrategy("Zero-Shot", zero_shot_prompt),
    PromptStrategy("Role Prompting", role_prompt),
    PromptStrategy("Structured Prompting", structured_prompt),
    PromptStrategy("Expert Prompting", expert_prompt),
    PromptStrategy("Reasoning-Oriented Prompting", reasoning_oriented_prompt),
)

_PROMPT_STRATEGY_BY_NAME: dict[str, PromptStrategy] = {
    strategy.name: strategy for strategy in PROMPT_STRATEGIES
}


def list_strategy_names() -> list[str]:
    return [strategy.name for strategy in PROMPT_STRATEGIES]


def build_prompt(strategy_name: str, user_input: str) -> str:
    cleaned_input = _clean_user_input(user_input)
    try:
        strategy = _PROMPT_STRATEGY_BY_NAME[strategy_name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt strategy: {strategy_name}") from exc
    return strategy.build_prompt(cleaned_input)


def generate_prompt_variants(user_input: str) -> list[PromptVariant]:
    cleaned_input = _clean_user_input(user_input)
    return [
        PromptVariant(strategy_name=strategy.name, prompt=strategy.build_prompt(cleaned_input))
        for strategy in PROMPT_STRATEGIES
    ]