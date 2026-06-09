from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class PromptStrategy:
    name: str
    builder: Callable[[str], str]

    def build_prompt(self, user_input: str) -> str:
        return self.builder(user_input)


def zero_shot_prompt(user_input: str) -> str:
    return user_input


def role_prompt(user_input: str) -> str:
    return f"You are an expert evaluator. Complete the task carefully.\n\nTask:\n{user_input}"


def chain_of_thought_prompt(user_input: str) -> str:
    return f"Reason through the task step by step before giving the final answer.\n\nTask:\n{user_input}"


def structured_prompt(user_input: str) -> str:
    return (
        "Complete the task using a clear structured response.\n\n"
        f"Task:\n{user_input}\n\n"
        "Response format:\n- Answer\n- Rationale\n- Key constraints"
    )


def expert_structured_prompt(user_input: str) -> str:
    return (
        "You are a domain expert. Complete the task with a structured, concise response.\n\n"
        f"Task:\n{user_input}\n\n"
        "Response format:\n- Final answer\n- Reasoning summary\n- Assumptions"
    )


PROMPT_STRATEGIES: dict[str, PromptStrategy] = {
    "Zero-Shot Prompting": PromptStrategy("Zero-Shot Prompting", zero_shot_prompt),
    "Role Prompting": PromptStrategy("Role Prompting", role_prompt),
    "Chain-of-Thought Prompting": PromptStrategy(
        "Chain-of-Thought Prompting",
        chain_of_thought_prompt,
    ),
    "Structured Prompting": PromptStrategy("Structured Prompting", structured_prompt),
    "Expert + Structured Prompting": PromptStrategy(
        "Expert + Structured Prompting",
        expert_structured_prompt,
    ),
}


def list_strategy_names() -> list[str]:
    return list(PROMPT_STRATEGIES.keys())


def build_prompt(strategy_name: str, user_input: str) -> str:
    strategy = PROMPT_STRATEGIES[strategy_name]
    return strategy.build_prompt(user_input)
