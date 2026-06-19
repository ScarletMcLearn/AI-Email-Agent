"""Interactive command-line interface."""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.config import ConfigurationError, Settings
from src.generator import EmailGenerator
from src.llm_client import LLMError, create_llm_client
from src.models import Strategy


def collect_key_facts(
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> list[str]:
    """Collect one key fact per line until the user enters a blank line."""
    print_fn("Enter key facts one per line. Press Enter on a blank line when finished.")
    facts: list[str] = []
    while True:
        fact = input_fn(f"Fact {len(facts) + 1}: ").strip()
        if not fact:
            break
        facts.append(fact)
    return facts


def run_interactive(
    generator: EmailGenerator,
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> str:
    """Run one interactive email generation session."""
    print_fn("Email Generation Assistant")
    print_fn("--------------------------")
    intent = input_fn("Intent: ").strip()
    facts = collect_key_facts(input_fn, print_fn)
    tone = input_fn("Tone: ").strip()
    strategy_value = (
        input_fn("Strategy [A=advanced, B=baseline] (default A): ").strip() or "A"
    )
    strategy = Strategy.from_user_input(strategy_value)
    email = generator.generate(intent, facts, tone, strategy)
    print_fn("\nGenerated email\n---------------")
    print_fn(email)
    return email


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = Settings.from_env()
        client = create_llm_client(settings)
        generator = EmailGenerator(client, settings)
        run_interactive(generator)
        return 0
    except (ConfigurationError, LLMError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
