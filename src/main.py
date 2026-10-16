"""Interactive CLI runner for the PC Builder Agent."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from src.agent.graph import app

console = Console()


def run_agent(query: str):
    console.print(
        Panel(
            f"[bold cyan]User Request:[/bold cyan] {query}",
            border_style="cyan",
        )
    )

    initial_state = {
        "user_query": query,
        "requirements": None,
        "budget_allocation": None,
        "current_build": None,
        "validation_result": None,
        "iteration_count": 0,
        "feedback_history": [],
        "final_response": None,
    }

    with console.status("[bold green]Agent is analyzing requirements and validating hardware compatibility...[/bold green]"):
        final_state = app.invoke(initial_state)

    response_text = final_state.get("final_response", "Failed to generate response.")
    console.print(Markdown(response_text))


def main():
    console.print(
        Panel.fit(
            "[bold green]🤖 PC Builder Autonomous Agent[/bold green]\n"
            "[dim]LangGraph + Deterministic Hardware Compatibility Engine[/dim]",
            border_style="green",
        )
    )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_agent(query)
        return

    # Interactive mode
    console.print("[yellow]Enter your PC requirements (or 'exit' to quit):[/yellow]")
    console.print("[dim]Example: 'Build a PC for Cyberpunk 2077 and CS2 at 1440p, budget $1300'[/dim]\n")

    while True:
        try:
            query = console.input("[bold blue]You > [/bold blue]")
            if not query.strip():
                continue
            if query.strip().lower() in ["exit", "quit", "q"]:
                console.print("[dim]Exiting...[/dim]")
                break
            run_agent(query)
            console.print("\n" + "=" * 60 + "\n")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
