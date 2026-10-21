"""Interactive CLI runner for the PC Builder Agent."""

import io
import sys
from pathlib import Path

# Ensure project root and virtual environment site-packages are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

venv_site = ROOT_DIR / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

# Force UTF-8 output on Windows to support emoji / box-drawing characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from src.agent.graph import app
from src.agent.nodes import _determine_workload_profile

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
        "parser_source": None,
    }

    with console.status("[bold green]Agent is analyzing requirements and validating hardware compatibility...[/bold green]"):
        final_state = app.invoke(initial_state)

    # Display Requirements Recognition Log
    req = final_state.get("requirements")
    source = final_state.get("parser_source", "Deterministic Heuristic Engine")
    if req:
        profile = _determine_workload_profile(req)
        profile_names = {
            "gaming": "🎮 High-Performance Gaming",
            "productivity": "💼 Workstation & Development",
            "ai_3d": "🧠 AI / ML & 3D Rendering (Blender/CUDA)",
            "office": "📄 General Office & Web Surfing",
        }
        profile_label = profile_names.get(profile, profile)

        games_str = ", ".join(req.target_games) if req.target_games else "None"
        tasks_str = ", ".join(req.work_tasks) if req.work_tasks else "None"
        priorities_str = ", ".join(req.priorities) if req.priorities else "Default balance"
        res_str = req.resolution.upper() if req.resolution else "Auto (Workload-based)"

        log_lines = [
            f"[bold yellow]⚙️ Parsing Source:[/bold yellow] [bold green]{source}[/bold green]",
            f"[bold yellow]🎯 Detected Profile:[/bold yellow] [bold cyan]{profile_label}[/bold cyan]",
            f"[bold yellow]💰 Parsed Budget:[/bold yellow] ${req.budget_usd:,}",
            f"[bold yellow]🛠️ Workload Tasks:[/bold yellow] {tasks_str}",
            f"[bold yellow]🕹️ Target Games:[/bold yellow] {games_str}",
            f"[bold yellow]🖥️ Target Resolution:[/bold yellow] {res_str}",
            f"[bold yellow]⚡ Hardware Priorities:[/bold yellow] {priorities_str}",
        ]
        console.print(
            Panel(
                "\n".join(log_lines),
                title="[bold green]🧠 Requirements Recognition Log[/bold green]",
                border_style="green",
            )
        )

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
