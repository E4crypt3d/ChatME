from __future__ import annotations

import sys
import traceback
import argparse

from engine import RoleplayEngine, PROVIDERS, DEFAULT_PROVIDER, DEFAULT_MODEL


def _handle_exc(t, v, tb):
    if issubclass(t, KeyboardInterrupt):
        sys.__excepthook__(t, v, tb)
        return
    traceback.print_exception(t, v, tb)


sys.excepthook = _handle_exc


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ChatME — Immersive AI Roleplay Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # OpenRouter, auto free-model selection
  python main.py -p groq                      # Use Groq provider
  python main.py -m meta-llama/llama-3-8b-instruct:free
  python main.py -p groq -m llama-3.3-70b-versatile
  python main.py --showmodel                  # List available free models and exit
""",
    )
    ap.add_argument("--model", "-m", help="Model ID override")
    ap.add_argument("--key", "-k", help="API key (overrides env var)")
    ap.add_argument(
        "--provider",
        "-p",
        default=DEFAULT_PROVIDER,
        choices=list(PROVIDERS.keys()),
        help=f"Provider to use  (default: {DEFAULT_PROVIDER})",
    )
    ap.add_argument(
        "--showmodel",
        "-sm",
        action="store_true",
        help="List available free models and exit",
    )
    args = ap.parse_args()

    engine = RoleplayEngine(model=args.model, api_key=args.key, provider=args.provider)

    if args.showmodel:
        engine.console.print("\n[bold]Available Free Models:[/bold]")
        free = engine._fetch_free_models()
        if free:
            for i, m in enumerate(free, 1):
                short = m.split("/")[-1].replace(":free", "")
                engine.console.print(f"  [dim]{i:3}.[/dim]  {short}  [dim]{m}[/dim]")
            engine.console.print(f"\n[dim]Total: {len(free)} models[/dim]")
        else:
            engine.console.print("[yellow]No free models found.[/yellow]")
            fallback = args.model or PROVIDERS[args.provider]["default_model"]
            engine.console.print(f"[dim]Fallback: {fallback}[/dim]")
        return

    try:
        engine.run()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
