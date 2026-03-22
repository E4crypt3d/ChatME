from engine import RoleplayEngine

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChatME Roleplay Engine")
    parser.add_argument(
        "--model",
        "-m",
        help="Model to use (default: openrouter/free). Examples: openrouter/free, arcee-ai/trinity-large-preview:free, deepseek/deepseek-r1:free",
    )
    parser.add_argument(
        "--key",
        "-k",
        help="OpenRouter API key for this session (overrides OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--showmodel",
        "-sm",
        action="store_true",
        help="Show available models and exit (does not start chat)",
    )
    args = parser.parse_args()

    if args.showmodel:
        from engine import DEFAULT_MODEL

        engine = RoleplayEngine(model=args.model, api_key=args.key)
        engine.console.print("[bold]Available Models:[/bold]")
        free = engine._fetch_free_models()
        if free:
            for i, model in enumerate(free, 1):
                short_name = model.split("/")[-1].replace(":free", "")
                engine.console.print(f"  {i}. {short_name}")
            engine.console.print(f"\n[dim]Total: {len(free)} free models[/dim]")
        else:
            engine.console.print("[yellow]No free models available[/yellow]")
            if args.model:
                engine.console.print(f"[dim]Using fallback: {args.model}[/dim]")
            else:
                engine.console.print(f"[dim]Using default: {DEFAULT_MODEL}[/dim]")
    else:
        RoleplayEngine(model=args.model, api_key=args.key).run()
