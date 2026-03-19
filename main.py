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
    args = parser.parse_args()

    RoleplayEngine(model=args.model, api_key=args.key).run()
