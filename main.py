import os
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.rule import Rule
from dotenv import load_dotenv

load_dotenv()

LORE_MARKER = "<<<LORE_BLOCK>>>"

# Settings
CONDENSE_THRESHOLD = 22
KEEP_RECENT_TURNS = 10
REPLY_MAX_TOKENS = 400
SUMMARY_MAX_TOKENS = 200


class RoleplayEngine:
    def __init__(self):
        self.console = Console()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.debug = os.environ.get("DEBUG", "false").lower() == "true"
        self.models = [
            "arcee-ai/trinity-large-preview:free",
            "stepfun/step-3.5-flash:free",
            "z-ai/glm-4.5-air:free",
            "arcee-ai/trinity-mini:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        ]
        self.summary_model = "arcee-ai/trinity-mini:free"
        self.history: list[dict] = []
        self.persona_name = "Character"
        self.persona_desc = ""
        self.lore = "The story is just beginning."

    def _build_system_content(self, persona_desc: str, lore: str) -> str:
        return (
            f"ROLEPLAY: {persona_desc}\n\n"
            "RULES:\n"
            "- Stay in character always\n"
            "- Never break character or mention being AI\n"
            "- Use *italics* for actions/thoughts, **bold** for emphasis\n"
            "- Keep responses concise and natural\n\n"
            f"STORY SO FAR:\n{LORE_MARKER}\n{lore}"
        )

    def _update_system_lore(self, new_lore: str) -> None:
        self.lore = new_lore
        content = self.history[0]["content"]
        idx = content.find(LORE_MARKER)
        if idx != -1:
            self.history[0]["content"] = (
                content[: idx + len(LORE_MARKER)] + "\n" + new_lore
            )

    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in m.items() if k in ("role", "content")} for m in messages
        ]

    def call_with_failover(self, messages: list[dict], stream: bool = True):
        clean = self._clean(messages)
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=clean,
                    stream=stream,
                    max_tokens=REPLY_MAX_TOKENS,
                    timeout=45.0,
                )
                return response, model
            except Exception as e:
                if self.debug:
                    self.console.print(f"[dim red]  ↳ {model}: {e}[/dim red]")
                else:
                    self.console.print(
                        f"[dim red]Model {model} unavailable, trying next…[/dim red]"
                    )
        return None, None

    def _summarise(self, messages: list[dict]) -> str | None:
        try:
            result = self.client.chat.completions.create(
                model=self.summary_model,
                messages=self._clean(messages),
                stream=False,
                max_tokens=SUMMARY_MAX_TOKENS,
                timeout=30.0,
            )
            return (result.choices[0].message.content or "").strip() or None
        except Exception as e:
            if self.debug:
                self.console.print(f"[dim red]Summarise failed: {e}[/dim red]")
            return None

    def _consume_stream(self, response_obj) -> str:
        content = ""
        try:
            with Live(Markdown(""), auto_refresh=True, console=self.console) as live:
                for chunk in response_obj:
                    if not chunk.choices:
                        continue
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        content += token
                        live.update(Panel(Markdown(content), border_style="magenta"))
        except Exception:
            pass
        return content

    def _stream_response(self, response_obj, model_used: str) -> str:
        content = self._consume_stream(response_obj)

        # retry if response is too short
        if len(content.strip()) < 20:
            self.console.print("[dim yellow]⚠ Stream dropped, retrying…[/dim yellow]")
            for model in [m for m in self.models if m != model_used]:
                try:
                    retry = self.client.chat.completions.create(
                        model=model,
                        messages=self._clean(self.history),
                        stream=True,
                        max_tokens=REPLY_MAX_TOKENS,
                        timeout=45.0,
                    )
                    content = self._consume_stream(retry)
                    if len(content.strip()) >= 20:
                        break
                except Exception:
                    continue

        return content

    def condense_logic(self) -> None:
        self.console.print(
            "\n[dim italic yellow]⚡ Consolidating memories…[/dim italic yellow]"
        )

        to_summarise = self.history[1:-KEEP_RECENT_TURNS]
        if not to_summarise:
            return

        # Build transcript for summarization
        lines = []
        for msg in to_summarise:
            label = self.persona_name if msg["role"] == "assistant" else "User"
            lines.append(f"{label}: {msg['content']}")
        transcript = "\n".join(lines)

        summarise_prompt = [
            {
                "role": "system",
                "content": "You summarise roleplay sessions. Be brief and factual. Bullet points only.",
            },
            {
                "role": "user",
                "content": (
                    f"Summarise into a 'Story So Far' block (max 5 bullets).\n"
                    f"Cover: setting, key events, emotional tone, {self.persona_name}'s mood.\n\n"
                    f"{transcript}"
                ),
            },
        ]

        new_lore = self._summarise(summarise_prompt)
        if new_lore:
            self._update_system_lore(new_lore)
            if self.debug:
                self.console.print(f"[dim yellow]DEBUG lore:\n{new_lore}[/dim yellow]")

        # Trim history but keep recent turns
        self.history = [self.history[0]] + self.history[-KEEP_RECENT_TURNS:]

    @staticmethod
    def _extract_persona_name(raw: str) -> str:
        skip = {
            "you",
            "i",
            "im",
            "i'm",
            "am",
            "are",
            "is",
            "my",
            "the",
            "a",
            "an",
            "be",
            "as",
            "play",
            "playing",
            "named",
            "called",
        }
        for word in raw.split():
            clean = word.strip(".,!?\"'").lower()
            if clean not in skip and len(clean) > 1 and clean.isalpha():
                return word.strip(".,!?\"'").capitalize()
        return "Character"

    @staticmethod
    def _trim_user_self_reference(raw: str) -> str:
        # Remove player's self-intro from character description
        markers = [
            ", im ",
            ", i am ",
            ", i'm ",
            " im ",
            " i am ",
            "i'm ",
            " my character ",
        ]
        lower = raw.lower()
        for m in markers:
            idx = lower.find(m)
            if idx != -1:
                return raw[:idx].strip()
        return raw

    def run(self) -> None:
        self.console.clear()
        self.console.print(
            Panel(
                "[bold white]✦ ChatME ROLEPLAY ENGINE ✦[/bold white]",
                style="bold blue",
                expand=False,
            )
        )

        raw_persona = Prompt.ask(
            "[bold green]Who am I roleplaying as?[/bold green] [dim](character name + description)[/dim]"
        )

        self.persona_name = self._extract_persona_name(raw_persona)
        self.persona_desc = self._trim_user_self_reference(raw_persona)

        self.history.append(
            {
                "role": "system",
                "content": self._build_system_content(self.persona_desc, self.lore),
            }
        )

        self.console.print(
            f"\n[bold blue]Entering roleplay as[/bold blue] "
            f"[bold magenta]{self.persona_name}[/bold magenta]. "
            "Type [bold red]exit[/bold red] to quit.\n"
        )

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            except KeyboardInterrupt:
                self.console.print("\n[bold green]Goodbye![/bold green]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "gracefulshutdown", "shutdown"}:
                self.console.print("[bold green]Goodbye![/bold green]")
                break

            self.history.append({"role": "user", "content": user_input})

            # Condense if history gets too long
            if len(self.history) > CONDENSE_THRESHOLD:
                self.condense_logic()

            self.console.print(Rule(style="dim"))
            self.console.print(f"[bold magenta]{self.persona_name}[/bold magenta]")

            response_obj, model_used = self.call_with_failover(self.history)

            if response_obj is None:
                self.console.print(
                    "[bold red]All models failed. Check your OpenRouter status.[/bold red]"
                )
                self.history.pop()
                continue

            if self.debug and model_used:
                self.console.print(f"[dim yellow]  ↳ using {model_used}[/dim yellow]")

            content = self._stream_response(response_obj, model_used)

            if not content:
                self.console.print(
                    "[dim red]Empty response — please try again.[/dim red]"
                )
                self.history.pop()
                continue

            self.history.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    RoleplayEngine().run()
