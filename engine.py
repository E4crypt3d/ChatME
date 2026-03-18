import os
import re
import sys
import json
import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI, APIError, APITimeoutError
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from dotenv import load_dotenv

from constants import (
    CHARACTERS_MARKER,
    LORE_MARKER,
    RELATION_WORDS,
    LOCATION_STOPWORDS,
    is_valid_name,
)
from models import WorldMemory

load_dotenv()


class RoleplayEngine:
    CONDENSE_THRESHOLD_MSGS = 30
    CONDENSE_THRESHOLD_TOKENS = 5000
    KEEP_RECENT_PAIRS = 8
    MAX_HISTORY_HARD_CAP = 80

    REPLY_MAX_TOKENS = 500
    SUMMARY_MAX_TOKENS = 350
    REQUEST_TIMEOUT = 90.0
    SUMMARY_TIMEOUT = 50.0
    MIN_RESPONSE_LEN = 10

    SESSIONS_DIR = Path("chatme_sessions")

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        if sys.platform == "win32":
            import io

            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        self.console = Console(force_terminal=True)
        self._override_model = model
        self._override_api_key = api_key
        self._init_client()
        self._init_state()

    def _init_client(self) -> None:
        # Use override API key if provided, otherwise fall back to env var
        api_key = self._override_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.console.print("[bold red]OPENROUTER_API_KEY not set[/bold red]")
            sys.exit(1)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=self.REQUEST_TIMEOUT,
        )
        # Use override model if provided, otherwise use default model list
        if self._override_model:
            self.models: list[str] = [self._override_model]
            self.summary_models: list[str] = [self._override_model]
        else:
            self.models: list[str] = [
                "arcee-ai/trinity-large-preview:free",
                "stepfun/step-3.5-flash:free",
                "z-ai/glm-4.5-air:free",
                "arcee-ai/trinity-mini:free",
                "nvidia/nemotron-3-super-120b-a12b:free",
                "nvidia/nemotron-3-nano-30b-a3b:free",
                "liquid/lfm-2.5-1.2b-instruct:free",
                "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            ]
            self.summary_models: list[str] = [
                "arcee-ai/trinity-mini:free",
                "z-ai/glm-4.5-air:free",
                "stepfun/step-3.5-flash:free",
            ]
        self._last_working_model: Optional[str] = None
        self._failed_models: set[str] = set()

    def _init_state(self) -> None:
        self.debug = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
        self.history: list[dict] = []
        self.persona_name = "Character"
        self.persona_desc = ""
        self.player_label = "You"
        self.lore = "The story is just beginning."
        self.memory = WorldMemory()
        self._msg_count = 0
        self._condense_count = 0
        # scene/mood context for system prompt
        self.scene: str = ""
        self.mood: str = ""
        self._last_assistant_content: str = ""
        # assistant describes scene+action, not just dialogue
        self.narrate_mode: bool = False

    def _build_system_content(self, persona_desc: str, lore: str) -> str:
        world_info = self.memory.format_world()
        scene_block = f"\nCURRENT SCENE: {self.scene}" if self.scene else ""
        mood_block = f"\nCHARACTER MOOD: {self.mood}" if self.mood else ""
        narrate_block = (
            "\n- Write responses as a narrator: describe the scene, actions, AND dialogue together"
            if self.narrate_mode
            else ""
        )
        return (
            f"CHARACTER IDENTITY (permanent — never changes):\n{persona_desc}"
            f"{scene_block}{mood_block}\n\n"
            "RULES:\n"
            "- Stay in character at all times\n"
            "- Never break character or mention being AI\n"
            "- Use *italics* for actions/thoughts, **bold** for emphasis\n"
            "- Keep responses natural and concise (2-4 sentences)\n"
            "- Your PRIMARY task: respond directly to the LAST USER MESSAGE\n"
            "- NEVER repeat a phrase, sentence, or action you already used this conversation\n"
            "- NEVER reuse the same opening action (e.g. 'smiles softly') more than once\n"
            f"- STORY SO FAR is background only — do not re-enact or echo it{narrate_block}\n\n"
            f"CHARACTER DATABASE:\n{CHARACTERS_MARKER}\n{world_info}\n{CHARACTERS_MARKER}\n\n"
            "STORY SO FAR (past events — these already happened, do not revisit them):\n"
            f"{LORE_MARKER}\n{lore}\n{LORE_MARKER}"
        )

    def _rebuild_system(self) -> None:
        if self.history and self.history[0]["role"] == "system":
            self.history[0]["content"] = self._build_system_content(
                self.persona_desc, self.lore
            )

    def _patch_system_marker(self, marker: str, new_block: str) -> None:
        if not self.history or self.history[0]["role"] != "system":
            return
        pat = re.compile(rf"{re.escape(marker)}.*?{re.escape(marker)}", re.DOTALL)
        replace = f"{marker}\n{new_block}\n{marker}"
        content = self.history[0]["content"]
        self.history[0]["content"] = (
            pat.sub(replace, content)
            if pat.search(content)
            else content + f"\n\n{replace}"
        )

    def _update_system_lore(self, new_lore: str) -> None:
        self.lore = new_lore
        # rebuild keeps scene/mood/narrate in sync
        self._rebuild_system()

    def _update_system_memory(self) -> None:
        self._patch_system_marker(CHARACTERS_MARKER, self.memory.format_world())

    def _extract_info_from_message(self, text: str, is_user: bool) -> None:
        # explicit introduction patterns
        for pat in [
            r"(?:i am|i'm|my name is)\s+([A-Z][a-z]{1,})",
            r"(?:this is|meet|introduce[d]?)\s+([A-Z][a-z]{1,})",
            r"(?:call(?:ed)?|named?)\s+([A-Z][a-z]{1,})",
        ]:
            for m in re.finditer(pat, text):
                if is_valid_name(m.group(1)):
                    self.memory.add_character(m.group(1), context=text[:80])

        # age pattern
        for m in re.finditer(
            r"\b([A-Z][a-z]{1,})\s+is\s+(\d{1,3})\s*(?:years?\s*old)?", text
        ):
            name, age = m.group(1), m.group(2)
            if is_valid_name(name):
                c = self.memory.add_character(name, age=age, context=text[:80])
                if c and not c.age:
                    c.age = age

        # "[Name] is my/his/her [relation]"
        for m in re.finditer(
            r"\b([A-Z][a-z]{1,})\s+is\s+(?:my|his|her|their|our)\s+(\w+)", text
        ):
            name, rel = m.group(1), m.group(2).lower()
            if is_valid_name(name) and rel in RELATION_WORDS:
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        # "my/his/her [relation] [Name]"
        for m in re.finditer(
            r"(?:my|his|her|their|our)\s+(\w+)\s+([A-Z][a-z]{1,})", text
        ):
            rel, name = m.group(1).lower(), m.group(2)
            if rel in RELATION_WORDS and is_valid_name(name):
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        # "[Name] and [Name] are [word]"
        for m in re.finditer(
            r"\b([A-Z][a-z]{1,})\s+and\s+([A-Z][a-z]{1,})\s+are\s+(\w+)", text
        ):
            n1, n2, rt = m.group(1), m.group(2), m.group(3).lower()
            if is_valid_name(n1) and is_valid_name(n2):
                self.memory.add_character(n1, context=text[:80])
                self.memory.add_character(n2, context=text[:80])
                self.memory.add_relationship(n1, n2, rt, context=text[:80])

        # location after spatial preposition
        known_names = {c.lower() for c in self.memory.characters}
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in {
                "in",
                "at",
                "near",
                "inside",
                "outside",
                "through",
                "across",
                "into",
            } and i + 1 < len(words):
                cand = words[i + 1].strip(".,!?;:\"'")
                if (
                    len(cand) > 2
                    and cand[0].isupper()
                    and cand.isalpha()
                    and cand.lower() not in LOCATION_STOPWORDS
                    and cand.lower() not in known_names
                ):
                    self.memory.add_location(cand)

    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if isinstance(m, dict)
            and m.get("role") in ("system", "user", "assistant")
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        ]

    def _estimate_tokens(self, messages: list[dict]) -> int:
        return sum(len(m.get("content", "")) // 4 + 1 for m in messages)

    def call_with_failover(
        self,
        messages: list[dict],
        stream: bool = True,
        is_summary: bool = False,
    ) -> tuple[Optional[object], Optional[str]]:
        clean = self._clean(messages)
        if not clean:
            return None, None

        if is_summary:
            models_to_try = list(self.summary_models)
        else:
            ordered = (
                [self._last_working_model]
                if self._last_working_model and self._last_working_model in self.models
                else []
            )
            untried = [
                m
                for m in self.models
                if m not in ordered and m not in self._failed_models
            ]
            fallback = [
                m for m in self.models if m not in ordered and m in self._failed_models
            ]
            models_to_try = ordered + untried + fallback

        last_error: Optional[str] = None
        for model in models_to_try:
            try:
                kwargs: dict = dict(
                    model=model,
                    messages=clean,
                    stream=stream,
                    max_tokens=(
                        self.SUMMARY_MAX_TOKENS if is_summary else self.REPLY_MAX_TOKENS
                    ),
                    timeout=(
                        self.SUMMARY_TIMEOUT if is_summary else self.REQUEST_TIMEOUT
                    ),
                )
                if not is_summary:
                    kwargs["temperature"] = 0.9
                resp = self.client.chat.completions.create(**kwargs)
                self._last_working_model = model
                self._failed_models.discard(model)
                return resp, model
            except APITimeoutError:
                self._failed_models.add(model)
                last_error = f"timeout ({model.split('/')[-1]})"
                if self.debug:
                    self.console.print(f"[dim red]timeout: {model}[/dim red]")
            except APIError as e:
                self._failed_models.add(model)
                code = getattr(e, "code", "?")
                last_error = f"api {code} ({model.split('/')[-1]})"
                if self.debug:
                    self.console.print(f"[dim red]api error {code}: {model}[/dim red]")
            except Exception as e:
                self._failed_models.add(model)
                last_error = str(e)[:60]
                if self.debug:
                    self.console.print(f"[dim red]error {model}: {e}[/dim red]")

        self.console.print(
            f"[bold red]all models unavailable[/bold red] — {last_error}\n"
            "[yellow]check OPENROUTER_API_KEY or wait and retry[/yellow]"
        )
        return None, None

    def _consume_stream(self, response_obj) -> str:
        parts: list[str] = []
        try:
            # Live auto-refreshes display
            with Live(
                Panel("", border_style="magenta"),
                auto_refresh=True,
                console=self.console,
                transient=True,
                refresh_per_second=12,
            ) as live:
                for chunk in response_obj:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    token = (delta.content or "") if delta else ""
                    if token:
                        parts.append(token)
                        if len(parts) % 5 == 0:
                            live.update(
                                Panel(Markdown("".join(parts)), border_style="magenta")
                            )
                if parts:
                    live.update(Panel(Markdown("".join(parts)), border_style="magenta"))
        except Exception as e:
            if self.debug:
                self.console.print(f"[dim red]stream error: {e}[/dim red]")
        content = "".join(parts)
        # only print panel if we got content
        if content.strip():
            self.console.print(Panel(Markdown(content), border_style="magenta"))
        return content

    def _get_reply(self) -> str:
        resp, used_model = self.call_with_failover(self.history, stream=True)
        if resp is None:
            return ""
        content = self._consume_stream(resp)
        # force different model for short/empty response
        if len(content.strip()) < self.MIN_RESPONSE_LEN:
            if used_model:
                self._failed_models.add(used_model)
            resp2, _ = self.call_with_failover(self.history, stream=True)
            if used_model:
                self._failed_models.discard(used_model)
            if resp2:
                c2 = self._consume_stream(resp2)
                if len(c2.strip()) >= self.MIN_RESPONSE_LEN:
                    return c2
        return content

    def _convo_msg_count(self) -> int:
        return sum(1 for m in self.history if m.get("role") != "system")

    def _should_condense(self) -> bool:
        n = self._convo_msg_count()
        # need new messages before re-condensing
        keep_n = self.KEEP_RECENT_PAIRS * 2
        threshold = (
            (keep_n + self.CONDENSE_THRESHOLD_MSGS)
            if self._condense_count > 0
            else self.CONDENSE_THRESHOLD_MSGS
        )
        if len(self.history) > self.MAX_HISTORY_HARD_CAP:
            return True
        if n >= threshold:
            return True
        if self._estimate_tokens(self.history) > self.CONDENSE_THRESHOLD_TOKENS:
            return True
        return False

    def _summarise(self, messages: list[dict]) -> Optional[str]:
        if not messages:
            return None
        lines = []
        for m in messages:
            role = "Assistant" if m["role"] == "assistant" else "User"
            cleaned = re.sub(r"\*[^*]+\*", "", m.get("content", "")).strip()
            if cleaned:
                lines.append(f"[{role}]: {cleaned}")
        transcript = "\n".join(lines)
        if len(transcript) > 6000:
            transcript = transcript[-6000:]

        known_chars = ", ".join(self.memory.characters.keys()) or "none"
        prompt = [
            {
                "role": "system",
                "content": (
                    "You write brief story notes for a roleplay session. "
                    "Output 3-5 bullet points only. "
                    "Summarise what HAPPENED in past tense — events, decisions, emotional beats. "
                    "Do NOT describe the current state or scene (that is handled elsewhere). "
                    "Do NOT invent any detail not explicitly in the transcript. "
                    "Do NOT quote dialogue. No headers, no preamble."
                ),
            },
            {
                "role": "user",
                "content": f"Characters: {known_chars}\n\nTranscript of past events:\n{transcript}",
            },
        ]
        resp, model_used = self.call_with_failover(
            prompt, stream=False, is_summary=True
        )
        if resp:
            raw = (resp.choices[0].message.content or "").strip()
            # reject if summary is longer than transcript
            if raw and len(raw) < len(transcript) * 0.8:
                if self.debug:
                    self.console.print(f"[dim]summary via {model_used}[/dim]")
                return raw
            elif raw and self.debug:
                self.console.print(
                    "[dim yellow]summary too long, using fallback[/dim yellow]"
                )
        return self._memory_fallback_summary(messages)

    def _memory_fallback_summary(self, messages: list[dict]) -> str:
        bullets = []
        if self.lore and self.lore != "The story is just beginning.":
            bullets.append(f"• Previous lore: {self.lore[:200]}")
        for c in list(self.memory.characters.values())[:5]:
            parts = [c.name]
            if c.age:
                parts.append(f"age {c.age}")
            if c.description:
                parts.append(c.description[:60])
            bullets.append(f"• Character: {', '.join(parts)}")
        for r in self.memory.relationships[:4]:
            bullets.append(f"• {r.from_char} ↔ {r.to_char} ({r.rel_type})")
        if self.memory.locations:
            bullets.append(f"• Locations: {', '.join(self.memory.locations[:5])}")
        for msg in [m["content"] for m in messages if m.get("role") == "assistant"][
            -2:
        ]:
            clean = re.sub(r"\*[^*]+\*", "").strip()[:100]
            if clean:
                bullets.append(f"• Recent: {clean}")
        return "\n".join(bullets) if bullets else "• Story continued."

    def condense_logic(self) -> None:
        self.console.print(
            "\n[dim italic yellow]⚡ consolidating memories…[/dim italic yellow]"
        )
        keep_n = self.KEEP_RECENT_PAIRS * 2
        if len(self.history) <= keep_n + 1:
            return
        to_summarise = self.history[1:-keep_n]
        if not to_summarise:
            self.history = [self.history[0]] + self.history[-keep_n:]
            return
        new_summary = self._summarise(to_summarise)
        if new_summary:
            # combine: first condense replaces, later ones prepend
            if self._condense_count == 0 or self.lore == "The story is just beginning.":
                combined_lore = new_summary
            else:
                combined_lore = f"{self.lore}\n\n{new_summary}"
            # cap lore size to avoid confusing weak models
            if len(combined_lore) > 1200:
                combined_lore = combined_lore[-1200:]
        else:
            combined_lore = (
                self.lore
                if self.lore != "The story is just beginning."
                else "• Story began."
            )
        self._update_system_lore(combined_lore)
        self.history = [self.history[0]] + self.history[-keep_n:]
        self._condense_count += 1
        if self.debug:
            self.console.print(
                f"[dim]condense #{self._condense_count}, {self._convo_msg_count()} msgs remain[/dim]"
            )

    def _check_and_condense(self) -> None:
        if self._should_condense():
            self.condense_logic()

    def _handle_set_command(self, args: str) -> None:
        """
        /set scene <desc>   — set current scene context
        /set mood <desc>    — set character's current mood
        /set name <name>    — rename the character mid-session
        /set player <name>  — rename the player label
        """
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            self.console.print(
                "[yellow]Usage: /set <option> <value>[/yellow]\n"
                "[dim]Options:[/dim]\n"
                "  • [cyan]/set name <name>[/cyan]    - Set character name\n"
                "  • [cyan]/set player <name>[/cyan]  - Set player name\n"
                "  • [cyan]/set scene <desc>[/cyan]   - Set current scene\n"
                "  • [cyan]/set mood <desc>[/cyan]    - Set character mood\n\n"
                "[dim]Example: /set scene in a dark forest[/dim]"
            )
            return
        key, value = parts[0].lower(), parts[1].strip()
        if key == "scene":
            self.scene = value
            self._rebuild_system()
            self.console.print(f"[green]✓ Scene set:[/green] {value}")
        elif key == "mood":
            self.mood = value
            self._rebuild_system()
            self.console.print(f"[green]✓ Mood set:[/green] {value}")
        elif key == "name":
            old = self.persona_name
            self.persona_name = value.capitalize()
            self.console.print(
                f"[green]✓ Character renamed:[/green] {old} → {self.persona_name}"
            )
        elif key == "player":
            self.player_label = value.capitalize()
            self.console.print(f"[green]✓ Player name set:[/green] {self.player_label}")
        else:
            self.console.print(
                f"[yellow]Unknown option: {key}[/yellow]\n"
                "[dim]Valid options: name, player, scene, mood[/dim]"
            )

    def _get_session_path(self, name: str) -> Path:
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        safe = re.sub(r"[^\w-]", "_", name)
        return self.SESSIONS_DIR / f"{safe}.json"

    def save_session(self, name: Optional[str] = None) -> None:
        if not name:
            name = f"{self.persona_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = self._get_session_path(name)
        path.write_text(
            json.dumps(
                {
                    "persona_name": self.persona_name,
                    "persona_desc": self.persona_desc,
                    "player_label": self.player_label,
                    "lore": self.lore,
                    "history": self.history,
                    "memory": self.memory.to_dict(),
                    "condense_count": self._condense_count,
                    "scene": self.scene,
                    "mood": self.mood,
                    "narrate_mode": self.narrate_mode,
                    "saved_at": datetime.datetime.now().isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.console.print(f"[bold green]✓ saved:[/bold green] {path}")

    def load_session(self, name: str) -> bool:
        path = self._get_session_path(name)
        if not path.exists():
            direct = Path(name)
            path = direct if direct.exists() else path
        if not path.exists():
            self.console.print(f"[bold red]session not found:[/bold red] {name}")
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.persona_name = data.get("persona_name", "Character")
            self.persona_desc = data.get("persona_desc", "")
            self.player_label = data.get("player_label", "You")
            self.lore = data.get("lore", "")
            self.history = data.get("history", [])
            self.memory = WorldMemory.from_dict(data.get("memory", {}))
            self._condense_count = data.get("condense_count", 0)
            self.scene = data.get("scene", "")
            self.mood = data.get("mood", "")
            self.narrate_mode = data.get("narrate_mode", False)
            self.console.print(f"[bold green]✓ loaded:[/bold green] {path}")
            return True
        except Exception as e:
            self.console.print(f"[bold red]load failed:[/bold red] {e}")
            return False

    def _list_sessions(self) -> list[dict]:
        """List all saved sessions and return their data."""
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        files = sorted(self.SESSIONS_DIR.glob("*.json"))
        sessions = []
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "name": f.stem,
                        "saved": d.get("saved_at", "?")[:19],
                        "persona": d.get("persona_name", "?"),
                        "scene": d.get("scene", "")[:30],
                        "path": f,
                    }
                )
            except Exception:
                sessions.append(
                    {
                        "name": f.stem,
                        "saved": "?",
                        "persona": "?",
                        "scene": "",
                        "path": f,
                    }
                )
        return sessions

    def _show_sessions_table(self) -> None:
        """Display sessions as a table."""
        sessions = self._list_sessions()
        if not sessions:
            self.console.print("[dim]No saved sessions found.[/dim]")
            return
        t = Table(title="💾 Saved Sessions", show_header=True)
        t.add_column("#", style="dim", width=4)
        t.add_column("Name", style="cyan")
        t.add_column("Character", style="magenta")
        t.add_column("Scene", style="dim")
        t.add_column("Saved", style="dim")
        for i, s in enumerate(sessions, 1):
            t.add_row(
                str(i),
                s["name"],
                s["persona"],
                s["scene"] or "(none)",
                s["saved"][:16],
            )
        self.console.print(t)

    def load_session_interactive(self) -> bool:
        """Load a session interactively with a menu."""
        sessions = self._list_sessions()
        if not sessions:
            self.console.print("[yellow]No saved sessions found.[/yellow]")
            self.console.print(
                "[dim]Start a new character with /new or just type a name.[/dim]"
            )
            return False

        # Show table
        self._show_sessions_table()

        # Ask for selection
        choice = Prompt.ask(
            "\n[bold green]Choose a session[/bold green] [dim](number or name, enter to cancel)[/dim]",
            default="",
        ).strip()

        if not choice:
            self.console.print("[dim]Cancelled.[/dim]")
            return False

        # Try to find matching session
        selected_session = None

        # Check if it's a number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                selected_session = sessions[idx]
        else:
            # Search by name (partial match)
            choice_lower = choice.lower()
            for s in sessions:
                if choice_lower in s["name"].lower():
                    selected_session = s
                    break

        if not selected_session:
            self.console.print(f"[yellow]Session not found: {choice}[/yellow]")
            return False

        # Load the session
        path = selected_session["path"]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.persona_name = data.get("persona_name", "Character")
            self.persona_desc = data.get("persona_desc", "")
            self.player_label = data.get("player_label", "You")
            self.lore = data.get("lore", "")
            self.history = data.get("history", [])
            self.memory = WorldMemory.from_dict(data.get("memory", {}))
            self._condense_count = data.get("condense_count", 0)
            self.scene = data.get("scene", "")
            self.mood = data.get("mood", "")
            self.narrate_mode = data.get("narrate_mode", False)

            self.console.print(
                f"\n[bold green]✓ Loaded:[/bold green] {selected_session['name']}"
            )
            self.console.print(f"   [bold]Character:[/bold] {self.persona_name}")
            if self.scene:
                self.console.print(f"   [bold]Scene:[/bold] {self.scene}")

            # Ask if user wants to edit
            edit = Prompt.ask(
                "\n[bold yellow]Edit character?[/bold yellow] [dim](y/N)[/dim]",
                default="n",
            ).lower()

            if edit == "y":
                # Edit name
                new_name = Prompt.ask(
                    f"[bold]Character name[/bold] [dim](current: {self.persona_name})[/dim]",
                    default=self.persona_name,
                ).strip()
                if new_name:
                    self.persona_name = new_name.capitalize()

                # Edit description
                new_desc = Prompt.ask(
                    "[bold]Character description[/bold] [dim](enter to keep current)[/dim]",
                    default=self.persona_desc,
                ).strip()
                if new_desc:
                    self.persona_desc = new_desc

                # Rebuild system prompt with new info
                self._rebuild_system()
                self.console.print(f"[green]✓ Character updated![/green]")

            return True
        except Exception as e:
            self.console.print(f"[bold red]Load failed:[/bold red] {e}")
            return False

    def _load_session_by_path(self, path: Path) -> bool:
        """Load a session from a path."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.persona_name = data.get("persona_name", "Character")
            self.persona_desc = data.get("persona_desc", "")
            self.player_label = data.get("player_label", "You")
            self.lore = data.get("lore", "")
            self.history = data.get("history", [])
            self.memory = WorldMemory.from_dict(data.get("memory", {}))
            self._condense_count = data.get("condense_count", 0)
            self.scene = data.get("scene", "")
            self.mood = data.get("mood", "")
            self.narrate_mode = data.get("narrate_mode", False)

            self.console.print(f"\n[bold green]✓ Loaded session![/bold green]")
            self.console.print(f"   [bold]Character:[/bold] {self.persona_name}")
            if self.scene:
                self.console.print(f"   [bold]Scene:[/bold] {self.scene}")
            return True
        except Exception as e:
            self.console.print(f"[bold red]Load failed:[/bold red] {e}")
            return False

    def _edit_character_prompts(self) -> None:
        """Prompt user to edit character name and description."""
        # Edit name
        new_name = Prompt.ask(
            f"[bold]Character name[/bold] [dim](current: {self.persona_name})[/dim]",
            default=self.persona_name,
        ).strip()
        if new_name:
            self.persona_name = new_name.capitalize()

        # Edit description
        new_desc = Prompt.ask(
            "[bold]Character description[/bold] [dim](enter to keep current)[/dim]",
            default=self.persona_desc,
        ).strip()
        if new_desc:
            self.persona_desc = new_desc

        # Rebuild system prompt
        self._rebuild_system()
        self.console.print(f"[green]✓ Character updated![/green]")

    def _show_memory(self) -> None:
        self.console.print(
            Panel(
                self.memory.format_world(), title="📚 world memory", border_style="blue"
            )
        )

    def _show_status(self) -> None:
        """Display current session status."""
        model_info = self._last_working_model or "(trying...)"
        lines = [
            f"[bold]Character:[/bold]   {self.persona_name}",
            f"[bold]Player:[/bold]     {self.player_label}",
            f"[bold]Model:[/bold]      {model_info}",
            f"[bold]Scene:[/bold]      {self.scene or '(none set)'}",
            f"[bold]Mood:[/bold]       {self.mood or '(none set)'}",
            f"[bold]Narration:[/bold]  {'enabled' if self.narrate_mode else 'disabled'}",
            f"[bold]Messages:[/bold]   {self._convo_msg_count()}",
            f"[bold]Condenses:[/bold]  {self._condense_count}",
        ]
        self.console.print(
            Panel("\n".join(lines), title="📊 Session Status", border_style="cyan")
        )

    def _show_help(self) -> None:
        """Display help information with command usage."""
        self.console.print(
            Panel(
                "[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]\n"
                "[bold cyan]║                    ChatME COMMANDS                       ║[/bold cyan]\n"
                "[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]\n\n"
                "[bold green]📖 GENERAL[/bold green]\n"
                "  [cyan]/help[/cyan]               Show this help message\n"
                "  [cyan]/status[/cyan]             Show session status (character, model, scene)\n"
                "  [cyan]/memory[/cyan]  [dim]/mem[/dim]      View characters, relationships & locations\n"
                "  [cyan]/lore[/cyan]               View story history/lore\n\n"
                "[bold green]🎭 CHARACTER[/bold green]\n"
                "  [cyan]/set name <name>[/cyan]    Rename your character [dim]e.g. /set name Alice[/dim]\n"
                "  [cyan]/set player <name>[/cyan]  Set player's display name [dim]e.g. /set player Bob[/dim]\n"
                "  [cyan]/set scene <desc>[/cyan]   Set current scene [dim]e.g. /set scene in a dark forest[/dim]\n"
                "  [cyan]/set mood <desc>[/cyan]    Set character mood [dim]e.g. /set mood angry[/dim]\n"
                "  [cyan]/narrate[/cyan]           Toggle narration mode (descriptive responses)\n\n"
                "[bold green]💬 CONVERSATION[/bold green]\n"
                "  [cyan]/retry[/cyan]              Regenerate last AI response\n"
                "  [cyan]/clear[/cyan]              Clear message history (keeps character info)\n\n"
                "[bold green]💾 SESSIONS[/bold green]\n"
                "  [cyan]/load[/cyan]  [dim]/new[/dim]          Load or start new character (interactive)\n"
                "  [cyan]/save [name][/cyan]        Save session [dim]e.g. /save or /save my_adventure[/dim]\n"
                "  [cyan]/sessions[/cyan]           List all saved sessions\n\n"
                "[bold green]🔧 DEBUGGING[/bold green]\n"
                "  [cyan]/debug[/cyan]              Toggle debug mode (shows API errors)\n\n"
                "[bold green]🚪 EXIT[/bold green]\n"
                "  [cyan]exit[/cyan] or [cyan]quit[/cyan]    End session (prompts to save)",
                title="❓ Help",
                border_style="yellow",
            )
        )

    def run(self) -> None:
        self.console.clear()
        self.console.print(
            Panel(
                "[bold white]╔═══════════════════════════════════════════════════════════╗[/bold white]\n"
                "[bold white]║            ★ ChatME ROLEPLAY ENGINE ★                   ║[/bold white]\n"
                "[bold white]║                                                           ║[/bold white]\n"
                "[bold white]║   Type [cyan]/help[/cyan] anytime for commands                        ║[/bold white]\n"
                "[bold white]║   Type [cyan]/status[/cyan] to see your current settings             ║[/bold white]\n"
                "[bold white]║   Type [cyan]/memory[/cyan] to view tracked characters & locations   ║[/bold white]\n"
                "[bold white]╚═══════════════════════════════════════════════════════════╝[/bold white]",
                style="bold blue",
                expand=False,
            )
        )

        while True:
            # Show sessions and ask what to do
            sessions = self._list_sessions()

            # If no sessions, skip the menu and go straight to creating new
            if not sessions:
                self.console.print(
                    "\n[bold]Starting fresh! Creating a new character...[/bold]\n"
                )
            else:
                self.console.print("\n[bold]Saved sessions:[/bold]")
                self._show_sessions_table()
                self.console.print(
                    "\n   [cyan]• Press ENTER[/cyan] to create a new character"
                )
                self.console.print("   [cyan]• Type a number[/cyan] to load a session")
                self.console.print(
                    "   [cyan]• Type a name[/cyan] to search for a session"
                )
                choice = Prompt.ask(
                    "\n[bold green]Your choice?[/bold green]",
                    default="",
                ).strip()

                if choice:
                    loaded = False
                    # Try to load by number or name
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(sessions):
                            s = sessions[idx]
                            if self._load_session_by_path(s["path"]):
                                if (
                                    Prompt.ask(
                                        "\n[bold yellow]Edit character?[/bold yellow] [dim](y/N)[/dim]",
                                        default="n",
                                    ).lower()
                                    == "y"
                                ):
                                    self._edit_character_prompts()
                                loaded = True
                    else:
                        # Try loading by name
                        for s in sessions:
                            if choice.lower() in s["name"].lower():
                                if self._load_session_by_path(s["path"]):
                                    if (
                                        Prompt.ask(
                                            "\n[bold yellow]Edit character?[/bold yellow] [dim](y/N)[/dim]",
                                            default="n",
                                        ).lower()
                                        == "y"
                                    ):
                                        self._edit_character_prompts()
                                    loaded = True
                                    break
                                break
                    if loaded:
                        break
                    self.console.print(
                        f"[yellow]Could not find session: {choice}[/yellow]"
                    )
                    continue
                # User pressed enter - create new
                self.console.print("\n[bold]Creating new character...[/bold]\n")

            # Ask for name
            persona_name_input = Prompt.ask(
                "[bold green]Character name?[/bold green]"
            ).strip()
            if not persona_name_input:
                self.console.print("[yellow]Please enter a character name[/yellow]")
                continue

            self.persona_name = persona_name_input.capitalize()

            # Ask for description
            raw = Prompt.ask(
                "[bold green]Character description?[/bold green]"
                " [dim](personality, appearance, backstory)[/dim]"
            ).strip()
            if not raw:
                self.console.print("[yellow]Please enter a description[/yellow]")
                continue

            self.persona_desc = raw

            # Player name
            player_raw = Prompt.ask(
                "[bold green]Your name?[/bold green] [dim](enter to skip)[/dim]",
                default="",
            ).strip()
            if player_raw:
                self.player_label = player_raw.capitalize()

            # Optional scene
            scene_raw = Prompt.ask(
                "[bold green]Opening scene?[/bold green] [dim](enter to skip)[/dim]",
                default="",
            ).strip()
            if scene_raw:
                self.scene = scene_raw

            self.memory.add_character(
                self.persona_name,
                description=self.persona_desc,
                context="player character",
            )
            self.history.append(
                {
                    "role": "system",
                    "content": self._build_system_content(self.persona_desc, self.lore),
                }
            )
            break

        mode_tag = " [dim]📖 narration enabled[/dim]" if self.narrate_mode else ""
        self.console.print(
            f"\n[bold green]✓ Ready to roleplay![/bold green]\n\n"
            f"   [bold]Character:[/bold] [magenta]{self.persona_name}[/magenta]{mode_tag}\n"
            f"   [bold]Player:[/bold]    [cyan]{self.player_label}[/cyan]\n"
            f"   [bold]Scene:[/bold]     {self.scene or '(none set)'}\n\n"
            f"   [dim]Quick tips:[/dim]\n"
            f"   • Type [cyan]/help[/cyan] for all commands\n"
            f"   • Type [cyan]/memory[/cyan] to see tracked characters\n"
            f"   • Type [cyan]/save[/cyan] to save your progress\n"
            f"   • Type [cyan]exit[/cyan] when done\n"
        )

        while True:
            try:
                user_input = (
                    Prompt.ask(f"[bold cyan]{self.player_label}[/bold cyan]")
                    .strip()
                    .strip("\"'")
                )
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[bold green]goodbye[/bold green]")
                break

            if not user_input:
                continue

            cmd = user_input.lower().strip()

            if cmd in ("exit", "/exit", "quit", "/quit"):
                if (
                    Prompt.ask(
                        "[yellow]save before exit?[/yellow] [dim](y/N)[/dim]",
                        default="n",
                    ).lower()
                    == "y"
                ):
                    self.save_session()
                self.console.print("[bold green]goodbye[/bold green]")
                break

            if cmd == "/help":
                self._show_help()
                continue
            if cmd in ("/memory", "/mem"):
                self._show_memory()
                continue
            if cmd == "/status":
                self._show_status()
                continue

            if cmd == "/lore":
                self.console.print(Panel(self.lore, title="lore", border_style="green"))
                continue

            if cmd == "/narrate":
                self.narrate_mode = not self.narrate_mode
                self._rebuild_system()
                state = "on" if self.narrate_mode else "off"
                self.console.print(f"[dim green]narration mode {state}[/dim green]")
                continue

            if cmd == "/retry":
                # pop last assistant message to re-ask
                if self.history and self.history[-1]["role"] == "assistant":
                    self.history.pop()
                    self.console.print("[dim yellow]regenerating…[/dim yellow]")
                    self.console.print(Rule(style="dim"))
                    self.console.print(
                        f"[bold magenta]{self.persona_name}[/bold magenta]"
                    )
                    content = self._get_reply()
                    if content and len(content.strip()) >= 5:
                        self._last_assistant_content = content
                        self.history.append({"role": "assistant", "content": content})
                    else:
                        self.console.print("[dim red]retry failed[/dim red]")
                else:
                    self.console.print("[dim]nothing to retry[/dim]")
                continue

            if cmd.startswith("/set"):
                self._handle_set_command(user_input[4:])
                continue

            if cmd.startswith("/save"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    self.save_session(parts[1].strip())
                else:
                    self.save_session()
                continue

            if cmd.startswith("/load") or cmd == "/new":
                # Show sessions list and let user choose
                if self.load_session_interactive():
                    break
                continue

            if cmd == "/sessions":
                self._show_sessions_table()
                continue

            if cmd == "/clear":
                if (
                    Prompt.ask(
                        "[yellow]Clear all messages?[/yellow] [dim](y/N)[/dim]",
                        default="n",
                    ).lower()
                    == "y"
                ):
                    self.history = [self.history[0]] if self.history else []
                    self._condense_count = 0
                    self.console.print("[green]✓ History cleared[/green]")
                continue

            if cmd == "/debug":
                self.debug = not self.debug
                self.console.print(
                    f"[green]✓ Debug mode:[/green] {'enabled' if self.debug else 'disabled'}"
                )
                continue

            # normal turn
            self._extract_info_from_message(user_input, is_user=True)
            # append user message first so it's in kept window if condense fires
            self.history.append({"role": "user", "content": user_input})
            self._check_and_condense()
            self.console.print(Rule(style="dim"))
            self.console.print(f"[bold magenta]{self.persona_name}[/bold magenta]")

            content = self._get_reply()

            if not content or len(content.strip()) < 5:
                self.console.print("[dim red]empty response — try again[/dim red]")
                self.history.pop()
                continue

            self._extract_info_from_message(content, is_user=False)
            self._update_system_memory()
            self._last_assistant_content = content
            self.history.append({"role": "assistant", "content": content})
            self._msg_count += 1

            if self._msg_count % 20 == 0:
                self.console.print(
                    "[dim yellow]💾 /save to keep this session[/dim yellow]"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChatME Roleplay Engine")
    parser.add_argument(
        "--model",
        "-m",
        help="Model to use (e.g., cognitivecomputations/dolphin-mistral-24b-venice-edition:free)",
    )
    parser.add_argument(
        "--key",
        "-k",
        help="OpenRouter API key for this session (overrides OPENROUTER_API_KEY env var)",
    )
    args = parser.parse_args()

    RoleplayEngine(model=args.model, api_key=args.key).run()
