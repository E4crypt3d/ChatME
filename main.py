"""
ChatME Roleplay Engine - Enhanced with Memory Features
"""

import os
import re
import sys
import json
from typing import Optional
from openai import OpenAI, APIError, APITimeoutError
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from dotenv import load_dotenv

load_dotenv()

LORE_MARKER = "<<<LORE_BLOCK>>>"
CHARACTERS_MARKER = "<<<CHARACTERS>>>"
RELATIONSHIPS_MARKER = "<<<RELATIONSHIPS>>>"


class Character:
    """a character in the roleplay."""

    def __init__(self, name: str):
        self.name = name
        self.age: Optional[str] = None
        self.description: str = ""
        self.personality: list[str] = []
        self.first_appearance: str = ""
        self.mentions: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "description": self.description,
            "personality": self.personality,
            "first_appearance": self.first_appearance,
            "mentions": self.mentions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Character":
        char = cls(data.get("name", ""))
        char.age = data.get("age")
        char.description = data.get("description", "")
        char.personality = data.get("personality", [])
        char.first_appearance = data.get("first_appearance", "")
        char.mentions = data.get("mentions", 0)
        return char

    def format_info(self) -> str:
        info = [f"**{self.name}**"]
        if self.age:
            info.append(f"Age: {self.age}")
        if self.description:
            info.append(f"Description: {self.description}")
        if self.personality:
            info.append(f"Personality: {', '.join(self.personality)}")
        info.append(f"Mentions: {self.mentions}")
        return "\n".join(info)


class Relationship:
    """a relationship between characters."""

    def __init__(
        self, from_char: str, to_char: str, relation_type: str, description: str = ""
    ):
        self.from_char = from_char
        self.to_char = to_char
        self.relation_type = relation_type
        self.description = description
        self.first_appearance: str = ""

    def to_dict(self) -> dict:
        return {
            "from": self.from_char,
            "to": self.to_char,
            "type": self.relation_type,
            "description": self.description,
            "first_appearance": self.first_appearance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        rel = cls(
            data.get("from", ""),
            data.get("to", ""),
            data.get("type", "unknown"),
            data.get("description", ""),
        )
        rel.first_appearance = data.get("first_appearance", "")
        return rel

    def format_info(self) -> str:
        desc = f" - {self.description}" if self.description else ""
        return f"**{self.from_char}** ↔ **{self.to_char}**: {self.relation_type}{desc}"


class WorldMemory:
    """manages characters, relationships, and world info."""

    def __init__(self):
        self.characters: dict[str, Character] = {}
        self.relationships: list[Relationship] = []
        self.locations: list[str] = []
        self.events: list[str] = []
        self.custom_notes: list[str] = []

    def add_character(
        self,
        name: str,
        age: Optional[str] = None,
        description: str = "",
        context: str = "",
    ) -> Character:
        name = name.strip().title()
        if name not in self.characters:
            self.characters[name] = Character(name)
            self.characters[name].age = age
            self.characters[name].description = description
            self.characters[name].first_appearance = context[:100] if context else ""
        self.characters[name].mentions += 1
        return self.characters[name]

    def add_relationship(
        self,
        from_char: str,
        to_char: str,
        relation_type: str,
        description: str = "",
        context: str = "",
    ) -> Relationship:
        from_char = from_char.strip().title()
        to_char = to_char.strip().title()

        # check if relationship already exists
        for rel in self.relationships:
            if rel.from_char == from_char and rel.to_char == to_char:
                rel.relation_type = relation_type
                rel.description = description
                return rel

        rel = Relationship(from_char, to_char, relation_type, description)
        rel.first_appearance = context[:100] if context else ""
        self.relationships.append(rel)
        return rel

    def add_location(self, location: str) -> None:
        location = location.strip()
        if location and location not in self.locations:
            self.locations.append(location)

    def add_event(self, event: str) -> None:
        if event and event not in self.events:
            self.events.append(event)

    def add_note(self, note: str) -> None:
        if note:
            self.custom_notes.append(note)

    def get_character(self, name: str) -> Optional[Character]:
        return self.characters.get(name.strip().title())

    def get_relationships_for(self, character: str) -> list[Relationship]:
        character = character.strip().title()
        return [
            r
            for r in self.relationships
            if r.from_char == character or r.to_char == character
        ]

    def format_characters(self) -> str:
        if not self.characters:
            return "No characters recorded yet."
        lines = ["**Characters:**"]
        for char in sorted(self.characters.values(), key=lambda x: -x.mentions):
            lines.append(f"- {char.format_info()}")
        return "\n".join(lines)

    def format_relationships(self) -> str:
        if not self.relationships:
            return "No relationships recorded yet."
        lines = ["**Relationships:**"]
        for rel in self.relationships:
            lines.append(f"- {rel.format_info()}")
        return "\n".join(lines)

    def format_world(self) -> str:
        parts = []
        if self.characters:
            parts.append(self.format_characters())
        if self.relationships:
            parts.append(self.format_relationships())
        if self.locations:
            parts.append(f"**Locations:** {', '.join(self.locations)}")
        if self.events:
            parts.append(f"**Events:** {', '.join(self.events[-5:])}")  # Last 5 events
        return "\n\n".join(parts) if parts else "No world info recorded yet."

    def to_dict(self) -> dict:
        return {
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "locations": self.locations,
            "events": self.events,
            "custom_notes": self.custom_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldMemory":
        mem = cls()
        for name, char_data in data.get("characters", {}).items():
            mem.characters[name] = Character.from_dict(char_data)
        mem.relationships = [
            Relationship.from_dict(r) for r in data.get("relationships", [])
        ]
        mem.locations = data.get("locations", [])
        mem.events = data.get("events", [])
        mem.custom_notes = data.get("custom_notes", [])
        return mem


class RoleplayEngine:
    """roleplay engine with memory management."""

    # settings
    CONDENSE_THRESHOLD_MSGS = 20
    CONDENSE_THRESHOLD_TOKENS = 4000
    KEEP_RECENT_TURNS = 8
    REPLY_MAX_TOKENS = 400
    SUMMARY_MAX_TOKENS = 200
    REQUEST_TIMEOUT = 120.0
    SUMMARY_TIMEOUT = 60.0
    MIN_RESPONSE_LENGTH = 10
    MAX_HISTORY_HARD_CAP = 50

    def __init__(self):
        import sys

        if sys.platform == "win32":
            import io

            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )

        self.console = Console(force_terminal=True)
        self._init_client()
        self._init_state()

    def _init_client(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.console.print("[bold red]Error: OPENROUTER_API_KEY not set[/bold red]")
            sys.exit(1)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=self.REQUEST_TIMEOUT,
        )

        self.models = [
            "arcee-ai/trinity-large-preview:free",
            "stepfun/step-3.5-flash:free",
            "z-ai/glm-4.5-air:free",
            "arcee-ai/trinity-mini:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        ]
        self.summary_model = "arcee-ai/trinity-mini:free"
        self._last_working_model: Optional[str] = None

    def _init_state(self) -> None:
        self.debug = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
        self.history: list[dict] = []
        self.persona_name = "Character"
        self.persona_desc = ""
        self.lore = "The story is just beginning."
        self._failed_models: set[str] = set()
        self._msg_count = 0

        # Memory features
        self.memory = WorldMemory()

    def _build_system_content(self, persona_desc: str, lore: str) -> str:
        """build system prompt."""
        world_info = self.memory.format_world()

        return (
            f"ROLEPLAY: {persona_desc}\n\n"
            "RULES:\n"
            "- Stay in character always\n"
            "- Never break character or mention being AI\n"
            "- Use *italics* for actions/thoughts, **bold** for emphasis\n"
            "- Keep responses concise and natural (2-4 sentences)\n\n"
            f"CHARACTER DATABASE:\n{CHARACTERS_MARKER}\n{world_info}\n{CHARACTERS_MARKER}\n\n"
            f"STORY SO FAR:\n{LORE_MARKER}\n{lore}\n{LORE_MARKER}"
        )

    def _extract_info_from_message(self, text: str, is_user: bool) -> None:
        """extract character/relationship info from messages."""
        text_lower = text.lower()

        # Extract character mentions
        name_patterns = [
            r"(?:i am|i'm|my name is)\s+(\w+)",
            r"(\w+)\s+is\s+(\d+)\s*(?:years?\s*old)?",
            r"(\w+)\s+(?:is|was)\s+(?:a|an)\s+(\w+)",
            r"(?:meet|introduce[d]?)\s+(\w+)",
            r"this is\s+(\w+)",
        ]

        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if is_user:
                    # skip self-reference
                    if match.group(1).lower() in ["i", "me", "myself"]:
                        continue
                    # new character
                    char_name = match.group(1)
                    if char_name.lower() not in [
                        "a",
                        "an",
                        "the",
                        "my",
                        "your",
                        "his",
                        "her",
                    ]:
                        context = text[: match.start()]
                        self.memory.add_character(
                            char_name,
                            description=f"Mentioned: {text[match.start():match.end()]}",
                            context=context,
                        )

        # Extract relationship hints
        rel_patterns = [
            (r"(\w+)\s+is\s+(?:my|his|her|their)\s+(\w+)", 2),  # He is my brother
            (r"(?:my|his|her|their)\s+(\w+)\s+is\s+(\w+)", 1),  # My friend is John
            (r"(\w+)\s+and\s+(\w+)\s+are\s+(\w+)", 3),  # John and Mary are friends
            (r"(?:i|we)\s+(?:meet|know)\s+(\w+)", 1),  # I met John
        ]

        for pattern, name_group in rel_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if len(match.groups()) >= name_group:
                    char1 = match.group(1).title()
                    if name_group >= 2:
                        char2 = match.group(2).title() if name_group > 1 else ""
                        rel_type = (
                            match.group(3).title() if name_group > 2 else "related"
                        )
                        if char1 and char2 and char1 != char2:
                            self.memory.add_relationship(
                                char1, char2, rel_type, text[:50]
                            )

        # Extract locations
        location_words = ["in", "at", "to", "from"]
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in location_words and i + 1 < len(words):
                loc = words[i + 1].strip(".,!?")
                if len(loc) > 2 and loc[0].isupper():
                    self.memory.add_location(loc)

    def _update_system_lore(self, new_lore: str) -> None:
        """update lore in system message."""
        if not self.history or self.history[0]["role"] != "system":
            return

        content = self.history[0]["content"]
        self.lore = new_lore

        # replace lore between markers
        pattern = re.compile(
            rf"{re.escape(LORE_MARKER)}.*?{re.escape(LORE_MARKER)}", re.DOTALL
        )
        new_content = pattern.sub(f"{LORE_MARKER}\n{new_lore}\n{LORE_MARKER}", content)

        if LORE_MARKER not in new_content:
            new_content += f"\n\n{LORE_MARKER}\n{new_lore}\n{LORE_MARKER}"

        self.history[0]["content"] = new_content

        # update character/relationship info in system prompt
        self._update_system_memory()

    def _update_system_memory(self) -> None:
        """update character/relationship info in system message."""
        if not self.history or self.history[0]["role"] != "system":
            return

        world_info = self.memory.format_world()
        content = self.history[0]["content"]

        # replace between CHARACTERS markers
        pattern = re.compile(
            rf"{re.escape(CHARACTERS_MARKER)}.*?{re.escape(CHARACTERS_MARKER)}",
            re.DOTALL,
        )
        new_content = pattern.sub(
            f"{CHARACTERS_MARKER}\n{world_info}\n{CHARACTERS_MARKER}", content
        )

        if CHARACTERS_MARKER not in new_content:
            new_content += f"\n\n{CHARACTERS_MARKER}\n{world_info}\n{CHARACTERS_MARKER}"

        self.history[0]["content"] = new_content

    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        """filter messages efficiently."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if isinstance(m, dict)
            and "role" in m
            and "content" in m
            and m["content"]
            and isinstance(m["content"], str)
            and m["content"].strip()
        ]

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """rough token estimation."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4 + 1
        return total

    def call_with_failover(
        self, messages: list[dict], stream: bool = True, is_summary: bool = False
    ) -> tuple[Optional[any], Optional[str]]:
        """call API with smart failover."""
        clean = self._clean(messages)
        if not clean:
            return None, None

        models_to_try = []

        if self._last_working_model and self._last_working_model in self.models:
            models_to_try.append(self._last_working_model)

        models_to_try.extend(m for m in self.models if m not in models_to_try)

        if is_summary:
            models_to_try = [self.summary_model]

        self._failed_models.clear()
        last_error = None

        for model in models_to_try:
            try:
                response = self.client.chat.completions.create(
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
                self._last_working_model = model
                return response, model

            except APITimeoutError as e:
                self._failed_models.add(model)
                last_error = f"timeout ({model.split('/')[-1]})"
                if self.debug:
                    self.console.print(f"[dim red]Timeout: {model}[/dim red]")

            except APIError as e:
                self._failed_models.add(model)
                code = getattr(e, "code", "unknown")
                last_error = f"API error {code} ({model.split('/')[-1]})"
                if self.debug:
                    self.console.print(
                        f"[dim red]API Error ({code}): {model}[/dim red]"
                    )

            except Exception as e:
                self._failed_models.add(model)
                last_error = str(e)[:50]
                if self.debug:
                    self.console.print(f"[dim red]Error: {model} - {e}[/dim red]")

        if self._failed_models:
            failed_names = ", ".join(m.split("/")[-1] for m in self._failed_models)
            self.console.print(
                f"[bold red]All models failed: {failed_names}[/bold red]"
            )
            if last_error:
                self.console.print(f"[dim red]Last error: {last_error}[/dim red]")

        return None, None

    def _summarise(self, messages: list[dict]) -> Optional[str]:
        """generate summary with memory extraction."""
        if not messages:
            return None

        lines = []
        for msg in messages:
            role = "Assistant" if msg["role"] == "assistant" else "User"
            lines.append(f"[{role}]: {msg.get('content', '')}")

        transcript = "\n".join(lines)

        if len(transcript) > 8000:
            transcript = transcript[-8000:] + "\n[...truncated...]"

        # include memory info in summary request
        memory_context = ""
        if self.memory.characters:
            chars = ", ".join(self.memory.characters.keys())
            memory_context = f"\nKnown characters: {chars}"

        summarise_prompt = [
            {
                "role": "system",
                "content": "Summarize roleplay sessions. Include: setting, key events, character info (names, ages, relationships), emotional tone. Bullet points. Max 5 items.",
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this transcript:{memory_context}\n\n{transcript}"
                ),
            },
        ]

        try:
            result = self.call_with_failover(
                summarise_prompt, stream=False, is_summary=True
            )
            if result[0]:
                content = result[0].choices[0].message.content
                return (content or "").strip() if content else None
        except Exception as e:
            if self.debug:
                self.console.print(f"[dim red]Summary failed: {e}[/dim red]")

        return None

    def _consume_stream(self, response_obj) -> str:
        """consume streaming response with live display."""
        content_parts = []
        content = ""

        try:
            with Live(
                Panel(Text("▌", style="magenta"), border_style="magenta"),
                auto_refresh=True,
                console=self.console,
                transient=False,
                refresh_per_second=10,
            ) as live:
                for chunk in response_obj:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    if not delta:
                        continue

                    token = delta.content or ""
                    if token:
                        content_parts.append(token)
                        content = "".join(content_parts)

                        if len(content_parts) % 3 == 0:
                            live.update(
                                Panel(Markdown(content), border_style="magenta")
                            )

                if content:
                    live.update(Panel(Markdown(content), border_style="magenta"))

        except Exception as e:
            if self.debug:
                self.console.print(f"[dim red]Stream error: {e}[/dim red]")

        return content

    def _stream_response(self, response_obj, model_used: str) -> str:
        """handle streaming with intelligent retry."""
        content = self._consume_stream(response_obj)

        if len(content.strip()) < self.MIN_RESPONSE_LENGTH:
            self.console.print(
                "[dim yellow]⚠ Response too short, retrying…[/dim yellow]"
            )

            for model in self.models:
                try:
                    retry_resp, _ = self.call_with_failover(
                        self._clean(self.history), stream=True
                    )
                    if retry_resp:
                        content = self._consume_stream(retry_resp)
                        if len(content.strip()) >= self.MIN_RESPONSE_LENGTH:
                            if self.debug:
                                self.console.print(
                                    f"[dim green]Retry succeeded[/dim green]"
                                )
                            return content
                except Exception:
                    continue

            self.console.print("[dim red]All retries failed[/dim red]")

        return content

    def _should_condense(self) -> bool:
        """check if history should be condensed."""
        msg_count = len(self.history)
        if msg_count > self.MAX_HISTORY_HARD_CAP:
            return True
        if msg_count > self.CONDENSE_THRESHOLD_MSGS:
            return True
        if self._estimate_tokens(self.history) > self.CONDENSE_THRESHOLD_TOKENS:
            return True
        return False

    def condense_logic(self) -> None:
        """condense old history into lore."""
        self.console.print(
            "\n[dim italic yellow]⚡ Consolidating memories…[/dim italic yellow]"
        )

        total_msgs = len(self.history)
        if total_msgs <= self.KEEP_RECENT_TURNS + 1:
            return

        to_summarise = self.history[1 : -self.KEEP_RECENT_TURNS]

        if not to_summarise:
            self.history = [self.history[0]] + self.history[-self.KEEP_RECENT_TURNS :]
            return

        new_lore = self._summarise(to_summarise)

        if new_lore:
            self._update_system_lore(new_lore)
            if self.debug:
                self.console.print(f"[dim yellow]Lore updated[/dim yellow]")
        else:
            self._update_system_lore(f"{self.lore}\n- Continued conversation...")

        self.history = [self.history[0]] + self.history[-self.KEEP_RECENT_TURNS :]

    def _check_and_condense(self) -> None:
        """trigger condensation if needed."""
        if self._should_condense():
            self.condense_logic()

    def _extract_persona_name(self, raw: str) -> str:
        """extract character name from input."""
        skip_words = {
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
            "character",
            "who",
            "me",
            "roleplay",
            "roleplaying",
            "rp",
            "being",
            "act",
            "acting",
            "like",
            "someone",
            "person",
            "name",
        }

        words = raw.split()
        candidates = []

        for i, word in enumerate(words):
            clean = word.strip(".,!?\"'").lower()
            if clean in skip_words or len(clean) <= 2 or not clean.isalpha():
                continue

            original = word.strip(".,!?\"'")
            score = len(clean)

            if original[0].isupper():
                score += 5

            if i > 0:
                prev = words[i - 1].lower().strip(".,!?\"'")
                if prev in ("named", "called", "is", "name"):
                    score += 10

            candidates.append((score, original))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1].capitalize()

        return "Character"

    def _trim_user_self_reference(self, raw: str) -> str:
        """remove self-reference patterns from persona."""
        lower = raw.lower()

        patterns = [
            r",?\s*\bi\s+am\s+",
            r",?\s*\bi'm\s+",
            r",?\s*\bim\s+",
            r",?\s*\bmy\s+(?:name\s+is\s+|character\s+|name['']?s\s+)",
            r",?\s*\bi\s+play\s+(?:as\s+)?",
            r",?\s*\bmy\s+character\s+is\s+",
            r",?\s*\bi\s+will\s+be\s+",
            r",?\s*\bact\s+as\s+",
            r",?\s*\bacting\s+as\s+",
            r",?\s*\broleplay\s+(?:as\s+)?",
            r",?\s*\brp\s+(?:as\s+)?",
        ]

        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                result = raw[: match.start()].strip()
                if result:
                    return result

        return raw.strip()

    def _show_memory(self) -> None:
        """display character and relationship info."""
        self.console.print(Panel(self.memory.format_world(), title="📚 World Memory"))

    def run(self) -> None:
        self.console.clear()
        self.console.print(
            Panel(
                "[bold white]* ChatME ROLEPLAY ENGINE *[/bold white]\n[dim]Enhanced with Memory Features[/dim]",
                style="bold blue",
                expand=False,
            )
        )

        raw_persona = Prompt.ask(
            "[bold green]Who am I roleplaying as?[/bold green] [dim](character name + description)[/dim]"
        )

        self.persona_name = self._extract_persona_name(raw_persona)
        self.persona_desc = self._trim_user_self_reference(raw_persona)

        # add player character to memory
        self.memory.add_character(
            self.persona_name, description=self.persona_desc, context="Player character"
        )

        self.history.append(
            {
                "role": "system",
                "content": self._build_system_content(self.persona_desc, self.lore),
            }
        )

        self.console.print(
            f"\n[bold blue]Entering roleplay as[/bold blue] "
            f"[bold magenta]{self.persona_name}[/bold magenta]. "
            f"Type [bold red]exit[/bold red] to quit, [bold]/memory[/bold] to view characters.\n"
        )

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
                # strip quotes from piped input
                user_input = user_input.strip("\"'")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[bold green]Goodbye![/bold green]")
                break

            if not user_input:
                continue

            cmd = user_input.strip().lower()

            # Exit commands
            if cmd in ("exit", "/exit", "quit", "/quit"):
                self.console.print("[bold green]Goodbye![/bold green]")
                break

            # /memory command - show tracked characters
            if cmd == "/memory" or user_input.strip().startswith("/memory"):
                self._show_memory()
                continue

            # Extract info from user message
            self._extract_info_from_message(user_input, is_user=True)

            self._check_and_condense()

            self.history.append({"role": "user", "content": user_input})

            self.console.print(Rule(style="dim"))
            self.console.print(f"[bold magenta]{self.persona_name}[/bold magenta]")

            response_obj, model_used = self.call_with_failover(self.history)

            if response_obj is None:
                self.console.print("[bold red]All models failed.[/bold red]")
                self.history.pop()
                continue

            content = self._stream_response(response_obj, model_used)

            if not content or len(content.strip()) < 5:
                self.console.print(
                    "[dim red]Empty response — please try again.[/dim red]"
                )
                self.history.pop()
                continue

            # Extract info from assistant response
            self._extract_info_from_message(content, is_user=False)

            # Update memory in system prompt
            self._update_system_memory()

            self.history.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    RoleplayEngine().run()
