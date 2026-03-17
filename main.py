"""ChatME Roleplay Engine"""

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

load_dotenv()

LORE_MARKER = "<<<LORE_BLOCK>>>"
CHARACTERS_MARKER = "<<<CHARACTERS>>>"

_NAME_STOPWORDS = {
    "a",
    "an",
    "the",
    "my",
    "your",
    "his",
    "her",
    "their",
    "our",
    "i",
    "me",
    "we",
    "you",
    "he",
    "she",
    "they",
    "it",
    "this",
    "that",
    "in",
    "at",
    "to",
    "from",
    "with",
    "for",
    "of",
    "and",
    "or",
    "but",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "had",
    "do",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "here",
    "there",
    "when",
    "where",
    "what",
    "who",
    "how",
    "why",
    "old",
    "young",
    "big",
    "small",
    "new",
    "good",
    "bad",
    "just",
    "then",
    "very",
    "well",
    "also",
    "back",
    "even",
    "still",
    "way",
    "look",
    "come",
    "want",
    "know",
    "think",
    "see",
    "get",
    "make",
    "said",
    "yes",
    "no",
    "not",
    "now",
    "up",
    "out",
    "so",
    "if",
    "than",
    "okay",
    "ok",
    "hey",
    "oh",
    "ah",
    "hmm",
    "uh",
    "um",
}

_RELATION_WORDS = {
    "friend",
    "brother",
    "sister",
    "mother",
    "father",
    "son",
    "daughter",
    "husband",
    "wife",
    "partner",
    "ally",
    "enemy",
    "rival",
    "mentor",
    "student",
    "teacher",
    "boss",
    "colleague",
    "neighbour",
    "neighbor",
    "cousin",
    "uncle",
    "aunt",
    "grandma",
    "grandpa",
    "parent",
    "child",
    "king",
    "queen",
    "lord",
    "lady",
    "knight",
    "guard",
    "servant",
    "companion",
    "lover",
    "master",
    "apprentice",
    "captain",
    "general",
    "girlfriend",
    "boyfriend",
    "fiancee",
    "fiance",
    "spouse",
}

_LOCATION_STOPWORDS = _NAME_STOPWORDS | {
    "not",
    "but",
    "so",
    "if",
    "then",
    "than",
    "by",
    "on",
    "up",
    "out",
}

_FALSE_NAME_WORDS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "north",
    "south",
    "east",
    "west",
    "true",
    "false",
    "none",
    "null",
    "hollywood",
    "okay",
    "sorry",
    "thanks",
    "please",
    "indie",
}


def _is_valid_name(word: str) -> bool:
    w = word.lower()
    return (
        len(word) >= 3
        and word[0].isupper()
        and word.isalpha()
        and w not in _NAME_STOPWORDS
        and w not in _RELATION_WORDS
        and w not in _FALSE_NAME_WORDS
    )


class Character:
    def __init__(self, name: str):
        self.name = name
        self.age: Optional[str] = None
        self.description: str = ""
        self.personality: list[str] = []
        self.first_appearance = ""
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
        c = cls(data.get("name", ""))
        c.age = data.get("age")
        c.description = data.get("description", "")
        c.personality = data.get("personality", [])
        c.first_appearance = data.get("first_appearance", "")
        c.mentions = data.get("mentions", 0)
        return c

    def format_info(self) -> str:
        parts = [f"**{self.name}**"]
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.description:
            parts.append(f"Desc: {self.description}")
        if self.personality:
            parts.append(f"Traits: {', '.join(self.personality)}")
        parts.append(f"Mentions: {self.mentions}")
        return " | ".join(parts)


class Relationship:
    def __init__(
        self, from_char: str, to_char: str, rel_type: str, description: str = ""
    ):
        self.from_char = from_char
        self.to_char = to_char
        self.rel_type = rel_type
        self.description = description
        self.first_appearance = ""

    def to_dict(self) -> dict:
        return {
            "from": self.from_char,
            "to": self.to_char,
            "type": self.rel_type,
            "description": self.description,
            "first_appearance": self.first_appearance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        r = cls(
            data.get("from", ""),
            data.get("to", ""),
            data.get("type", "related"),
            data.get("description", ""),
        )
        r.first_appearance = data.get("first_appearance", "")
        return r

    def format_info(self) -> str:
        desc = f" — {self.description}" if self.description else ""
        return f"**{self.from_char}** ↔ **{self.to_char}**: {self.rel_type}{desc}"


class WorldMemory:
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
    ) -> Optional[Character]:
        key = name.strip().title()
        if not _is_valid_name(key):
            return None
        if key not in self.characters:
            c = Character(key)
            c.age = age
            c.description = description
            c.first_appearance = context[:100]
            c.mentions = 1
            self.characters[key] = c
        else:
            self.characters[key].mentions += 1
            if description and not self.characters[key].description:
                self.characters[key].description = description
            if age and not self.characters[key].age:
                self.characters[key].age = age
        return self.characters[key]

    def add_relationship(
        self,
        from_char: str,
        to_char: str,
        rel_type: str,
        description: str = "",
        context: str = "",
    ) -> Optional[Relationship]:
        fc = from_char.strip().title()
        tc = to_char.strip().title()
        if not fc or not tc or fc == tc:
            return None
        if not _is_valid_name(fc) or not _is_valid_name(tc):
            return None
        for rel in self.relationships:
            if (rel.from_char == fc and rel.to_char == tc) or (
                rel.from_char == tc and rel.to_char == fc
            ):
                rel.rel_type = rel_type
                if description:
                    rel.description = description
                return rel
        r = Relationship(fc, tc, rel_type, description)
        r.first_appearance = context[:100]
        self.relationships.append(r)
        return r

    def add_location(self, location: str) -> None:
        loc = location.strip().rstrip(".,!?;:")
        if (
            loc
            and len(loc) > 2
            and loc.lower() not in _LOCATION_STOPWORDS
            and loc not in self.locations
        ):
            self.locations.append(loc)

    def add_event(self, event: str) -> None:
        if event and event not in self.events:
            self.events.append(event)

    def get_character(self, name: str) -> Optional[Character]:
        return self.characters.get(name.strip().title())

    def format_world(self) -> str:
        parts = []
        if self.characters:
            lines = ["**Characters:**"]
            for c in sorted(self.characters.values(), key=lambda x: -x.mentions):
                lines.append(f"  • {c.format_info()}")
            parts.append("\n".join(lines))
        if self.relationships:
            lines = ["**Relationships:**"]
            for r in self.relationships:
                lines.append(f"  • {r.format_info()}")
            parts.append("\n".join(lines))
        if self.locations:
            parts.append(f"**Locations:** {', '.join(self.locations)}")
        if self.events:
            parts.append(f"**Events:** {', '.join(self.events[-5:])}")
        if self.custom_notes:
            parts.append(f"**Notes:** {', '.join(self.custom_notes[-3:])}")
        return "\n\n".join(parts) if parts else "No world info yet."

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
        m = cls()
        for name, cd in data.get("characters", {}).items():
            m.characters[name] = Character.from_dict(cd)
        m.relationships = [
            Relationship.from_dict(r) for r in data.get("relationships", [])
        ]
        m.locations = data.get("locations", [])
        m.events = data.get("events", [])
        m.custom_notes = data.get("custom_notes", [])
        return m


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

    def __init__(self):
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
            self.console.print("[bold red]OPENROUTER_API_KEY not set[/bold red]")
            sys.exit(1)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=self.REQUEST_TIMEOUT,
        )
        self.models: list[str] = [
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "arcee-ai/trinity-large-preview:free",
            "stepfun/step-3.5-flash:free",
            "z-ai/glm-4.5-air:free",
            "arcee-ai/trinity-mini:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "liquid/lfm-2.5-1.2b-instruct:free",
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
                if _is_valid_name(m.group(1)):
                    self.memory.add_character(m.group(1), context=text[:80])

        # age pattern
        for m in re.finditer(
            r"\b([A-Z][a-z]{1,})\s+is\s+(\d{1,3})\s*(?:years?\s*old)?", text
        ):
            name, age = m.group(1), m.group(2)
            if _is_valid_name(name):
                c = self.memory.add_character(name, age=age, context=text[:80])
                if c and not c.age:
                    c.age = age

        # "[Name] is my/his/her [relation]"
        for m in re.finditer(
            r"\b([A-Z][a-z]{1,})\s+is\s+(?:my|his|her|their|our)\s+(\w+)", text
        ):
            name, rel = m.group(1), m.group(2).lower()
            if _is_valid_name(name) and rel in _RELATION_WORDS:
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        # "my/his/her [relation] [Name]"
        for m in re.finditer(
            r"(?:my|his|her|their|our)\s+(\w+)\s+([A-Z][a-z]{1,})", text
        ):
            rel, name = m.group(1).lower(), m.group(2)
            if rel in _RELATION_WORDS and _is_valid_name(name):
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        # "[Name] and [Name] are [word]"
        for m in re.finditer(
            r"\b([A-Z][a-z]{1,})\s+and\s+([A-Z][a-z]{1,})\s+are\s+(\w+)", text
        ):
            n1, n2, rt = m.group(1), m.group(2), m.group(3).lower()
            if _is_valid_name(n1) and _is_valid_name(n2):
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
                    and cand.lower() not in _LOCATION_STOPWORDS
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
            clean = re.sub(r"\*[^*]+\*", "", msg).strip()[:100]
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

    def _extract_persona_name(self, raw: str) -> str:
        quoted = re.search(r'["\']([A-Za-z][a-z]{1,})["\']', raw)
        if quoted and _is_valid_name(quoted.group(1)):
            return quoted.group(1).capitalize()

        explicit = re.search(
            r"(?:named?|called|as)\s+([A-Z][a-z]{1,}|[a-z]{3,})", raw, re.IGNORECASE
        )
        if explicit:
            cand = explicit.group(1).capitalize()
            if _is_valid_name(cand):
                return cand

        skip = {
            "you",
            "i",
            "im",
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
            "or",
        }
        words = raw.split()
        candidates: list[tuple[int, str]] = []
        for i, word in enumerate(words):
            original = word.strip(".,!?\"'")
            if original.lower() in skip or not _is_valid_name(original):
                continue
            score = len(original)
            if original[0].isupper():
                score += 5
            if i > 0 and words[i - 1].lower().strip(".,!?\"'") in (
                "named",
                "called",
                "is",
                "name",
                "as",
            ):
                score += 10
            candidates.append((score, original))

        return max(candidates)[1].capitalize() if candidates else "Character"

    def _trim_user_self_reference(self, raw: str) -> str:
        lower = raw.lower()
        for pat in [
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
        ]:
            m = re.search(pat, lower)
            if m:
                result = raw[: m.start()].strip()
                if result:
                    return result
        return raw.strip()

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
                "[yellow]usage: /set scene|mood|name|player <value>[/yellow]"
            )
            return
        key, value = parts[0].lower(), parts[1].strip()
        if key == "scene":
            self.scene = value
            self._rebuild_system()
            self.console.print(f"[dim green]scene set: {value}[/dim green]")
        elif key == "mood":
            self.mood = value
            self._rebuild_system()
            self.console.print(f"[dim green]mood set: {value}[/dim green]")
        elif key == "name":
            old = self.persona_name
            self.persona_name = value.capitalize()
            self.console.print(
                f"[dim green]character renamed: {old} → {self.persona_name}[/dim green]"
            )
        elif key == "player":
            self.player_label = value.capitalize()
            self.console.print(
                f"[dim green]player label set: {self.player_label}[/dim green]"
            )
        else:
            self.console.print(f"[yellow]unknown setting: {key}[/yellow]")

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

    def _list_sessions(self) -> None:
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        files = sorted(self.SESSIONS_DIR.glob("*.json"))
        if not files:
            self.console.print("[dim]no saved sessions[/dim]")
            return
        t = Table(title="saved sessions", show_header=True)
        t.add_column("name", style="cyan")
        t.add_column("saved", style="dim")
        t.add_column("persona")
        t.add_column("scene", style="dim")
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                t.add_row(
                    f.stem,
                    d.get("saved_at", "?")[:19],
                    d.get("persona_name", "?"),
                    d.get("scene", "")[:30],
                )
            except Exception:
                t.add_row(f.stem, "?", "?", "")
        self.console.print(t)

    def _show_memory(self) -> None:
        self.console.print(
            Panel(
                self.memory.format_world(), title="📚 world memory", border_style="blue"
            )
        )

    def _show_status(self) -> None:
        lines = [
            f"[bold]persona:[/bold] {self.persona_name}",
            f"[bold]player:[/bold]  {self.player_label}",
            f"[bold]scene:[/bold]   {self.scene or '(none)'}",
            f"[bold]mood:[/bold]    {self.mood or '(none)'}",
            f"[bold]narrate:[/bold] {'on' if self.narrate_mode else 'off'}",
            f"[bold]condense:[/bold] #{self._condense_count}",
            f"[bold]messages:[/bold] {self._convo_msg_count()}",
        ]
        self.console.print(Panel("\n".join(lines), title="status", border_style="cyan"))

    def _show_help(self) -> None:
        self.console.print(
            Panel(
                "[bold]commands:[/bold]\n\n"
                "  [cyan]/memory[/cyan]              characters, relationships, locations\n"
                "  [cyan]/status[/cyan]              show current scene, mood, settings\n"
                "  [cyan]/set scene <desc>[/cyan]    set current scene context\n"
                "  [cyan]/set mood <desc>[/cyan]     set character mood\n"
                "  [cyan]/set name <name>[/cyan]     rename character mid-session\n"
                "  [cyan]/set player <name>[/cyan]   rename player label\n"
                "  [cyan]/narrate[/cyan]             toggle narration mode\n"
                "  [cyan]/retry[/cyan]               regenerate last response\n"
                "  [cyan]/lore[/cyan]                show story lore\n"
                "  [cyan]/save [name][/cyan]         save session\n"
                "  [cyan]/load <name>[/cyan]         load session\n"
                "  [cyan]/sessions[/cyan]            list saved sessions\n"
                "  [cyan]/clear[/cyan]               reset history (keeps persona)\n"
                "  [cyan]/debug[/cyan]               toggle debug output\n"
                "  [cyan]/help[/cyan]                this screen\n"
                "  [cyan]exit / quit[/cyan]          end session",
                title="help",
                border_style="yellow",
            )
        )

    def run(self) -> None:
        self.console.clear()
        self.console.print(
            Panel(
                "[bold white]★ ChatME ROLEPLAY ENGINE ★[/bold white]\n[dim]/help for commands[/dim]",
                style="bold blue",
                expand=False,
            )
        )

        while True:
            raw = Prompt.ask(
                "[bold green]Who am I roleplaying as?[/bold green] "
                "[dim](name + description, or /load <name>)[/dim]"
            ).strip()
            if not raw:
                self.console.print("[yellow]enter a character description[/yellow]")
                continue
            if raw.lower().startswith("/load "):
                if self.load_session(raw[6:].strip()):
                    break
                continue

            self.persona_name = self._extract_persona_name(raw)
            self.persona_desc = self._trim_user_self_reference(raw)

            player_raw = Prompt.ask(
                "[bold green]Your name in the story?[/bold green] [dim](enter to skip)[/dim]",
                default="",
            ).strip()
            if player_raw:
                self.player_label = player_raw.capitalize()

            # optional scene at startup
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

        mode_tag = " [dim][narrate][/dim]" if self.narrate_mode else ""
        self.console.print(
            f"\n[bold blue]playing as[/bold blue] [bold magenta]{self.persona_name}[/bold magenta]"
            f" — talking to [bold cyan]{self.player_label}[/bold cyan]{mode_tag}. "
            f"[bold red]exit[/bold red] to quit.\n"
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
                self.save_session(parts[1].strip() if len(parts) > 1 else None)
                continue

            if cmd.startswith("/load"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    self.console.print("[yellow]usage: /load <name>[/yellow]")
                else:
                    self.load_session(parts[1].strip())
                continue

            if cmd == "/sessions":
                self._list_sessions()
                continue

            if cmd == "/clear":
                self.history = [self.history[0]] if self.history else []
                self._condense_count = 0
                self.console.print("[yellow]history cleared[/yellow]")
                continue

            if cmd == "/debug":
                self.debug = not self.debug
                self.console.print(f"[dim]debug {'on' if self.debug else 'off'}[/dim]")
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
    RoleplayEngine().run()
