from __future__ import annotations

import os
import re
import sys
import json
import time
import datetime
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

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
    RELATION_WORDS,
    LOCATION_STOPWORDS,
    is_valid_name,
)
from models import WorldMemory

DEFAULT_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

load_dotenv()

_NAME_PATTERNS = [
    re.compile(r"(?:i am|i'm|my name is)\s+([A-Z][a-z]{1,})", re.IGNORECASE),
    re.compile(r"(?:this is|meet|introduce[d]?)\s+([A-Z][a-z]{1,})", re.IGNORECASE),
    re.compile(r"(?:call(?:ed)?|named?)\s+([A-Z][a-z]{1,})", re.IGNORECASE),
]
_AGE_PATTERN = re.compile(r"\b([A-Z][a-z]{1,})\s+is\s+(\d{1,3})\s*(?:years?\s*old)?")
_REL_PATTERN = re.compile(r"\b([A-Z][a-z]{1,})\s+is\s+(?:my|his|her|their|our)\s+(\w+)")
_REL_PATTERN2 = re.compile(r"(?:my|his|her|their|our)\s+(\w+)\s+([A-Z][a-z]{1,})")
_GROUP_REL_PATTERN = re.compile(
    r"\b([A-Z][a-z]{1,})\s+and\s+([A-Z][a-z]{1,})\s+are\s+(\w+)"
)
_DIRECTIVE_PATTERN = re.compile(r"\*([^*]+)\*")
_ACTION_STRIP = re.compile(r"\*[^*]+\*")
_NAME_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z ]{0,20}:\s*")
_LOCATION_PREPS = frozenset(
    ("in", "at", "near", "inside", "outside", "through", "across", "into")
)

# permanent skip codes
_HARD_FAIL_CODES = frozenset({401, 403, 404})


def _first_line_excerpt(text: str, limit: int = 80) -> str:
    return text.strip().split("\n")[0][:limit]


def _safe_json_load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _is_free(pricing: dict) -> bool:
    # both prompt and completion zero
    p = str(pricing.get("prompt", "")).strip()
    c = str(pricing.get("completion", "")).strip()
    return p == "0" and c == "0"


class _ModelPool:
    """smart rotation with tiered failure tracking"""

    _SOFT_TTL = 90.0  # seconds before soft-fail retried

    def __init__(self, models: list[str]) -> None:
        self._all: list[str] = list(models)
        self._soft: set[str] = set()
        self._hard: set[str] = set()
        self._last_ok: Optional[str] = None
        self._fail_ts: dict[str, float] = {}

    def mark_ok(self, m: str) -> None:
        self._last_ok = m
        self._soft.discard(m)
        self._hard.discard(m)
        self._fail_ts.pop(m, None)

    def mark_soft(self, m: str) -> None:
        self._soft.add(m)
        self._fail_ts[m] = time.monotonic()

    def mark_hard(self, m: str) -> None:
        self._hard.add(m)
        self._soft.discard(m)
        self._fail_ts.pop(m, None)

    def _promote_expired_soft(self) -> None:
        now = time.monotonic()
        expire = {
            m for m in self._soft if now - self._fail_ts.get(m, now) > self._SOFT_TTL
        }
        self._soft -= expire
        for m in expire:
            self._fail_ts.pop(m, None)

    def replace(self, models: list[str]) -> None:
        self._all = list(models)
        self._soft.clear()
        self._hard.clear()
        self._fail_ts.clear()

    def ordered(self) -> list[str]:
        self._promote_expired_soft()
        hard = self._hard
        soft = self._soft
        last = self._last_ok
        first = [last] if last and last in self._all and last not in hard else []
        clean = [m for m in self._all if m not in hard and m not in soft and m != last]
        retry = [m for m in self._all if m in soft and m not in hard]
        return first + clean + retry

    def available(self) -> int:
        return sum(1 for m in self._all if m not in self._hard)

    def top(self) -> Optional[str]:
        o = self.ordered()
        return o[0] if o else None


class RoleplayEngine:
    CONDENSE_THRESHOLD_MSGS = 30
    CONDENSE_THRESHOLD_TOKENS = 5000
    KEEP_RECENT_PAIRS = 8
    MAX_HISTORY_HARD_CAP = 80

    REPLY_MAX_TOKENS = 500
    SUMMARY_MAX_TOKENS = 350
    REQUEST_TIMEOUT = 60.0
    SUMMARY_TIMEOUT = 40.0
    MIN_RESPONSE_LEN = 8

    SESSIONS_DIR = Path("chatme_sessions")

    def __init__(
        self, model: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        self._original_stdout = None
        # windows utf-8 fix
        if sys.platform == "win32":
            import io

            self._original_stdout = sys.stdout
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )

        self.console = Console(force_terminal=True)
        self._override_model = model or os.environ.get("MODEL") or DEFAULT_MODEL
        self._override_api_key = api_key

        try:
            self._init_client()
            self._init_state()
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        if sys.platform == "win32" and self._original_stdout is not None:
            sys.stdout = self._original_stdout

    def _init_client(self) -> None:
        api_key = self._override_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            self.console.print("[bold red]OPENROUTER_API_KEY not set[/bold red]")
            sys.exit(1)

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=self.REQUEST_TIMEOUT
        )
        self._api_key = api_key

        free = self._fetch_free_models()
        if free:
            self._pool = _ModelPool(free)
            self._summary_pool = _ModelPool(free[:6])
            top = free[0].split("/")[-1].replace(":free", "")
            self.console.print(f"[green]{len(free)} free models  •  {top}[/green]")
        else:
            self._pool = _ModelPool([self._override_model])
            self._summary_pool = _ModelPool([self._override_model])
            self.console.print(f"[yellow]fallback: {self._override_model}[/yellow]")

    def _fetch_free_models(self) -> list[str]:
        try:
            req = Request(
                _MODELS_URL, headers={"Authorization": f"Bearer {self._api_key}"}
            )
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())

            if "error" in data:
                return []

            results = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                ctx = m.get("context_length") or 0
                pricing = m.get("pricing", {})
                arch = m.get("architecture", {})
                mod_in = arch.get("modality", "")

                if ctx <= 0 or not _is_free(pricing):
                    continue
                if "text" not in mod_in:
                    continue

                results.append(
                    {
                        "id": mid,
                        "ctx": ctx,
                        "is_moderated": m.get("top_provider", {}).get(
                            "is_moderated", True
                        ),
                    }
                )

            # moderated + highest context first
            results.sort(key=lambda x: (int(x["is_moderated"]), x["ctx"]), reverse=True)
            return [m["id"] for m in results]

        except (HTTPError, URLError, json.JSONDecodeError):
            return []

    def _refresh_models(self) -> bool:
        fresh = self._fetch_free_models()
        if not fresh:
            return False
        self._pool.replace(fresh)
        self._summary_pool.replace(fresh[:6])
        return True

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
        self.scene = ""
        self.mood = ""
        self._last_assistant_content = ""
        self._recent_assistant_excerpts: list[str] = []

    def _apply_session_data(self, data: dict) -> None:
        self.persona_name = data.get("persona_name", "Character")
        self.persona_desc = data.get("persona_desc", "")
        self.player_label = data.get("player_label", "You")
        self.lore = data.get("lore", "")
        self.history = data.get("history", [])
        self.memory = WorldMemory.from_dict(data.get("memory", {}))
        self._condense_count = data.get("condense_count", 0)
        self.scene = data.get("scene", "")
        self.mood = data.get("mood", "")
        self._recent_assistant_excerpts = data.get("recent_excerpts", [])

    def _build_few_shot(self) -> str:
        return (
            f"\nexample format:\n"
            f"{self.player_label}: Hey, you okay?\n"
            f'{self.persona_name}: *glances up* "Yeah, just thinking." *shifts slightly*\n'
        )

    def _build_no_repeat_block(self) -> str:
        if not self._recent_assistant_excerpts:
            return ""
        lines = "\n".join(f"- {e}" for e in self._recent_assistant_excerpts[-4:])
        return f"\navoid reusing these openings:\n{lines}\n"

    def _build_system_content(self, persona_desc: str, lore: str) -> str:
        world_info = self.memory.format_world()
        scene_block = f"\nSCENE: {self.scene}" if self.scene else ""
        mood_block = f"\nMOOD: {self.mood}" if self.mood else ""
        lore_block = (
            f"\nBACKSTORY (already happened, do not re-enact):\n{lore}\n"
            if lore and lore != "The story is just beginning."
            else ""
        )
        world_block = (
            f"\nKNOWN CHARACTERS:\n{world_info}\n" if world_info.strip() else ""
        )

        return (
            f"You are {persona_desc}\n"
            f"Your name: {self.persona_name}\n"
            f"The player is: {self.player_label}{scene_block}{mood_block}\n\n"
            f"RULES:\n"
            f"- Stay in character. Never say you are an AI.\n"
            f"- Only write your own character's words and actions.\n"
            f"- Never write what {self.player_label} does or says.\n"
            f"- Do NOT prefix your response with the character name.\n"
            f"- Never respond to your own actions — only respond to the player.\n"
            f"- When player writes *action*, perform that action.\n"
            f"- Reply as the character only. 2-3 sentences max.\n"
            f"- Use *italics* for actions/thoughts, dialogue in quotes.\n"
            f"- Always respond to the very last message.\n"
            f"{self._build_no_repeat_block()}"
            f"{self._build_few_shot()}"
            f"{lore_block}"
            f"{world_block}"
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
        self._rebuild_system()

    def _update_system_memory(self) -> None:
        self._patch_system_marker(CHARACTERS_MARKER, self.memory.format_world())

    def _track_excerpt(self, content: str) -> None:
        first = _first_line_excerpt(content)
        if first:
            self._recent_assistant_excerpts.append(first)
            if len(self._recent_assistant_excerpts) > 6:
                self._recent_assistant_excerpts = self._recent_assistant_excerpts[-6:]
        self._rebuild_system()

    def _extract_info_from_message(self, text: str, is_user: bool) -> None:
        for pat in _NAME_PATTERNS:
            for m in pat.finditer(text):
                if is_valid_name(m.group(1)):
                    self.memory.add_character(m.group(1), context=text[:80])

        for m in _AGE_PATTERN.finditer(text):
            name, age = m.group(1), m.group(2)
            if is_valid_name(name):
                c = self.memory.add_character(name, age=age, context=text[:80])
                if c and not c.age:
                    c.age = age

        for m in _REL_PATTERN.finditer(text):
            name, rel = m.group(1), m.group(2).lower()
            if is_valid_name(name) and rel in RELATION_WORDS:
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        for m in _REL_PATTERN2.finditer(text):
            rel, name = m.group(1).lower(), m.group(2)
            if rel in RELATION_WORDS and is_valid_name(name):
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        for m in _GROUP_REL_PATTERN.finditer(text):
            n1, n2, rt = m.group(1), m.group(2), m.group(3).lower()
            if is_valid_name(n1) and is_valid_name(n2):
                self.memory.add_character(n1, context=text[:80])
                self.memory.add_character(n2, context=text[:80])
                self.memory.add_relationship(n1, n2, rt, context=text[:80])

        # extract location candidates
        known = {c.lower() for c in self.memory.characters}
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in _LOCATION_PREPS and i + 1 < len(words):
                cand = words[i + 1].strip(".,!?;:\"'")
                if (
                    len(cand) > 2
                    and cand[0].isupper()
                    and cand.isalpha()
                    and cand.lower() not in LOCATION_STOPWORDS
                    and cand.lower() not in known
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

    def _build_kwargs(
        self, model: str, messages: list[dict], stream: bool, is_summary: bool
    ) -> dict:
        kw: dict = dict(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=self.SUMMARY_MAX_TOKENS if is_summary else self.REPLY_MAX_TOKENS,
            timeout=self.SUMMARY_TIMEOUT if is_summary else self.REQUEST_TIMEOUT,
        )
        if not is_summary:
            kw["temperature"] = 0.9
        return kw

    def call_with_failover(
        self,
        messages: list[dict],
        stream: bool = True,
        is_summary: bool = False,
    ) -> tuple[Optional[object], Optional[str]]:
        clean = self._clean(messages)
        if not clean:
            return None, None

        pool = self._summary_pool if is_summary else self._pool
        tried: set[str] = set()
        last_error: Optional[str] = None

        for attempt in range(2):
            for model in pool.ordered():
                if model in tried:
                    continue
                tried.add(model)

                try:
                    resp = self.client.chat.completions.create(
                        **self._build_kwargs(model, clean, stream, is_summary)
                    )
                    pool.mark_ok(model)
                    if self.debug:
                        self.console.print(f"[dim]✓ {model.split('/')[-1]}[/dim]")
                    return resp, model

                except APITimeoutError:
                    pool.mark_soft(model)
                    last_error = "timeout"
                    if self.debug:
                        self.console.print(
                            f"[dim red]timeout: {model.split('/')[-1]}[/dim red]"
                        )

                except APIError as e:
                    raw_code = getattr(e, "status_code", None) or getattr(e, "code", 0)
                    try:
                        code = int(raw_code)
                    except (TypeError, ValueError):
                        code = 0

                    if code in _HARD_FAIL_CODES:
                        pool.mark_hard(model)
                    else:
                        pool.mark_soft(model)

                    last_error = str(code)
                    if self.debug:
                        self.console.print(
                            f"[dim red]{code}: {model.split('/')[-1]}[/dim red]"
                        )

                except Exception as e:
                    pool.mark_soft(model)
                    last_error = str(e)[:40]
                    if self.debug:
                        self.console.print(
                            f"[dim red]{model.split('/')[-1]}: {last_error}[/dim red]"
                        )

            # refresh and retry once
            if attempt == 0:
                if self.debug:
                    self.console.print(
                        "[dim yellow]refreshing model list…[/dim yellow]"
                    )
                if not self._refresh_models():
                    break

        self.console.print(
            f"[bold red]no models available[/bold red] ({last_error})\n"
            "[dim]check your API key or wait a moment[/dim]"
        )
        return None, None

    def _consume_stream(self, response_obj) -> str:
        parts: list[str] = []
        try:
            with Live(
                Panel("", border_style="magenta"),
                auto_refresh=True,
                console=self.console,
                transient=True,
                refresh_per_second=15,
            ) as live:
                for chunk in response_obj:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    token = (delta.content or "") if delta else ""
                    if token:
                        parts.append(token)
                        if len(parts) % 4 == 0:
                            live.update(
                                Panel(Markdown("".join(parts)), border_style="magenta")
                            )
                if parts:
                    live.update(Panel(Markdown("".join(parts)), border_style="magenta"))
        except Exception as e:
            if self.debug:
                self.console.print(f"[dim red]stream: {e}[/dim red]")

        content = "".join(parts)
        if content.strip():
            self.console.print(Panel(Markdown(content), border_style="magenta"))
        return content

    def _get_reply(self, max_retries: int = 3) -> str:
        last_error = ""
        for attempt in range(max_retries):
            resp, used = self.call_with_failover(self.history, stream=True)
            if resp is None:
                last_error = "no response"
                continue

            content = self._consume_stream(resp)
            content_len = len(content.strip())

            # short response — try next model
            if content_len < self.MIN_RESPONSE_LEN:
                if self.debug:
                    self.console.print(
                        f"[dim yellow]short response ({content_len} chars), retrying...[/dim yellow]"
                    )
                if used:
                    self._pool.mark_soft(used)
                if attempt < max_retries - 1:
                    continue  # retry with next model
                else:
                    # Last attempt: if still short, return empty to signal failure
                    if content_len > 0:
                        self.console.print(
                            f"[dim red]warning: short response ({content_len} chars) from model[/dim red]"
                        )
                    return (
                        content if content_len >= 4 else ""
                    )  # Accept min 4 chars on final attempt

            return content

        # All retries failed
        self.console.print(
            f"[bold red]all {max_retries} retries failed: {last_error}[/bold red]"
        )
        return ""

    def _convo_msg_count(self) -> int:
        return sum(1 for m in self.history if m.get("role") != "system")

    def _should_condense(self) -> bool:
        n = self._convo_msg_count()
        keep_n = self.KEEP_RECENT_PAIRS * 2
        threshold = (
            keep_n + self.CONDENSE_THRESHOLD_MSGS
            if self._condense_count > 0
            else self.CONDENSE_THRESHOLD_MSGS
        )
        return (
            len(self.history) > self.MAX_HISTORY_HARD_CAP
            or n >= threshold
            or self._estimate_tokens(self.history) > self.CONDENSE_THRESHOLD_TOKENS
        )

    def _summarise(self, messages: list[dict]) -> Optional[str]:
        if not messages:
            return None

        lines = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if not role or role == "system":
                continue
            label = "Assistant" if role == "assistant" else "User"
            cleaned = _ACTION_STRIP.sub("", m.get("content", "")).strip()
            if cleaned:
                lines.append(f"[{label}]: {cleaned}")

        if not lines:
            return self._memory_fallback_summary(messages)

        transcript = "\n".join(lines)
        if len(transcript) > 6000:
            transcript = transcript[-6000:]

        known = ", ".join(self.memory.characters.keys()) or "none"
        prompt = [
            {
                "role": "system",
                "content": (
                    "Write 3-5 bullet points summarising what happened. "
                    "Past tense only. Events and decisions only. "
                    "No dialogue quotes. No scene description. No preamble. "
                    "Ignore actions in asterisks."
                ),
            },
            {"role": "user", "content": f"Characters: {known}\n\n{transcript}"},
        ]

        resp, _ = self.call_with_failover(prompt, stream=False, is_summary=True)
        if resp:
            raw = (resp.choices[0].message.content or "").strip()
            if raw and len(raw) < len(transcript) * 0.8:
                return raw

        return self._memory_fallback_summary(messages)

    def _memory_fallback_summary(self, messages: list[dict]) -> str:
        bullets = []
        if self.lore and self.lore != "The story is just beginning.":
            bullets.append(f"• Lore: {self.lore[:200]}")
        for c in list(self.memory.characters.values())[:5]:
            parts = [c.name]
            if c.age:
                parts.append(f"age {c.age}")
            if c.description:
                parts.append(c.description[:60])
            bullets.append(f"• {', '.join(parts)}")
        for r in self.memory.relationships[:4]:
            bullets.append(f"• {r.from_char} ↔ {r.to_char} ({r.rel_type})")
        if self.memory.locations:
            bullets.append(f"• Locations: {', '.join(self.memory.locations[:5])}")
        for msg in [m["content"] for m in messages if m.get("role") == "assistant"][
            -2:
        ]:
            clean = _ACTION_STRIP.sub("", msg).strip()[:100]
            if clean:
                bullets.append(f"• Recent: {clean}")
        return "\n".join(bullets) if bullets else "• Story continued."

    def condense_logic(self) -> None:
        try:
            self.console.print(
                "\n[dim italic yellow]⚡ condensing…[/dim italic yellow]"
            )
            keep_n = self.KEEP_RECENT_PAIRS * 2

            if len(self.history) <= keep_n + 1:
                return

            to_summarise = self.history[1:-keep_n]
            if not to_summarise:
                self.history = [self.history[0]] + self.history[-keep_n:]
                return

            summary = self._summarise(to_summarise)
            if summary:
                combined = (
                    summary
                    if self._condense_count == 0
                    or self.lore == "The story is just beginning."
                    else f"{self.lore}\n\n{summary}"
                )
                # cap lore size
                if len(combined) > 1200:
                    combined = combined[-1200:]
            else:
                combined = (
                    self.lore
                    if self.lore != "The story is just beginning."
                    else "• Story began."
                )

            self._update_system_lore(combined)
            self.history = [self.history[0]] + self.history[-keep_n:]
            self._condense_count += 1

        except Exception as e:
            if self.debug:
                self.console.print(f"[dim red]condense error: {e}[/dim red]")
            keep_n = self.KEEP_RECENT_PAIRS * 2
            if len(self.history) > keep_n + 1:
                self.history = [self.history[0]] + self.history[-keep_n:]

    def _check_and_condense(self) -> None:
        if self._should_condense():
            self.condense_logic()

    def _handle_set_command(self, args: str) -> None:
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            self.console.print(
                "[yellow]usage: /set <name|player|scene|mood> <value>[/yellow]"
            )
            return
        key, value = parts[0].lower(), parts[1].strip()
        if key == "scene":
            self.scene = value
            self._rebuild_system()
            self.console.print(f"[green]✓ scene:[/green] {value}")
        elif key == "mood":
            self.mood = value
            self._rebuild_system()
            self.console.print(f"[green]✓ mood:[/green] {value}")
        elif key == "name":
            old = self.persona_name
            self.persona_name = value.capitalize()
            self._rebuild_system()
            self.console.print(f"[green]✓ name:[/green] {old} → {self.persona_name}")
        elif key == "player":
            self.player_label = value.capitalize()
            self._rebuild_system()
            self.console.print(f"[green]✓ player:[/green] {self.player_label}")
        else:
            self.console.print(f"[yellow]unknown: {key}[/yellow]")

    def _get_session_path(self, name: str) -> Path:
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        session_re = re.sub(r"[\w-]", "_", name)
        return self.SESSIONS_DIR / f"{session_re}.json"

    def save_session(self, name: Optional[str] = None) -> None:
        if not name:
            exact = self.SESSIONS_DIR / f"{self.persona_name}.json"
            if exact.exists():
                name = self.persona_name
            else:
                existing = list(self.SESSIONS_DIR.glob(f"{self.persona_name}*.json"))
                name = (
                    existing[0].stem
                    if existing
                    else (
                        f"{self.persona_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                )
        path = self._get_session_path(name)
        try:
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
                        "recent_excerpts": self._recent_assistant_excerpts,
                        "saved_at": datetime.datetime.now().isoformat(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.console.print(
                f"[bold green]✓ saved[/bold green]  [dim]{path.name}[/dim]"
            )
        except OSError as e:
            self.console.print(f"[bold red]save failed:[/bold red] {e}")

    def load_session(self, name: str) -> bool:
        path = self._get_session_path(name)
        if not path.exists():
            direct = Path(name)
            path = direct if direct.exists() else path
        if not path.exists():
            self.console.print(f"[bold red]not found:[/bold red] {name}")
            return False
        data = _safe_json_load(path)
        if data is None:
            self.console.print("[bold red]load failed: invalid json[/bold red]")
            return False
        self._apply_session_data(data)
        self.console.print(f"[bold green]✓ loaded[/bold green]  [dim]{path.name}[/dim]")
        return True

    def _list_sessions(self) -> list[dict]:
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        out = []
        for f in sorted(self.SESSIONS_DIR.glob("*.json")):
            d = _safe_json_load(f)
            out.append(
                {
                    "name": f.stem,
                    "saved": d.get("saved_at", "?")[:16] if d else "?",
                    "persona": d.get("persona_name", "?") if d else "?",
                    "scene": d.get("scene", "")[:30] if d else "",
                    "path": f,
                }
            )
        return out

    def _show_sessions_table(self) -> None:
        sessions = self._list_sessions()
        if not sessions:
            self.console.print("[dim]no saved sessions[/dim]")
            return
        t = Table(title="💾 Sessions", show_header=True, header_style="bold cyan")
        t.add_column("#", style="dim", width=4)
        t.add_column("Name", style="cyan")
        t.add_column("Character", style="magenta")
        t.add_column("Scene", style="dim")
        t.add_column("Saved", style="dim")
        for i, s in enumerate(sessions, 1):
            t.add_row(str(i), s["name"], s["persona"], s["scene"] or "—", s["saved"])
        self.console.print(t)

    def _load_session_by_path(self, path: Path) -> bool:
        data = _safe_json_load(path)
        if data is None:
            self.console.print("[bold red]load failed: invalid json[/bold red]")
            return False
        self._apply_session_data(data)
        self.console.print(
            f"\n[bold green]✓ loaded[/bold green]  "
            f"[magenta]{data.get('persona_name', '?')}[/magenta]"
        )
        if self.scene:
            self.console.print(f"  [dim]scene: {self.scene}[/dim]")
        return True

    def _edit_character_prompts(self) -> None:
        new_name = Prompt.ask(
            f"[bold]name[/bold] [dim]({self.persona_name})[/dim]",
            default=self.persona_name,
        ).strip()
        if new_name:
            self.persona_name = new_name.capitalize()
        new_desc = Prompt.ask(
            "[bold]description[/bold] [dim](enter to keep)[/dim]",
            default=self.persona_desc,
        ).strip()
        if new_desc:
            self.persona_desc = new_desc
        self._rebuild_system()
        self.console.print("[green]✓ updated[/green]")

    def _show_memory(self) -> None:
        self.console.print(
            Panel(self.memory.format_world(), title="📚 memory", border_style="blue")
        )

    def _show_status(self) -> None:
        top = self._pool.top()
        self.console.print(
            Panel(
                "\n".join(
                    [
                        f"[bold]Character:[/bold]  {self.persona_name}",
                        f"[bold]Player:[/bold]     {self.player_label}",
                        f"[bold]Model:[/bold]      {(top or '—').split('/')[-1].replace(':free','')}",
                        f"[bold]Available:[/bold]  {self._pool.available()} models",
                        f"[bold]Scene:[/bold]      {self.scene or '—'}",
                        f"[bold]Mood:[/bold]       {self.mood  or '—'}",
                        f"[bold]Messages:[/bold]   {self._convo_msg_count()}",
                        f"[bold]Condenses:[/bold]  {self._condense_count}",
                    ]
                ),
                title="📊 status",
                border_style="cyan",
            )
        )

    def _show_help(self) -> None:
        self.console.print(
            Panel(
                "[bold green]GENERAL[/bold green]\n"
                "  /help  /status  /memory  /lore\n\n"
                "[bold green]CHARACTER[/bold green]\n"
                "  /set name <n>    /set player <n>\n"
                "  /set scene <s>   /set mood <m>\n\n"
                "[bold green]CONVERSATION[/bold green]\n"
                "  /retry  /clear\n\n"
                "[bold green]SESSIONS[/bold green]\n"
                "  /save [name]  /load  /new  /sessions\n\n"
                "[bold green]OTHER[/bold green]\n"
                "  /debug    exit / quit",
                title="❓ help",
                border_style="yellow",
            )
        )

    def _strip_name_prefix(self, content: str) -> str:
        s = content.strip()
        return (
            _NAME_PREFIX_RE.sub("", s, count=1).strip()
            if _NAME_PREFIX_RE.match(s)
            else s
        )

    def _parse_directives(self, user_input: str) -> tuple[str, str]:
        directives = _DIRECTIVE_PATTERN.findall(user_input)
        clean_speech = _DIRECTIVE_PATTERN.sub("", user_input).strip()
        return clean_speech, " ".join(directives)

    def _startup_flow(self) -> bool:
        try:
            sessions = self._list_sessions()

            if not sessions:
                self.console.print("\n[bold]no sessions — new character[/bold]\n")
            else:
                self.console.print("\n[bold]sessions:[/bold]")
                self._show_sessions_table()
                self.console.print("\n  [dim]ENTER = new  |  # or name = load[/dim]")
                choice = Prompt.ask("\n[bold green]>[/bold green]", default="").strip()

                if choice.lower() in ("exit", "quit", "q", "e"):
                    self.console.print("[bold green]goodbye[/bold green]")
                    return False

                if choice:
                    selected = None
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(sessions):
                            selected = sessions[idx]
                    else:
                        for s in sessions:
                            if choice.lower() in s["name"].lower():
                                selected = s
                                break

                    if not selected:
                        self.console.print(f"[yellow]not found: {choice}[/yellow]")
                        return False

                    if self._load_session_by_path(selected["path"]):
                        if (
                            Prompt.ask(
                                "[bold yellow]edit character?[/bold yellow] [dim](y/N)[/dim]",
                                default="n",
                            ).lower()
                            == "y"
                        ):
                            self._edit_character_prompts()
                        return True
                    return False

            while True:
                name_in = Prompt.ask("[bold green]character name?[/bold green]").strip()
                if not name_in:
                    self.console.print("[yellow]name required[/yellow]")
                    continue
                self.persona_name = name_in.capitalize()

                desc_in = Prompt.ask(
                    "[bold green]description?[/bold green] [dim](personality, appearance, backstory)[/dim]"
                ).strip()
                if not desc_in:
                    self.console.print("[yellow]description required[/yellow]")
                    continue
                self.persona_desc = desc_in

                player_in = Prompt.ask(
                    "[bold green]your name?[/bold green] [dim](skip)[/dim]", default=""
                ).strip()
                if player_in:
                    self.player_label = player_in.capitalize()

                scene_in = Prompt.ask(
                    "[bold green]opening scene?[/bold green] [dim](skip)[/dim]",
                    default="",
                ).strip()
                if scene_in:
                    self.scene = scene_in
                    self.lore = f"The story begins: {scene_in}"

                self.memory.add_character(
                    self.persona_name,
                    description=self.persona_desc,
                    context="player character",
                )
                self.history.append(
                    {
                        "role": "system",
                        "content": self._build_system_content(
                            self.persona_desc, self.lore
                        ),
                    }
                )
                return True

        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[bold green]goodbye[/bold green]")
            return False

    def _chat_loop(self) -> None:
        top = self._pool.top()
        top_short = top.split("/")[-1].replace(":free", "") if top else "none"
        self.console.print(
            f"\n[bold green]✓[/bold green]  "
            f"[magenta]{self.persona_name}[/magenta]  "
            f"[dim]{self.player_label}  •  {top_short}"
            + (f"  •  {self.scene}" if self.scene else "")
            + f"[/dim]\n[dim]/help  •  exit to quit[/dim]\n"
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

            if cmd in ("exit", "/exit", "quit", "/quit", "q", "/q", "e", "/e"):
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
            if cmd == "/sessions":
                self._show_sessions_table()
                continue
            if cmd == "/lore":
                self.console.print(Panel(self.lore, title="lore", border_style="green"))
                continue
            if cmd == "/debug":
                self.debug = not self.debug
                self.console.print(
                    f"[green]✓ debug {'on' if self.debug else 'off'}[/green]"
                )
                continue

            if cmd == "/retry":
                if self.history and self.history[-1]["role"] == "assistant":
                    self.history.pop()
                    self.console.print(Rule(style="dim"))
                    self.console.print(
                        f"[bold magenta]{self.persona_name}[/bold magenta]"
                    )
                    content = self._strip_name_prefix(self._get_reply())
                    if content and len(content) >= 5:
                        self._last_assistant_content = content
                        self._track_excerpt(content)
                        self.history.append(
                            {
                                "role": "assistant",
                                "content": f"{self.persona_name}: {content}",
                            }
                        )
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

            if cmd.startswith("/load") or cmd == "/new":
                if self._startup_flow():
                    self._chat_loop()
                return

            if cmd == "/clear":
                if (
                    Prompt.ask(
                        "[yellow]clear history?[/yellow] [dim](y/N)[/dim]", default="n"
                    ).lower()
                    == "y"
                ):
                    self.history = [self.history[0]] if self.history else []
                    self._condense_count = 0
                    self._recent_assistant_excerpts = []
                    self.console.print("[green]✓ cleared[/green]")
                continue

            self._extract_info_from_message(user_input, is_user=True)
            clean_speech, directives = self._parse_directives(user_input)

            labeled = (
                f"{self.player_label}: {clean_speech}\n[ACTION FOR {self.persona_name.upper()}]: {directives}"
                if directives
                else f"{self.player_label}: {user_input}"
            )

            self.history.append({"role": "user", "content": labeled})
            self._check_and_condense()

            self.console.print(Rule(style="dim"))
            self.console.print(f"[bold magenta]{self.persona_name}[/bold magenta]")

            content = self._strip_name_prefix(self._get_reply())

            if not content or len(content) < 5:
                self.console.print("[dim red]no response — try again[/dim red]")
                self.history.pop()
                continue

            self._extract_info_from_message(content, is_user=False)
            self._update_system_memory()
            self._last_assistant_content = content
            self._track_excerpt(content)
            self.history.append(
                {"role": "assistant", "content": f"{self.persona_name}: {content}"}
            )
            self._msg_count += 1

            # periodic save reminder
            if self._msg_count % 20 == 0:
                self.console.print("[dim yellow]💾 /save to keep progress[/dim yellow]")

    def run(self) -> None:
        self.console.clear()
        self.console.print(
            Panel(
                "[bold white]★ ChatME ROLEPLAY ENGINE ★[/bold white]\n"
                "[dim]/help • /status • /memory[/dim]",
                style="bold blue",
                expand=False,
            )
        )
        while self._startup_flow():
            self._chat_loop()


if __name__ == "__main__":
    import argparse
    import traceback

    def _handle_exc(t, v, tb):
        if issubclass(t, KeyboardInterrupt):
            sys.__excepthook__(t, v, tb)
            return
        traceback.print_exception(t, v, tb)

    sys.excepthook = _handle_exc

    ap = argparse.ArgumentParser(description="ChatME Roleplay Engine")
    ap.add_argument("--model", "-m", help="model override")
    ap.add_argument("--key", "-k", help="api key override")
    ap.add_argument(
        "--showmodel",
        "-sm",
        action="store_true",
        help="Show available models and exit (does not start chat)",
    )
    args = ap.parse_args()

    if args.showmodel:
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
        try:
            RoleplayEngine(model=args.model, api_key=args.key).run()
        except (KeyboardInterrupt, EOFError):
            print("\ngoodbye")
