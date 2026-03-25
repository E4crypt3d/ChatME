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
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from dotenv import load_dotenv

from constants import (
    CHARACTERS_MARKER,
    RELATION_WORDS,
    LOCATION_STOPWORDS,
    is_valid_name,
    normalise_name,
)
from models import WorldMemory

# provider config
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "base_url": OPENROUTER_BASE_URL,
        "models_url": _MODELS_URL,
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "openrouter/free",
    },
    "groq": {
        "base_url": GROQ_BASE_URL,
        "models_url": None,
        "api_key_env": "GROQ_API_KEY",
        "default_model": "groq/compound-mini",
    },
}
DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "openrouter/free"

# compiled patterns
_NAME_PATTERNS = [
    re.compile(r"(?:i am|i'm|my name is)\s+([A-Z][a-z]{1,20})", re.IGNORECASE),
    re.compile(r"(?:this is|meet|introduce[d]?)\s+([A-Z][a-z]{1,20})", re.IGNORECASE),
    re.compile(r"(?:call(?:ed)?|named?)\s+([A-Z][a-z]{1,20})", re.IGNORECASE),
    re.compile(r"^([A-Z][a-z]{1,20})(?:\s+here|\s+speaking)?[,!.]", re.MULTILINE),
]
_AGE_PATTERN = re.compile(r"\b([A-Z][a-z]{1,20})\s+is\s+(\d{1,3})\s*(?:years?\s*old)?")
_REL_PATTERN = re.compile(
    r"\b([A-Z][a-z]{1,20})\s+is\s+(?:my|his|her|their|our)\s+(\w+)"
)
_REL_PATTERN2 = re.compile(r"(?:my|his|her|their|our)\s+(\w+)\s+([A-Z][a-z]{1,20})")
_GROUP_REL_PATTERN = re.compile(
    r"\b([A-Z][a-z]{1,20})\s+and\s+([A-Z][a-z]{1,20})\s+are\s+(\w+)"
)
_DIRECTIVE_PATTERN = re.compile(r"\*([^*]+)\*")
_ACTION_STRIP = re.compile(r"\*[^*]+\*")
_NAME_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z ]{0,20}:\s*")
_LOCATION_PREPS = frozenset(
    ("in", "at", "near", "inside", "outside", "through", "across", "into")
)

_HARD_FAIL_CODES = frozenset({401, 403, 404})
_RETRY_WAIT = 1.5  # seconds between model-fail retries

load_dotenv()


# helpers
def _first_line_excerpt(text: str, limit: int = 80) -> str:
    return text.strip().split("\n")[0][:limit]


def _safe_json_load(path: Path) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _is_free(pricing: dict) -> bool:
    p = str(pricing.get("prompt", "")).strip()
    c = str(pricing.get("completion", "")).strip()
    return p in ("0", "0.0") and c in ("0", "0.0")


# ══════════════════════════════════════════════════════════════════════════════
class _ModelPool:
    """Smart model rotation with tiered failure tracking."""

    _SOFT_TTL = 120.0  # seconds before a soft-failed model is retried

    def __init__(self, models: list[str]) -> None:
        self._all: list[str] = list(models)
        self._soft: set[str] = set()
        self._hard: set[str] = set()
        self._last_ok: Optional[str] = None
        self._fail_ts: dict[str, float] = {}

    # state mutation
    def mark_ok(self, m: str) -> None:
        self._last_ok = m
        self._soft.discard(m)
        self._hard.discard(m)
        self._fail_ts.pop(m, None)

    def mark_soft(self, m: str) -> None:
        self._soft.add(m)
        self._fail_ts.setdefault(m, time.monotonic())

    def mark_hard(self, m: str) -> None:
        self._hard.add(m)
        self._soft.discard(m)
        self._fail_ts.pop(m, None)

    def replace(self, models: list[str]) -> None:
        self._all = list(models)
        self._soft.clear()
        self._hard.clear()
        self._fail_ts.clear()
        # keep _last_ok if it's still in the new list
        if self._last_ok and self._last_ok not in self._all:
            self._last_ok = None

    # scheduling
    def _promote_expired_soft(self) -> None:
        now = time.monotonic()
        expired = {
            m for m in self._soft if now - self._fail_ts.get(m, now) > self._SOFT_TTL
        }
        self._soft -= expired
        for m in expired:
            self._fail_ts.pop(m, None)

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

    def total(self) -> int:
        return len(self._all)

    def top(self) -> Optional[str]:
        o = self.ordered()
        return o[0] if o else None

    def short_name(self, model: Optional[str] = None) -> str:
        m = model or self.top() or ""
        return m.split("/")[-1].replace(":free", "")


# ══════════════════════════════════════════════════════════════════════════════
class RoleplayEngine:
    # constants
    CONDENSE_THRESHOLD_MSGS = 30
    CONDENSE_THRESHOLD_TOKENS = 5_000
    KEEP_RECENT_PAIRS = 8
    MAX_HISTORY_HARD_CAP = 80

    REPLY_MAX_TOKENS = 2048
    SUMMARY_MAX_TOKENS = 400
    REQUEST_TIMEOUT = 60.0
    SUMMARY_TIMEOUT = 45.0
    MIN_RESPONSE_LEN = 8

    SESSIONS_DIR = Path("chatme_sessions")

    # init
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self._original_stdout = None
        if sys.platform == "win32":
            import io

            self._original_stdout = sys.stdout
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )

        self.console = Console(force_terminal=True)

        self._provider = (
            provider or os.environ.get("PROVIDER") or DEFAULT_PROVIDER
        ).lower()
        if self._provider not in PROVIDERS:
            self.console.print(
                f"[bold red]Unknown provider '{self._provider}'.[/bold red]"
            )
            self.console.print(f"[dim]Choose from: {', '.join(PROVIDERS)}[/dim]")
            sys.exit(1)

        self._explicit_model = model
        self._override_model = (
            model
            or os.environ.get("MODEL")
            or PROVIDERS[self._provider]["default_model"]
        )
        self._override_api_key = api_key
        self._api_key_env = PROVIDERS[self._provider]["api_key_env"]

        try:
            self._init_client()
            self._init_state()
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        if sys.platform == "win32" and self._original_stdout is not None:
            sys.stdout = self._original_stdout

    # client setup
    def _init_client(self) -> None:
        cfg = PROVIDERS[self._provider]
        api_key = self._override_api_key or os.environ.get(self._api_key_env)

        if not api_key:
            self.console.print(
                f"[bold red]No API key found.[/bold red]\n"
                f"[dim]Set the [bold]{self._api_key_env}[/bold] environment variable "
                f"or pass [bold]--key[/bold] on the command line.[/dim]"
            )
            sys.exit(1)

        self.client = OpenAI(
            base_url=cfg["base_url"], api_key=api_key, timeout=self.REQUEST_TIMEOUT
        )
        self._api_key = api_key

        # resolve explicit model override
        explicit = self._explicit_model
        if not explicit:
            env_m = os.environ.get("MODEL", "")
            if env_m and env_m != cfg["default_model"]:
                explicit = env_m

        if explicit:
            self._pool = _ModelPool([explicit])
            self._summary_pool = _ModelPool([explicit])
            self.console.print(f"[green]Model:[/green] {explicit}")
        elif self._provider == "openrouter":
            free = self._fetch_free_models()
            if free:
                self._pool = _ModelPool(free)
                self._summary_pool = _ModelPool(free[:8])
                self.console.print(
                    f"[green]{len(free)} free models available[/green]  "
                    f"[dim]top: {free[0].split('/')[-1].replace(':free','')}[/dim]"
                )
            else:
                self._pool = _ModelPool([self._override_model])
                self._summary_pool = _ModelPool([self._override_model])
                self.console.print(
                    f"[yellow]Fallback model: {self._override_model}[/yellow]"
                )
        else:
            self._pool = _ModelPool([self._override_model])
            self._summary_pool = _ModelPool([self._override_model])
            self.console.print(
                f"[green]Provider: {self._provider}[/green]  "
                f"[dim]{self._override_model}[/dim]"
            )

    # fetch free models
    def _fetch_free_models(self) -> list[str]:
        try:
            req = Request(
                _MODELS_URL, headers={"Authorization": f"Bearer {self._api_key}"}
            )
            with urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
        except (HTTPError, URLError, json.JSONDecodeError, OSError):
            return []

        if "error" in data:
            return []

        results = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            ctx = m.get("context_length") or 0
            pricing = m.get("pricing", {})
            arch = m.get("architecture", {})

            if ctx <= 0 or not _is_free(pricing):
                continue
            if "text" not in arch.get("modality", "text"):
                continue

            results.append(
                {
                    "id": mid,
                    "ctx": ctx,
                    "is_moderated": bool(
                        m.get("top_provider", {}).get("is_moderated", True)
                    ),
                }
            )

        # prefer moderated + largest context
        results.sort(key=lambda x: (int(x["is_moderated"]), x["ctx"]), reverse=True)
        return [m["id"] for m in results]

    def _refresh_models(self) -> bool:
        fresh = self._fetch_free_models()
        if not fresh:
            return False
        self._pool.replace(fresh)
        self._summary_pool.replace(fresh[:8])
        return True

    # state init
    def _init_state(self) -> None:
        self.debug: bool = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
        self.history: list[dict] = []
        self.persona_name: str = "Character"
        self.persona_desc: str = ""
        self.player_label: str = "You"
        self.lore: str = "The story is just beginning."
        self.memory: WorldMemory = WorldMemory()
        self._msg_count: int = 0
        self._condense_count: int = 0
        self.scene: str = ""
        self.mood: str = ""
        self._last_assistant_content: str = ""
        self._recent_assistant_excerpts: list[str] = []
        self._unsaved_msgs: int = 0  # NEW: track since last save

    def _apply_session_data(self, data: dict) -> None:
        self.persona_name = data.get("persona_name", "Character")
        self.persona_desc = data.get("persona_desc", "")
        self.player_label = data.get("player_label", "You")
        self.lore = data.get("lore", "The story is just beginning.")
        self.history = data.get("history", [])
        self.memory = WorldMemory.from_dict(data.get("memory", {}))
        self._condense_count = data.get("condense_count", 0)
        self.scene = data.get("scene", "")
        self.mood = data.get("mood", "")
        self._recent_assistant_excerpts = data.get("recent_excerpts", [])
        self._msg_count = data.get("msg_count", 0)
        self._unsaved_msgs = 0

    # build system prompt
    def _build_few_shot(self) -> str:
        return (
            f"\nExample format:\n"
            f"{self.player_label}: Hey, you okay?\n"
            f'{self.persona_name}: *glances up* "Yeah, just thinking." *shifts slightly*\n'
        )

    def _build_no_repeat_block(self) -> str:
        if not self._recent_assistant_excerpts:
            return ""
        lines = "\n".join(f"  - {e}" for e in self._recent_assistant_excerpts[-5:])
        return f"\nDo NOT reuse these recent openings:\n{lines}\n"

    def _build_system_content(self, persona_desc: str, lore: str) -> str:
        world_info = self.memory.format_world()
        scene_block = f"\nCURRENT SCENE: {self.scene}" if self.scene else ""
        mood_block = f"\nCURRENT MOOD: {self.mood}" if self.mood else ""
        lore_block = (
            f"\nBACKSTORY (already happened — do NOT re-enact):\n{lore}\n"
            if lore and lore != "The story is just beginning."
            else ""
        )
        world_block = (
            f"\nKNOWN WORLD:\n{world_info}\n"
            if world_info.strip() != "No world info yet."
            else ""
        )

        return (
            f"You are {persona_desc}\n"
            f"Your name: {self.persona_name}\n"
            f"The player is called: {self.player_label}"
            f"{scene_block}{mood_block}\n\n"
            f"STRICT RULES:\n"
            f"• Stay fully in character at all times. Never say you are an AI.\n"
            f"• Write ONLY your own character's words and actions.\n"
            f"• NEVER write what {self.player_label} does, says, thinks, or feels.\n"
            f"• Do NOT prefix your reply with your character's name.\n"
            f'• Use *italics* for actions/thoughts; use "quotes" for spoken dialogue.\n'
            f"• Keep replies to 2–3 sentences unless the scene demands more.\n"
            f"• Always respond to the VERY LAST message from {self.player_label}.\n"
            f"• When {self.player_label} writes *action*, react naturally to that action.\n"
            f"• Vary your sentence structure and opening words each reply.\n"
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
            if len(self._recent_assistant_excerpts) > 8:
                self._recent_assistant_excerpts = self._recent_assistant_excerpts[-8:]
        self._rebuild_system()

    # extract info
    def _extract_info_from_message(self, text: str, is_user: bool) -> None:
        # character names
        for pat in _NAME_PATTERNS:
            for m in pat.finditer(text):
                candidate = normalise_name(m.group(1))
                if is_valid_name(candidate):
                    self.memory.add_character(candidate, context=text[:80])

        # ages
        for m in _AGE_PATTERN.finditer(text):
            name, age = normalise_name(m.group(1)), m.group(2)
            if is_valid_name(name) and int(age) < 150:
                c = self.memory.add_character(name, age=age, context=text[:80])
                if c and not c.age:
                    c.age = age

        # relationships: "X is my/his/her/their friend"
        for m in _REL_PATTERN.finditer(text):
            name, rel = normalise_name(m.group(1)), m.group(2).lower()
            if is_valid_name(name) and rel in RELATION_WORDS:
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        # relationships: "my/his/her friend X"
        for m in _REL_PATTERN2.finditer(text):
            rel, name = m.group(1).lower(), normalise_name(m.group(2))
            if rel in RELATION_WORDS and is_valid_name(name):
                self.memory.add_character(name, context=text[:80])
                self.memory.add_relationship(
                    self.persona_name, name, rel, context=text[:80]
                )

        # group: "X and Y are brothers"
        for m in _GROUP_REL_PATTERN.finditer(text):
            n1, n2, rt = (
                normalise_name(m.group(1)),
                normalise_name(m.group(2)),
                m.group(3).lower(),
            )
            if is_valid_name(n1) and is_valid_name(n2):
                self.memory.add_character(n1, context=text[:80])
                self.memory.add_character(n2, context=text[:80])
                self.memory.add_relationship(n1, n2, rt, context=text[:80])

        # locations
        known_lower = {c.lower() for c in self.memory.characters}
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in _LOCATION_PREPS and i + 1 < len(words):
                cand = words[i + 1].strip(".,!?;:\"'")
                if (
                    len(cand) > 2
                    and cand[0].isupper()
                    and cand.isalpha()
                    and cand.lower() not in LOCATION_STOPWORDS
                    and cand.lower() not in known_lower
                ):
                    self.memory.add_location(cand)

    # api helpers
    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        """Strip unknown keys and empty messages."""
        cleaned = []
        for m in messages:
            if (
                isinstance(m, dict)
                and m.get("role") in ("system", "user", "assistant")
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            ):
                cleaned.append({"role": m["role"], "content": m["content"]})
        return cleaned

    def _estimate_tokens(self, messages: list[dict]) -> int:
        return sum(len(m.get("content", "")) // 4 + 1 for m in messages)

    def _build_kwargs(
        self,
        model: str,
        messages: list[dict],
        stream: bool,
        is_summary: bool,
    ) -> dict:
        kw: dict = dict(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=self.SUMMARY_MAX_TOKENS if is_summary else self.REPLY_MAX_TOKENS,
            timeout=self.SUMMARY_TIMEOUT if is_summary else self.REQUEST_TIMEOUT,
        )
        if not is_summary:
            kw["temperature"] = 0.92
            kw["top_p"] = 0.95
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
        tried: set = set()
        last_error = ""

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
                        last_error = f"HTTP {code} (permanent)"
                    else:
                        pool.mark_soft(model)
                        last_error = f"HTTP {code}"

                    if self.debug:
                        self.console.print(
                            f"[dim red]{code}: {model.split('/')[-1]}[/dim red]"
                        )

                except Exception as e:
                    pool.mark_soft(model)
                    last_error = str(e)[:60]
                    if self.debug:
                        self.console.print(
                            f"[dim red]{model.split('/')[-1]}: {last_error}[/dim red]"
                        )

                # small pause between failures to avoid hammering the API
                time.sleep(_RETRY_WAIT)

            if attempt == 0:
                if self.debug:
                    self.console.print(
                        "[dim yellow]Refreshing model list…[/dim yellow]"
                    )
                if not self._refresh_models():
                    break

        self.console.print(
            f"[bold red]❌ No models responded[/bold red] [dim]({last_error})[/dim]\n"
            "[dim]Check your API key, internet connection, or wait a moment.[/dim]"
        )
        return None, None

    # stream consumer
    def _consume_stream(self, response_obj) -> str:
        parts: list[str] = []
        try:
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
        if content.strip():
            self.console.print(Panel(Markdown(content), border_style="magenta"))
        return content

    # get reply
    def _get_reply(self, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            resp, used = self.call_with_failover(self.history, stream=True)
            if resp is None:
                continue

            content = self._consume_stream(resp)
            clen = len(content.strip())

            if clen >= self.MIN_RESPONSE_LEN:
                return content

            # too short — penalise and retry
            if self.debug:
                self.console.print(
                    f"[dim yellow]Short response ({clen} chars) on attempt {attempt+1}; retrying…[/dim yellow]"
                )
            if used:
                self._pool.mark_soft(used)
            if attempt < max_retries - 1:
                time.sleep(_RETRY_WAIT)
                continue
            # last chance: accept anything ≥ 4 chars
            return content if clen >= 4 else ""

        self.console.print(
            f"[bold red]All {max_retries} reply attempts failed.[/bold red]"
        )
        return ""

    # history helpers
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
        if len(transcript) > 6_000:
            transcript = transcript[-6_000:]

        known = ", ".join(self.memory.characters.keys()) or "none"
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarise the following roleplay exchange in 3–5 bullet points. "
                    "Past tense only. Concrete events and decisions only. "
                    "No dialogue quotes. No scene description. No preamble. "
                    "Strip all asterisk-actions."
                ),
            },
            {"role": "user", "content": f"Known characters: {known}\n\n{transcript}"},
        ]

        resp, _ = self.call_with_failover(prompt, stream=False, is_summary=True)
        if resp:
            raw = (resp.choices[0].message.content or "").strip()
            if raw and len(raw) < len(transcript) * 0.85:
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
            bullets.append(f"• Character: {', '.join(parts)}")
        for r in self.memory.relationships[:5]:
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
                "\n[dim italic yellow]⚡ Condensing memory…[/dim italic yellow]"
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
                if len(combined) > 1_500:
                    combined = combined[-1_500:]
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

    # set command
    def _handle_set_command(self, args: str) -> None:
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            self.console.print(
                "[yellow]Usage: /set <name|player|scene|mood|desc> <value>[/yellow]"
            )
            return
        key, value = parts[0].lower(), parts[1].strip()
        handlers = {
            "scene": self._set_scene,
            "mood": self._set_mood,
            "name": self._set_name,
            "player": self._set_player,
            "desc": self._set_desc,
        }
        h = handlers.get(key)
        if h:
            h(value)
        else:
            self.console.print(
                f"[yellow]Unknown setting: {key}[/yellow]  [dim](name|player|scene|mood|desc)[/dim]"
            )

    def _set_scene(self, v: str) -> None:
        self.scene = v
        self._rebuild_system()
        self.console.print(f"[green]✓ scene:[/green] {v}")

    def _set_mood(self, v: str) -> None:
        self.mood = v
        self._rebuild_system()
        self.console.print(f"[green]✓ mood:[/green] {v}")

    def _set_name(self, v: str) -> None:
        old = self.persona_name
        self.persona_name = v.capitalize()
        self._rebuild_system()
        self.console.print(f"[green]✓ name:[/green] {old} → {self.persona_name}")

    def _set_player(self, v: str) -> None:
        self.player_label = v.capitalize()
        self._rebuild_system()
        self.console.print(f"[green]✓ player:[/green] {self.player_label}")

    def _set_desc(self, v: str) -> None:
        self.persona_desc = v
        self._rebuild_system()
        self.console.print(f"[green]✓ description updated[/green]")

    # session helpers
    def _get_session_path(self, name: str) -> Path:
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        safe = re.sub(r"[^\w\-]", "_", name)
        return self.SESSIONS_DIR / f"{safe}.json"

    def save_session(self, name: Optional[str] = None) -> None:
        if not name:
            # prefer existing file with exact persona name
            exact = self.SESSIONS_DIR / f"{self.persona_name}.json"
            if exact.exists():
                name = self.persona_name
            else:
                existing = list(self.SESSIONS_DIR.glob(f"{self.persona_name}*.json"))
                name = (
                    existing[0].stem
                    if existing
                    else f"{self.persona_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
        path = self._get_session_path(name)
        payload = {
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
            "msg_count": self._msg_count,
            "saved_at": datetime.datetime.now().isoformat(),
            "version": 2,
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._unsaved_msgs = 0
            self.console.print(
                f"[bold green]✓ Saved[/bold green]  [dim]{path.name}[/dim]"
            )
        except OSError as e:
            self.console.print(f"[bold red]Save failed:[/bold red] {e}")

    def load_session(self, name: str) -> bool:
        path = self._get_session_path(name)
        if not path.exists():
            direct = Path(name)
            path = direct if direct.exists() else path
        if not path.exists():
            self.console.print(f"[bold red]Session not found:[/bold red] {name}")
            return False
        data = _safe_json_load(path)
        if data is None:
            self.console.print("[bold red]Load failed — file is corrupted.[/bold red]")
            return False
        self._apply_session_data(data)
        self.console.print(f"[bold green]✓ Loaded[/bold green]  [dim]{path.name}[/dim]")
        return True

    def _list_sessions(self) -> list[dict]:
        self.SESSIONS_DIR.mkdir(exist_ok=True)
        out = []
        for f in sorted(self.SESSIONS_DIR.glob("*.json")):
            d = _safe_json_load(f)
            if d is None:
                continue
            out.append(
                {
                    "name": f.stem,
                    "saved": d.get("saved_at", "?")[:16],
                    "persona": d.get("persona_name", "?"),
                    "scene": d.get("scene", "")[:30],
                    "msgs": d.get("msg_count", len(d.get("history", [])) // 2),
                    "path": f,
                }
            )
        return out

    def _show_sessions_table(self) -> None:
        sessions = self._list_sessions()
        if not sessions:
            self.console.print("[dim]No saved sessions.[/dim]")
            return
        t = Table(
            title="💾 Saved Sessions",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
        )
        t.add_column("#", style="dim", width=4)
        t.add_column("File", style="cyan", no_wrap=True)
        t.add_column("Character", style="magenta")
        t.add_column("Scene", style="dim", max_width=28)
        t.add_column("Msgs", style="dim", justify="right", width=6)
        t.add_column("Saved", style="dim", no_wrap=True)
        for i, s in enumerate(sessions, 1):
            t.add_row(
                str(i),
                s["name"],
                s["persona"],
                s["scene"] or "—",
                str(s["msgs"]),
                s["saved"],
            )
        self.console.print(t)

    def _load_session_by_path(self, path: Path) -> bool:
        data = _safe_json_load(path)
        if data is None:
            self.console.print("[bold red]Load failed — invalid JSON.[/bold red]")
            return False
        self._apply_session_data(data)
        self.console.print(
            f"\n[bold green]✓ Loaded[/bold green]  "
            f"[magenta]{data.get('persona_name','?')}[/magenta]"
            + (f"  [dim]scene: {self.scene}[/dim]" if self.scene else "")
        )
        return True

    # character editing
    def _edit_character_prompts(self) -> None:
        self.console.print(
            "\n[bold]Edit character  [dim](press Enter to keep current)[/dim][/bold]"
        )
        new_name = Prompt.ask(
            f"  [cyan]Name[/cyan] [dim]({self.persona_name})[/dim]",
            default=self.persona_name,
        ).strip()
        if new_name:
            self.persona_name = new_name.capitalize()

        new_desc = Prompt.ask(
            (
                f"  [cyan]Description[/cyan] [dim]({self.persona_desc[:40]}…)[/dim]"
                if len(self.persona_desc) > 40
                else f"  [cyan]Description[/cyan] [dim]({self.persona_desc})[/dim]"
            ),
            default=self.persona_desc,
        ).strip()
        if new_desc:
            self.persona_desc = new_desc

        new_scene = Prompt.ask(
            f"  [cyan]Scene[/cyan] [dim]({self.scene or 'none'})[/dim]",
            default=self.scene,
        ).strip()
        self.scene = new_scene

        new_mood = Prompt.ask(
            f"  [cyan]Mood[/cyan] [dim]({self.mood or 'none'})[/dim]",
            default=self.mood,
        ).strip()
        self.mood = new_mood

        self._rebuild_system()
        self.console.print("[green]✓ Character updated.[/green]\n")

    # display helpers
    def _show_memory(self) -> None:
        summary = self.memory.summary_line()
        self.console.print(
            Panel(
                self.memory.format_world(),
                title=f"📚 World Memory  [dim]({summary})[/dim]",
                border_style="blue",
            )
        )

    def _show_status(self) -> None:
        top = self._pool.top()
        rows = [
            ("Character", self.persona_name),
            ("Player", self.player_label),
            ("Model", self._pool.short_name()),
            ("Models", f"{self._pool.available()}/{self._pool.total()} available"),
            ("Scene", self.scene or "—"),
            ("Mood", self.mood or "—"),
            ("Messages", str(self._convo_msg_count())),
            ("World", self.memory.summary_line()),
            ("Condenses", str(self._condense_count)),
            (
                "Unsaved",
                f"{self._unsaved_msgs} msg{'s' if self._unsaved_msgs!=1 else ''}",
            ),
        ]
        body = "\n".join(f"[dim]{k}:[/dim]  {v}" for k, v in rows)
        self.console.print(Panel(body, title="📊 Status", border_style="cyan"))

    def _show_lore(self) -> None:
        self.console.print(
            Panel(self.lore, title="📜 Lore / Backstory", border_style="green")
        )

    def _show_help(self) -> None:
        self.console.print(
            Panel(
                "[bold green]CHAT[/bold green]\n"
                "  Just type to talk.\n"
                "  Wrap actions in *asterisks* — e.g. [italic]*waves hello*[/italic]\n\n"
                "[bold green]CHARACTER[/bold green]\n"
                "  [cyan]/set name[/cyan] <n>    rename the character\n"
                "  [cyan]/set player[/cyan] <n>  change your name\n"
                "  [cyan]/set scene[/cyan] <s>   change the current scene\n"
                "  [cyan]/set mood[/cyan] <m>    set emotional tone\n"
                "  [cyan]/set desc[/cyan] <d>    rewrite character description\n"
                "  [cyan]/edit[/cyan]            full guided character edit\n\n"
                "[bold green]CONVERSATION[/bold green]\n"
                "  [cyan]/retry[/cyan]    regenerate last response\n"
                "  [cyan]/undo[/cyan]     remove last exchange\n"
                "  [cyan]/clear[/cyan]    wipe conversation history\n\n"
                "[bold green]INFO[/bold green]\n"
                "  [cyan]/status[/cyan]   engine & model info\n"
                "  [cyan]/memory[/cyan]   world knowledge summary\n"
                "  [cyan]/lore[/cyan]     current backstory / lore\n\n"
                "[bold green]SESSIONS[/bold green]\n"
                "  [cyan]/save[/cyan] [name]   save current session\n"
                "  [cyan]/load[/cyan]          open session browser\n"
                "  [cyan]/sessions[/cyan]      list all sessions\n"
                "  [cyan]/new[/cyan]           start a new character\n\n"
                "[bold green]OTHER[/bold green]\n"
                "  [cyan]/debug[/cyan]   toggle debug output\n"
                "  [cyan]exit[/cyan]     quit (prompts to save)",
                title="❓ Help",
                border_style="yellow",
            )
        )

    # string utils
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

    # startup flow
    def _startup_flow(self) -> bool:
        try:
            sessions = self._list_sessions()

            if not sessions:
                self.console.print(
                    "\n[bold]No saved sessions — let's create a new character.[/bold]\n"
                )
            else:
                self.console.print()
                self._show_sessions_table()
                self.console.print(
                    "\n[dim]Enter a number or name to load, or press Enter for a new character.[/dim]"
                )
                choice = Prompt.ask("\n[bold green]>[/bold green]", default="").strip()

                if choice.lower() in ("exit", "quit", "q", "e"):
                    self.console.print("[bold green]Goodbye![/bold green]")
                    return False

                if choice:
                    # try numeric index
                    selected = None
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(sessions):
                            selected = sessions[idx]
                        else:
                            self.console.print(
                                f"[yellow]No session #{choice}.[/yellow]"
                            )
                            return False
                    else:
                        # fuzzy name match
                        cl = choice.lower()
                        for s in sessions:
                            if cl == s["name"].lower() or cl in s["name"].lower():
                                selected = s
                                break
                        if not selected:
                            self.console.print(
                                f"[yellow]Session not found: {choice}[/yellow]"
                            )
                            return False

                    if self._load_session_by_path(selected["path"]):
                        if Confirm.ask(
                            "[bold yellow]Edit character before continuing?[/bold yellow]",
                            default=False,
                        ):
                            self._edit_character_prompts()
                        return True
                    return False

            # new character wizard─
            self.console.print(
                "[bold cyan]New Character Setup[/bold cyan]  [dim](Ctrl+C to cancel)[/dim]\n"
            )

            while True:
                name_in = Prompt.ask("[bold green]Character name[/bold green]").strip()
                if not name_in:
                    self.console.print("[yellow]A name is required.[/yellow]")
                    continue
                self.persona_name = name_in.capitalize()
                break

            while True:
                desc_in = Prompt.ask(
                    "[bold green]Character description[/bold green]\n"
                    "[dim]  e.g. a cynical detective with a soft spot for strays[/dim]"
                ).strip()
                if not desc_in:
                    self.console.print("[yellow]A description is required.[/yellow]")
                    continue
                self.persona_desc = desc_in
                break

            player_in = Prompt.ask(
                "[bold green]Your name[/bold green] [dim](how the character addresses you — skip to use 'You')[/dim]",
                default="",
            ).strip()
            if player_in:
                self.player_label = player_in.capitalize()

            scene_in = Prompt.ask(
                "[bold green]Opening scene[/bold green] [dim](optional — where does the story begin?)[/dim]",
                default="",
            ).strip()
            if scene_in:
                self.scene = scene_in
                self.lore = f"The story begins: {scene_in}"

            self.memory.add_character(
                self.persona_name,
                description=self.persona_desc,
                context="persona definition",
            )
            self.history.append(
                {
                    "role": "system",
                    "content": self._build_system_content(self.persona_desc, self.lore),
                }
            )
            self.console.print(
                f"\n[bold green]Ready![/bold green]  "
                f"[magenta]{self.persona_name}[/magenta] awaits.\n"
            )
            return True

        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[bold green]Goodbye![/bold green]")
            return False

    # chat loop
    def _chat_loop(self) -> None:
        top_short = self._pool.short_name()
        header = (
            f"[bold green]✓[/bold green]  "
            f"[magenta bold]{self.persona_name}[/magenta bold]  "
            f"[dim]{self.player_label}  •  {top_short}"
            + (f"  •  {self.scene}" if self.scene else "")
            + "[/dim]"
        )
        self.console.print(header)
        self.console.print(
            "[dim]Type to chat  •  /help for commands  •  exit to quit[/dim]\n"
        )

        while True:
            try:
                # show unsaved warning inline with prompt
                prompt_label = f"[bold cyan]{self.player_label}[/bold cyan]"
                if self._unsaved_msgs >= 15:
                    prompt_label += " [dim yellow](unsaved)[/dim yellow]"
                user_input = Prompt.ask(prompt_label).strip().strip("\"'")
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[bold green]Goodbye![/bold green]")
                break

            if not user_input:
                continue

            cmd = user_input.lower().strip()

            # exit
            if cmd in ("exit", "/exit", "quit", "/quit", "q", "/q", "e", "/e"):
                if self._unsaved_msgs > 0:
                    if Confirm.ask(f"[yellow]Save before exit?[/yellow]", default=True):
                        self.save_session()
                self.console.print("[bold green]Goodbye![/bold green]")
                break

            # commands─
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
                self._show_lore()
                continue
            if cmd == "/edit":
                self._edit_character_prompts()
                continue
            if cmd == "/debug":
                self.debug = not self.debug
                self.console.print(
                    f"[green]✓ Debug {'ON' if self.debug else 'OFF'}[/green]"
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
                        self.console.print(
                            "[dim red]Retry failed — try again.[/dim red]"
                        )
                else:
                    self.console.print("[dim]Nothing to retry.[/dim]")
                continue

            if cmd == "/undo":
                # remove last user+assistant pair
                removed = 0
                while (
                    self.history
                    and self.history[-1]["role"] in ("assistant", "user")
                    and removed < 2
                ):
                    self.history.pop()
                    removed += 1
                if removed:
                    self._unsaved_msgs = max(0, self._unsaved_msgs - 1)
                    self.console.print(
                        f"[green]✓ Removed last {'exchange' if removed==2 else 'message'}.[/green]"
                    )
                else:
                    self.console.print("[dim]Nothing to undo.[/dim]")
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
                if Confirm.ask(
                    "[yellow]Clear conversation history?[/yellow]", default=False
                ):
                    self.history = [self.history[0]] if self.history else []
                    self._condense_count = 0
                    self._recent_assistant_excerpts = []
                    self._unsaved_msgs = 0
                    self.console.print("[green]✓ History cleared.[/green]")
                continue

            # unknown command
            if cmd.startswith("/"):
                self.console.print(
                    f"[yellow]Unknown command: {cmd}[/yellow]  [dim](/help for list)[/dim]"
                )
                continue

            # chat turn─
            self._extract_info_from_message(user_input, is_user=True)
            clean_speech, directives = self._parse_directives(user_input)

            labeled = (
                f"{self.player_label}: {clean_speech}\n"
                f"[ACTION FOR {self.persona_name.upper()}]: {directives}"
                if directives
                else f"{self.player_label}: {user_input}"
            )

            self.history.append({"role": "user", "content": labeled})
            self._check_and_condense()

            self.console.print(Rule(style="dim"))
            self.console.print(f"[bold magenta]{self.persona_name}[/bold magenta]")

            content = self._strip_name_prefix(self._get_reply())

            if not content or len(content) < 5:
                self.console.print(
                    "[dim red]No response received — try rephrasing or /retry.[/dim red]"
                )
                self.history.pop()  # remove the un-answered user message
                continue

            self._extract_info_from_message(content, is_user=False)
            self._update_system_memory()
            self._last_assistant_content = content
            self._track_excerpt(content)
            self.history.append(
                {
                    "role": "assistant",
                    "content": f"{self.persona_name}: {content}",
                }
            )
            self._msg_count += 1
            self._unsaved_msgs += 1

            # periodic auto-save reminder
            if self._unsaved_msgs > 0 and self._msg_count % 20 == 0:
                self.console.print(
                    "[dim yellow]💾  Tip: type /save to keep your progress.[/dim yellow]"
                )

    # entry point
    def run(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")
        self.console.clear()
        self.console.print(
            Panel(
                "[bold white]★  ChatME ROLEPLAY ENGINE  ★[/bold white]\n"
                "[dim]Immersive AI roleplay  •  /help to get started[/dim]",
                style="bold blue",
                expand=False,
            )
        )
        while self._startup_flow():
            self._chat_loop()
