from __future__ import annotations

from typing import Optional

from constants import LOCATION_STOPWORDS, is_valid_name, normalise_name

# display/trim limits
_MAX_EVENTS = 8
_MAX_NOTES = 5
_MAX_LOCS = 30
_CONTEXT_TRIM = 120
_MAX_CHARS = 50  # hard cap on tracked characters


class Character:
    """A single named entity in the world."""

    __slots__ = (
        "name",
        "age",
        "description",
        "personality",
        "first_appearance",
        "mentions",
        "last_seen",
    )

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.age: Optional[str] = None
        self.description: str = ""
        self.personality: list[str] = []
        self.first_appearance: str = ""
        self.mentions: int = 0
        self.last_seen: str = ""

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "description": self.description,
            "personality": self.personality,
            "first_appearance": self.first_appearance,
            "mentions": self.mentions,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Character:
        c = cls(data.get("name", ""))
        c.age = data.get("age")
        c.description = data.get("description", "")
        c.personality = data.get("personality", [])
        c.first_appearance = data.get("first_appearance", "")
        c.mentions = data.get("mentions", 0)
        c.last_seen = data.get("last_seen", "")
        return c

    # display
    def format_info(self) -> str:
        parts = [f"**{self.name}**"]
        if self.age:
            parts.append(f"age {self.age}")
        if self.description:
            desc = self.description[:80] + ("…" if len(self.description) > 80 else "")
            parts.append(desc)
        if self.personality:
            parts.append(f"[{', '.join(self.personality[:3])}]")
        parts.append(f"×{self.mentions}")
        return " | ".join(parts)

    # mutation
    def update(
        self,
        age: Optional[str] = None,
        description: str = "",
        context: str = "",
    ) -> None:
        """Increment mention counter; fill missing fields non-destructively."""
        self.mentions += 1
        if age and not self.age:
            self.age = age
        if description and not self.description:
            self.description = description[:_CONTEXT_TRIM]
        if context:
            self.last_seen = context[:_CONTEXT_TRIM]

    def add_trait(self, trait: str) -> None:
        t = trait.lower().strip()
        if t and t not in self.personality and len(self.personality) < 6:
            self.personality.append(t)


class Relationship:
    """A directed (but bidirectionally matched) link between two characters."""

    __slots__ = ("from_char", "to_char", "rel_type", "description", "first_appearance")

    def __init__(
        self,
        from_char: str,
        to_char: str,
        rel_type: str,
        description: str = "",
    ) -> None:
        self.from_char: str = from_char
        self.to_char: str = to_char
        self.rel_type: str = rel_type
        self.description: str = description
        self.first_appearance: str = ""

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "from": self.from_char,
            "to": self.to_char,
            "type": self.rel_type,
            "description": self.description,
            "first_appearance": self.first_appearance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Relationship:
        r = cls(
            data.get("from", ""),
            data.get("to", ""),
            data.get("type", "related"),
            data.get("description", ""),
        )
        r.first_appearance = data.get("first_appearance", "")
        return r

    # helpers
    def matches(self, a: str, b: str) -> bool:
        """Bidirectional pair equality."""
        return (self.from_char == a and self.to_char == b) or (
            self.from_char == b and self.to_char == a
        )

    def format_info(self) -> str:
        desc = f" — {self.description[:60]}" if self.description else ""
        return f"**{self.from_char}** ↔ **{self.to_char}**: {self.rel_type}{desc}"


class WorldMemory:
    """Persistent world-state: characters, relationships, locations, events."""

    def __init__(self) -> None:
        self.characters: dict[str, Character] = {}
        self.relationships: list[Relationship] = []
        self.locations: list[str] = []
        self.events: list[str] = []
        self.custom_notes: list[str] = []

    # character ops
    def add_character(
        self,
        name: str,
        age: Optional[str] = None,
        description: str = "",
        context: str = "",
    ) -> Optional[Character]:
        key = normalise_name(name)
        if not is_valid_name(key):
            return None
        if len(self.characters) >= _MAX_CHARS and key not in self.characters:
            return None  # world is full — don't leak memory
        if key in self.characters:
            self.characters[key].update(
                age=age, description=description, context=context
            )
        else:
            c = Character(key)
            c.age = age
            c.description = description[:_CONTEXT_TRIM] if description else ""
            c.first_appearance = context[:_CONTEXT_TRIM]
            c.last_seen = context[:_CONTEXT_TRIM]
            c.mentions = 1
            self.characters[key] = c
        return self.characters[key]

    def get_character(self, name: str) -> Optional[Character]:
        return self.characters.get(normalise_name(name))

    def remove_character(self, name: str) -> bool:
        key = normalise_name(name)
        if key in self.characters:
            del self.characters[key]
            self.relationships = [
                r for r in self.relationships if r.from_char != key and r.to_char != key
            ]
            return True
        return False

    # relationship ops
    def add_relationship(
        self,
        from_char: str,
        to_char: str,
        rel_type: str,
        description: str = "",
        context: str = "",
    ) -> Optional[Relationship]:
        fc = normalise_name(from_char)
        tc = normalise_name(to_char)

        if not fc or not tc or fc == tc:
            return None
        if not is_valid_name(fc) or not is_valid_name(tc):
            return None

        for rel in self.relationships:
            if rel.matches(fc, tc):
                # update in-place rather than duplicating
                rel.rel_type = rel_type
                if description:
                    rel.description = description
                return rel

        r = Relationship(fc, tc, rel_type, description)
        r.first_appearance = context[:_CONTEXT_TRIM]
        self.relationships.append(r)
        return r

    # location ops
    def add_location(self, location: str) -> bool:
        loc = location.strip().rstrip(".,!?;:")
        if (
            loc
            and len(loc) > 2
            and len(self.locations) < _MAX_LOCS
            and loc.lower() not in LOCATION_STOPWORDS
            and loc not in self.locations
        ):
            self.locations.append(loc)
            return True
        return False

    # event/note ops─
    def add_event(self, event: str) -> None:
        ev = event.strip()
        if ev and ev not in self.events:
            self.events.append(ev)
            if len(self.events) > _MAX_EVENTS * 2:
                self.events = self.events[-_MAX_EVENTS:]

    def add_note(self, note: str) -> None:
        n = note.strip()
        if n and n not in self.custom_notes:
            self.custom_notes.append(n)
            if len(self.custom_notes) > _MAX_NOTES * 2:
                self.custom_notes = self.custom_notes[-_MAX_NOTES:]

    # formatting─
    def format_world(self) -> str:
        parts: list[str] = []

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
            recent = self.events[-_MAX_EVENTS:]
            parts.append(f"**Events:** {' → '.join(recent)}")

        if self.custom_notes:
            parts.append(
                "**Notes:**\n"
                + "\n".join(f"  • {n}" for n in self.custom_notes[-_MAX_NOTES:])
            )

        return "\n\n".join(parts) if parts else "No world info yet."

    def summary_line(self) -> str:
        """One-line summary for status display."""
        c = len(self.characters)
        r = len(self.relationships)
        l = len(self.locations)
        return f"{c} char{'s' if c!=1 else ''}, {r} rel{'s' if r!=1 else ''}, {l} loc{'s' if l!=1 else ''}"

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "locations": self.locations,
            "events": self.events,
            "custom_notes": self.custom_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorldMemory:
        m = cls()
        for k, v in data.get("characters", {}).items():
            m.characters[k] = Character.from_dict(v)
        m.relationships = [
            Relationship.from_dict(r) for r in data.get("relationships", [])
        ]
        m.locations = data.get("locations", [])
        m.events = data.get("events", [])
        m.custom_notes = data.get("custom_notes", [])
        return m
