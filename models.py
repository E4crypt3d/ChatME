from __future__ import annotations
from typing import Optional

from constants import LOCATION_STOPWORDS, is_valid_name

# max recent items shown
_MAX_EVENTS = 5
_MAX_NOTES = 3
_MAX_LOCS = 20
_CONTEXT_TRIM = 100


class Character:
    __slots__ = (
        "name",
        "age",
        "description",
        "personality",
        "first_appearance",
        "mentions",
    )

    def __init__(self, name: str) -> None:
        self.name: str = name
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
    def from_dict(cls, data: dict) -> Character:
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

    def update(self, age: Optional[str] = None, description: str = "") -> None:
        # fill missing fields only
        self.mentions += 1
        if age and not self.age:
            self.age = age
        if description and not self.description:
            self.description = description


class Relationship:
    __slots__ = ("from_char", "to_char", "rel_type", "description", "first_appearance")

    def __init__(
        self, from_char: str, to_char: str, rel_type: str, description: str = ""
    ) -> None:
        self.from_char: str = from_char
        self.to_char: str = to_char
        self.rel_type: str = rel_type
        self.description: str = description
        self.first_appearance: str = ""

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

    def matches(self, a: str, b: str) -> bool:
        # bidirectional pair check
        return (self.from_char == a and self.to_char == b) or (
            self.from_char == b and self.to_char == a
        )

    def format_info(self) -> str:
        desc = f" — {self.description}" if self.description else ""
        return f"**{self.from_char}** ↔ **{self.to_char}**: {self.rel_type}{desc}"


class WorldMemory:
    def __init__(self) -> None:
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
        if not is_valid_name(key):
            return None
        if key in self.characters:
            self.characters[key].update(age=age, description=description)
        else:
            c = Character(key)
            c.age = age
            c.description = description
            c.first_appearance = context[:_CONTEXT_TRIM]
            c.mentions = 1
            self.characters[key] = c
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
        if not is_valid_name(fc) or not is_valid_name(tc):
            return None

        for rel in self.relationships:
            if rel.matches(fc, tc):
                rel.rel_type = rel_type
                if description:
                    rel.description = description
                return rel

        r = Relationship(fc, tc, rel_type, description)
        r.first_appearance = context[:_CONTEXT_TRIM]
        self.relationships.append(r)
        return r

    def add_location(self, location: str) -> None:
        loc = location.strip().rstrip(".,!?;:")
        if (
            loc
            and len(loc) > 2
            and len(self.locations) < _MAX_LOCS
            and loc.lower() not in LOCATION_STOPWORDS
            and loc not in self.locations
        ):
            self.locations.append(loc)

    def add_event(self, event: str) -> None:
        if event and event not in self.events:
            self.events.append(event)

    def get_character(self, name: str) -> Optional[Character]:
        return self.characters.get(name.strip().title())

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
            parts.append(f"**Events:** {', '.join(self.events[-_MAX_EVENTS:])}")
        if self.custom_notes:
            parts.append(f"**Notes:** {', '.join(self.custom_notes[-_MAX_NOTES:])}")

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
    def from_dict(cls, data: dict) -> WorldMemory:
        m = cls()
        m.characters = {
            k: Character.from_dict(v) for k, v in data.get("characters", {}).items()
        }
        m.relationships = [
            Relationship.from_dict(r) for r in data.get("relationships", [])
        ]
        m.locations = data.get("locations", [])
        m.events = data.get("events", [])
        m.custom_notes = data.get("custom_notes", [])
        return m
