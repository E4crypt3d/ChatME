from typing import Optional

from constants import (
    RELATION_WORDS,
    LOCATION_STOPWORDS,
    is_valid_name,
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
        if not is_valid_name(key):
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
        if not is_valid_name(fc) or not is_valid_name(tc):
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
