# Roleplay Features Enhancement Plan

## Current State
The current system uses a simple "lore" block that gets summarized. We need more structured memory for roleplay.

## Feature Ideas

### 1. Character Registry
- Track characters mentioned in conversation
- Store: name, age, description, personality traits
- Quick lookup during roleplay

### 2. Relationship Tracking
- Track relationships between characters
- Types: friend, enemy, family, romantic, etc.
- Track relationship changes over time

### 3. World/Setting Memory
- Locations mentioned
- Events that happened
- World rules/laws

### 4. Enhanced Lore System
- Multiple lore categories:
  - Character bios
  - Relationships
  - Plot events
  - World info

### 5. Smart Memory Extraction
- Auto-detect character names, ages from dialogue
- Extract relationship hints ("my brother", "his friend")
- Note important plot points

---

## Implementation Approach

### Option A: Structured JSON Lore
```python
lore = {
    "characters": [
        {"name": "John", "age": 25, "description": "tall warrior"}
    ],
    "relationships": [
        {"from": "John", "to": "Mary", "type": "friend"}
    ],
    "locations": ["castle", "forest"],
    "events": []
}
```

### Option B: Enhanced System Prompt
Keep lore as text but with better structure in the prompt.

### Option C: Hybrid
Use both - structured data + natural language lore

---

## Recommended: Enhanced System Prompt (Option B)
Easier to implement, works well with AI summarization.

**Changes:**
1. Add structured lore template to system prompt
2. Add memory extraction after each response
3. Update lore with new character/relationship info

---

## Features to Add

1. **Character Database** - Track all characters with attributes
2. **Relationship Map** - Track who knows whom
3. **Event Log** - Important plot events
4. **Auto-Extraction** - Detect and save new info from messages
5. **Lore Display** - Command to show character/relationship info
