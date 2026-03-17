# ChatME Roleplay Engine

A terminal-based roleplay engine using free OpenRouter models with automatic memory consolidation.

## Features

- **Multi-model failover** - Automatically switches to backup models if one fails
- **Memory consolidation** - Summarizes conversation history to stay within context limits
- **Character memory** - Tracks characters and relationships mentioned in conversation
- **Rich terminal UI** - Beautiful colored output with Rich library
- **Character persistence** - Remembers story context across sessions

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Configure environment:
- Copy `.env.example` to `.env`
- Add your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

3. Run the engine:
```
python main.py
```

## First Run

When you first run `python main.py`:

1. Enter your character when prompted (format: "Name, description")
2. Enter your name (the player character)
3. Optionally enter an opening scene
4. Start roleplaying!

### Character Format Tips

The best results come from describing your character with **name, relationship, and personality**:

```
Kelly, your wife, a celebrity from Hollywood who's protective and jealous
```

```
John, my husband, a 35-year-old businessman who works too much
```

```
Merlin, the ancient wizard who mentors me in magic
```

**Key tips:**
- Start with the **character name** (e.g., "Kelly")
- Add **relationship to you** (e.g., "your wife", "my husband", "my mentor")
- Include **personality traits** (e.g., "protective", "jealous", "wise")
- Keep descriptions **2-3 sentences** for best results

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help screen |
| `/memory` | View tracked characters and relationships |
| `/status` | Show current scene, mood, and settings |
| `/lore` | Show story lore |
| `/narrate` | Toggle narration mode (describes scene + actions + dialogue) |
| `/retry` | Regenerate last response |
| `/set scene <desc>` | Set current scene context |
| `/set mood <desc>` | Set character's mood |
| `/set name <name>` | Rename the character mid-session |
| `/set player <name>` | Rename the player label |
| `/save [name]` | Save current session |
| `/load <name>` | Load a saved session |
| `/sessions` | List all saved sessions |
| `/clear` | Reset conversation history (keeps persona) |
| `/debug` | Toggle debug output |
| `exit` or `/exit` | Quit the session |

## Configuration

Edit the settings in `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CONDENSE_THRESHOLD` | 22 | Messages before triggering condensation |
| `KEEP_RECENT_TURNS` | 10 | Recent turns to keep after condensation |
| `REPLY_MAX_TOKENS` | 400 | Max tokens per response |
| `DEBUG` | false | Enable debug output |

## Troubleshooting

- **Rate limiting**: Some models may be temporarily unavailable
- **Empty responses**: Try a different model from the list
- **Debug mode**: Set `DEBUG=true` in `.env` for detailed logs
