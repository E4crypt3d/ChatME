# ChatME Roleplay Engine

A terminal-based roleplay engine using free OpenRouter models with automatic memory consolidation.

## Features

- **Multi-model failover** - Automatically switches to backup models if one fails
- **Memory consolidation** - Summarizes conversation history to stay within context limits
- **Character memory** - Tracks characters and relationships mentioned in conversation
- **Rich terminal UI** - Beautiful colored output with Rich library
- **Character persistence** - Remembers story context across sessions
- **Interactive session management** - Easy load/save with visual selection menu
- **CLI options** - Specify model and API key from command line

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

## Command Line Options

```
python main.py --model <model_name> --key <api_key>
```

| Option | Short | Description |
|--------|-------|-------------|
| `--model` | `-m` | Specify a model to use |
| `--key` | `-k` | Provide API key for this session |

Examples:
```bash
# Use a specific model
python main.py -m cognitivecomputations/dolphin-mistral-24b-venice-edition:free

# Use a custom API key
python main.py --key sk-xxxxxxx

# Both options together
python main.py -m "openchat/openchat-7b" -k "sk-xxxx"
```

## First Run

When you first run `python main.py`:

1. **Choose an option:**
   - Press **ENTER** to create a new character
   - Type a **number** to load a saved session
   - Type a **name** to search for a session

2. If creating new, enter:
   - Character name
   - Character description (personality, appearance, backstory)
   - Your name (optional)
   - Opening scene (optional)

3. Start roleplaying!

### Character Description Tips

The best results come from describing your character with **personality and backstory**:

```
A wise old wizard who protects the kingdom, stern but caring
```

```
A mysterious thief from the streets, quick-witted and sneaky
```

**Key tips:**
- Include **personality traits** (e.g., "wise", "sneaky", "friendly")
- Add **appearance hints** if desired (e.g., "wears a red cloak")
- Keep descriptions **2-3 sentences** for best results

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help screen with all commands |
| `/memory` or `/mem` | View tracked characters and relationships |
| `/status` | Show current character, player, model, scene, mood |
| `/lore` | Show story history/lore |
| `/narrate` | Toggle narration mode (describes scene + actions + dialogue) |
| `/retry` | Regenerate last AI response |
| `/clear` | Reset conversation history (asks for confirmation) |
| `/debug` | Toggle debug output |
| `/set scene <desc>` | Set current scene context |
| `/set mood <desc>` | Set character's mood |
| `/set name <name>` | Rename the character |
| `/set player <name>` | Rename the player |
| `/save [name]` | Save session (optional custom name) |
| `/load` or `/new` | Load saved session or create new (interactive) |
| `/sessions` | List all saved sessions |
| `exit` or `/exit` | Quit (asks to save first) |

## Session Management

### Saving Sessions
- Type `/save` to save with automatic name (e.g., "Hannah_20240315_143022")
- Type `/save my_adventure` to save with custom name

### Loading Sessions
- Type `/load` to see saved sessions and choose one
- Works with:
  - Session number (e.g., "1", "2")
  - Session name (partial match, e.g., "hannah")
- After loading, you can edit the character name/description

## Configuration

The app now uses CLI arguments. For environment configuration:

| Environment Variable | Description |
|---------------------|-------------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key |
| `DEBUG` | Set to "true" for debug logs |

## Troubleshooting

- **Rate limiting**: Some models may be temporarily unavailable - the engine will automatically try other models
- **Empty responses**: Try `/narrate` mode or use `/retry` to regenerate
- **Debug mode**: Set `DEBUG=true` in `.env` for detailed logs
- **API key issues**: Use `--key` option to provide a key for a single session
