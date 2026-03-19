# ChatME Roleplay Engine

A terminal-based roleplay engine using free OpenRouter models with automatic memory consolidation.

## Features

- **Free Models Router** - Uses OpenRouter's free models router (`openrouter/free`) by default for automatic model selection
- **Memory consolidation** - Summarizes conversation history to stay within context limits
- **Character memory** - Tracks characters and relationships mentioned in conversation
- **Rich terminal UI** - Beautiful colored output with Rich library
- **Character persistence** - Remembers story context across sessions
- **Interactive session management** - Easy load/save with visual selection menu
- **CLI options** - Specify model and API key from command line or environment variables
- **Inline character control** - Direct the AI character's actions using *asterisks*

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

| Option    | Short | Description                      |
| --------- | ----- | -------------------------------- |
| `--model` | `-m`  | Specify a model to use           |
| `--key`   | `-k`  | Provide API key for this session |

Examples:
```bash
# Use default free models router
python main.py

# Use a specific model
python main.py -m deepseek/deepseek-r1:free

# Use a custom API key
python main.py --key sk-xxxxxxx

# Both options together
python main.py -m "deepseek/deepseek-r1:free" -k "sk-xxxx"
```

## First Run

When you first run `python main.py`:

1. **Choose an option:**
   - Press **ENTER** to create a new character
   - Type a **number** to load a saved session
   - Type a **name** to search for a session
   - Type **exit** or **quit** to exit

2. If creating new, enter:
   - Character name (the AI-controlled character)
   - Character description (personality, appearance, backstory)
   - Your name (optional - this is you, the player)
   - Opening scene (optional - sets the initial story context)

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

## Inline Character Control

You can directly control the AI character's actions, thoughts, and emotions using **asterisks** in your messages.

### Syntax

```
*YourCharacter does something*
*YourCharacter thinks about something*
*YourCharacter feels something*
```

### Examples

```
Hey Karen, want to swim? *Karen looks at you and thinks how hot he is*
```

```
*Karen smiles warmly* It's so nice to see you!
```

```
*Karen blushes* I... I didn't expect that...
```

The text inside asterisks is extracted and sent as a clear directive to the AI, ensuring your character reacts exactly as you want.

## Commands

| Command              | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `/help`              | Show help screen with all commands                           |
| `/memory` or `/mem`  | View tracked characters and relationships                    |
| `/status`            | Show current character, player, model, scene, mood           |
| `/lore`              | Show story history/lore                                      |
| `/retry`             | Regenerate last AI response                                  |
| `/clear`             | Reset conversation history (asks for confirmation)           |
| `/debug`             | Toggle debug output                                          |
| `/set scene <desc>`  | Set current scene context                                    |
| `/set mood <desc>`   | Set character's mood                                         |
| `/set name <name>`   | Rename the character                                         |
| `/set player <name>` | Rename the player                                            |
| `/save [name]`       | Save session (overwrites previous save for same character)   |
| `/load` or `/new`    | Load saved session or create new (interactive)               |
| `/sessions`          | List all saved sessions                                      |
| `exit`, `/exit`, `q` | Quit (asks to save first)                                    |

## Session Management

### Saving Sessions
- Type `/save` to save - automatically overwrites previous save for the same character
- Type `/save my_adventure` to save with custom name (creates new file)

### Loading Sessions
- Type `/load` to see saved sessions and choose one
- Works with:
  - Session number (e.g., "1", "2")
  - Session name (partial match, e.g., "hannah")
- After loading, you can edit the character name/description

## Configuration

Configure via environment variables or CLI arguments:

| Environment Variable | Description                  |
| -------------------- | ---------------------------- |
| `OPENROUTER_API_KEY` | Your OpenRouter API key      |
| `MODEL`             | Default model (default: `openrouter/free`) |
| `DEBUG`             | Set to "true" for debug logs |

Examples:
```bash
# Set default model via environment
export MODEL=deepseek/deepseek-r1:free
python main.py

# Or in .env file
MODEL=arcee-ai/trinity-large-preview:free
```

## Troubleshooting

- **Rate limiting**: Some models may be temporarily unavailable - the engine will automatically try other models when using `openrouter/free`
- **Empty responses**: Use `/retry` to regenerate
- **Debug mode**: Set `DEBUG=true` in `.env` for detailed logs
- **API key issues**: Use `--key` option to provide a key for a single session
- **Character confusion**: Use the inline control feature (`*character does X*`) to explicitly direct the AI character's actions
