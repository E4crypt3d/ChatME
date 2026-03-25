# ChatME Roleplay Engine

A terminal-based AI roleplay engine with multi-provider support and automatic memory consolidation.

## Features

- **Multi-Provider Support** - OpenRouter and Groq API support
- **Memory consolidation** - Summarizes conversation history to stay within context limits
- **Character memory** - Tracks characters and relationships mentioned in conversation
- **Rich terminal UI** - Beautiful colored output with Rich library
- **Character persistence** - Remembers story context across sessions
- **Interactive session management** - Easy load/save with visual selection menu
- **Inline character control** - Direct the AI character's actions using *asterisks*

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Configure environment:
- Copy `.env.example` to `.env`
- Choose a provider (openrouter or groq)
- Add your API key:
```
provider=openrouter
openrouter_api_key=your_key_here
# or for Groq:
provider=groq
groq_api_key=your_key_here
```

3. Run the engine:
```
python main.py
```

## Providers

| Provider   | Default Model                          | API Key Env        |
| ---------- | -------------------------------------- | ------------------ |
| openrouter | openrouter/free                        | OPENROUTER_API_KEY |
| groq       | groq/compound-mini (70K TPM, fastest!) | GROQ_API_KEY       |

## Command Line Options

```
python main.py -p <provider> -m <model> -k <api_key>
```

| Option        | Short | Description                  |
| ------------- | ----- | ---------------------------- |
| `--provider`  | `-p`  | Provider: openrouter or groq |
| `--model`     | `-m`  | Model ID to use              |
| `--key`       | `-k`  | API key for this session     |
| `--showmodel` | `-sm` | List available free models   |

Examples:
```bash
# Default (OpenRouter)
python main.py

# Use Groq provider
python main.py -p groq

# Use specific model
python main.py -m meta-llama/llama-3.1-8b-instant

# Show available free models
python main.py --showmodel
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

| Command              | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| `/help`              | Show help screen with all commands                         |
| `/memory` or `/mem`  | View tracked characters and relationships                  |
| `/status`            | Show current character, player, model, scene, mood         |
| `/lore`              | Show story history/lore                                    |
| `/retry`             | Regenerate last AI response                                |
| `/clear`             | Reset conversation history (asks for confirmation)         |
| `/debug`             | Toggle debug output                                        |
| `/set scene <desc>`  | Set current scene context                                  |
| `/set mood <desc>`   | Set character's mood                                       |
| `/set name <name>`   | Rename the character                                       |
| `/set player <name>` | Rename the player                                          |
| `/save [name]`       | Save session (overwrites previous save for same character) |
| `/load` or `/new`    | Load saved session or create new (interactive)             |
| `/sessions`          | List all saved sessions                                    |
| `exit`, `/exit`, `q` | Quit (asks to save first)                                  |

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

| Environment Variable | Description                              |
| -------------------- | ---------------------------------------- |
| `provider`           | openrouter or groq (default: openrouter) |
| `OPENROUTER_API_KEY` | Your OpenRouter API key                  |
| `GROQ_API_KEY`       | Your Groq API key                        |
| `MODEL`              | Default model                            |
| `DEBUG`              | Set to "true" for debug logs             |

Examples:
```bash
# Use Groq via environment
export provider=groq
export GROQ_API_KEY=your_key
python main.py
```

## Troubleshooting

- **Rate limiting**: Some models may be temporarily unavailable - try switching providers
- **Empty responses**: Use `/retry` to regenerate
- **Debug mode**: Set `DEBUG=true` in `.env` for detailed logs
- **API key issues**: Use `-k` option to provide a key for a single session
- **Character confusion**: Use the inline control feature (`*character does X*`) to explicitly direct the AI character's actions
