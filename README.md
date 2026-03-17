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

## Configuration

Edit the settings in `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CONDENSE_THRESHOLD` | 22 | Messages before triggering condensation |
| `KEEP_RECENT_TURNS` | 10 | Recent turns to keep after condensation |
| `REPLY_MAX_TOKENS` | 400 | Max tokens per response |
| `DEBUG` | false | Enable debug output |


## Usage

1. Run `python main.py`
2. Enter your character description when prompted
3. Type your messages to roleplay
4. Type `exit` or `quit` to end
5. Use `/memory` to view tracked characters and relationships

## How to Enter Your Character

When prompted "Who am I roleplaying as?", enter your character in this format:

```
Name, description
```

### Best Format Examples:

**Simple:**
```
Wizard, an old mage with a long white beard
```

**Detailed:**
```
Arthur, a noble knight of the Round Table, brave and honorable
```

**Fantasy:**
```
Merlin, the most powerful wizard in the realm, ancient and wise
```

**Modern:**
```
Detective Cole, a gritty noir detective in 1940s LA
```

The first line sets your character's name and description. The AI will use this to roleplay as your character.

## Commands

- `/memory` - View tracked characters and relationships
- `exit` or `/exit` - Quit the session

## Troubleshooting

- **Rate limiting**: Some models may be temporarily unavailable
- **Empty responses**: Try a different model from the list
- **Debug mode**: Set `DEBUG=true` in `.env` for detailed logs
