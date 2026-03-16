# ChatME Roleplay Engine

A terminal-based roleplay engine using free OpenRouter models with automatic memory consolidation.

## Features

- **Multi-model failover** - Automatically switches to backup models if one fails
- **Memory consolidation** - Summarizes conversation history to stay within context limits
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

## Troubleshooting

- **Rate limiting**: Some models may be temporarily unavailable
- **Empty responses**: Try a different model from the list
- **Debug mode**: Set `DEBUG=true` in `.env` for detailed logs
