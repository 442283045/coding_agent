# Coding Agent

AI-powered CLI coding assistant built with Python and uv.

## Features

- 🤖 Multi-model LLM support (OpenAI, Anthropic, and more via LiteLLM)
- 🛠️ Extensible tool system with MCP compatibility
- 🔍 Code search and analysis with Tree-sitter
- 📝 File read/write with safety checks
- 🔒 Secure shell command execution
- 💾 Conversation history management

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone <repo-url>
cd coding_agent

# Install dependencies
uv pip install -e ".[dev]"

# Or sync with lock file
uv sync
```

### Using pip

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MOONSHOT_API_KEY=your_moonshot_key
AGENT_DEFAULT_MODEL=moonshot/kimi-k2.5
```

## Usage

### Interactive mode

```bash
# Start interactive session
coding-agent

# Equivalent explicit interactive command
coding-agent chat

# Start in specific directory
coding-agent chat /path/to/project

# Use specific model
coding-agent chat -m gpt-4o

# Print detailed LLM request/response logs
coding-agent chat --debug
```

### Single command mode

```bash
coding-agent run "Explain the codebase structure"
```

### Available commands

```bash
coding-agent --help
coding-agent chat --help
coding-agent run --help
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for a repository-specific maturity roadmap and milestone plan.

## Project Structure

```
coding_agent/
├── cli.py              # CLI entry point
├── config.py           # Configuration management
├── agent/              # Agent core logic
│   ├── core.py         # Main agent loop
│   ├── state.py        # Session state
│   └── history.py      # Conversation history
├── llm/                # LLM clients
│   ├── client.py       # Unified LLM interface
│   └── tokenizer.py    # Token counting
├── tools/              # Tools registry
│   ├── registry.py     # Tool registration
│   ├── file_tools.py   # File operations
│   ├── shell_tools.py  # Shell execution
│   └── code_tools.py   # Code analysis
└── prompts/            # System prompts
    └── system.j2       # Main system prompt
```

## Development

```bash
# Run tests
pytest

# Run linting
ruff check .
ruff format .

# Type checking
mypy coding_agent
```

## License

MIT
