# Coding Agent

AI-powered CLI coding assistant built with Python and uv.

## Features

- 🤖 Multi-model LLM support (OpenAI, Anthropic, and more via LiteLLM)
- 🛠️ Extensible tool system with built-in tools plus FastMCP-backed MCP servers
- 🧠 Workspace skills discovery for project-local `SKILL.md` workflows
- 🔍 Code search and analysis with Tree-sitter
- 📝 File read/write/patch with safety checks and chunked appends for large generated files
- 🔒 Native shell command execution with OS-aware prompt guidance
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
MOONSHOT_API_BASE=https://api.moonshot.cn/v1
AGENT_DEFAULT_MODEL=moonshot/kimi-k2.5
```

Optional: configure MCP servers through `AGENT_MCP_SERVERS_JSON` or a JSON file at
`~/.config/coding-agent/mcp.json`.

If the default `mcp.json` does not exist yet, the agent creates it automatically with an
empty `mcpServers` object.

Example:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

MCP tools are registered into the runtime tool registry with namespaced names such as
`mcp__filesystem__read_file`.

## Usage

### Interactive mode

```bash
# Start interactive session
coding-agent

# Start interactive session in a specific workspace
coding-agent -w /path/to/project

# Equivalent explicit interactive command
coding-agent chat

# Start in specific directory
coding-agent chat /path/to/project

# Or use the workspace option
coding-agent chat -w /path/to/project

# Use specific model
coding-agent chat -m gpt-4o

# Print detailed LLM request/response logs
coding-agent chat --debug
```

After you submit the first interactive message, the agent creates a per-session LLM log file and prints its absolute path.
Interactive responses stream by default.
The startup banner shows `Workspace`, preserving the `-w/--workspace` value when provided.
Tool invocations are announced in the console when they run.
Long-running tool execution and MCP configuration or reload flows show a visible loading state in interactive mode.
Responses are rendered as Markdown in the CLI.

Interactive mode also supports local slash commands:

```bash
/mcp
/mcp add filesystem {"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}
/mcp remove filesystem
/mcp reload
/skills
/skills reload
```

Typing `/` in interactive mode will show slash command candidates automatically.
`/mcp` shows the effective MCP configuration, persists file-based changes to the active
`mcp.json`, and reloads MCP tools into the current session immediately.
When MCP servers are configured, startup also prints whether initialization succeeded and
which MCP tools are available.
`/skills` shows workspace-local installed skills discovered under `.coding-agent/skills`,
`.codex/skills`, and `.agents/skills`. The system prompt also advertises those skills so
the agent can inspect a matching `SKILL.md` on demand.
`execute_shell` now runs arbitrary commands in the detected native shell for the current
platform: PowerShell on Windows, bash on macOS/Linux.
When generating large files, the agent can now split writes across `write_file` followed
by `append_file` to avoid oversized tool-call payloads.
For focused edits to existing files, the agent can use `read_file` on a small line range
and then apply `patch_file` to replace or delete only that block.

### Single command mode

```bash
coding-agent run "Explain the codebase structure"
coding-agent run -w /path/to/project "Explain the codebase structure"
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
