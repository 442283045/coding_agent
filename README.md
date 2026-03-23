# Coding Agent

基于 Python 和 `uv` 构建的 AI 驱动 CLI 编码助手。

## 功能特性

- 🤖 支持多模型 LLM（OpenAI、Anthropic，以及通过 LiteLLM 接入的更多模型）
- 🛠️ 可扩展的工具系统，内置工具并支持基于 FastMCP 的 MCP 服务器
- 🧠 支持项目本地 `SKILL.md` 工作流的工作区技能发现
- 🔍 使用 Tree-sitter 进行代码搜索与分析
- 📝 提供带安全检查的文件读写/补丁能力，并支持对大型生成文件进行分块追加
- 🧩 提供精确文本替换、冲突检测和文件变更 diff 预览，降低编辑失败概率
- 🔒 原生 Shell 命令执行，并根据操作系统提供相应提示
- 💾 会话历史管理

## 安装

### 使用 uv（推荐）

```bash
# 克隆仓库
git clone <repo-url>
cd coding_agent

# 安装依赖
uv pip install -e ".[dev]"

# 或根据锁文件同步依赖
uv sync
```

### 使用 pip

```bash
pip install -e ".[dev]"
```

## 配置

创建 `.env` 文件：

```bash
cp .env.example .env
```

在 `.env` 中填写你的 API 密钥：

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MOONSHOT_API_KEY=your_moonshot_key
MOONSHOT_API_BASE=https://api.moonshot.cn/v1
AGENT_DEFAULT_MODEL=moonshot/kimi-k2.5
```

可选：通过 `AGENT_MCP_SERVERS_JSON` 或位于
`~/.config/coding-agent/mcp.json` 的 JSON 文件来配置 MCP 服务器。

如果默认的 `mcp.json` 尚不存在，代理会自动创建该文件，并写入一个空的
`mcpServers` 对象。

示例：

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

MCP 工具会注册到运行时工具注册表中，并使用带命名空间的名称，例如
`mcp__filesystem__read_file`。

## 使用方式

### 交互模式

```bash
# 启动交互会话
coding-agent

# 在指定工作区启动交互会话
coding-agent -w /path/to/project

# 等价的显式交互命令
coding-agent chat

# 在指定目录启动
coding-agent chat /path/to/project

# 或使用 workspace 选项
coding-agent chat -w /path/to/project

# 使用指定模型
coding-agent chat -m gpt-4o

# 打印详细的 LLM 请求/响应日志
coding-agent chat --debug

# 恢复已有会话
coding-agent chat --resume <session-id>
```

启动交互模式时，代理会创建一个可恢复的 session，并在横幅中显示 session id。
提交第一条交互消息后，代理会为当前 session 创建一个独立的 LLM 日志文件，并输出其绝对路径。
交互响应默认以流式方式输出。
启动横幅会显示 `Workspace`，并在提供 `-w/--workspace` 时保留其值。
工具调用在运行时会显示在控制台中。
文件编辑工具在成功修改文件后，会额外显示一个带行级着色的 diff 预览面板。
已有文件默认不会被 `write_file` 直接覆盖；代理会优先使用 `replace_text` 或带 `expected_old_text` 的 `patch_file` 做局部编辑。
耗时较长的工具执行，以及 MCP 配置或重新加载流程，在交互模式下都会显示可见的加载状态。
代理在等待模型输出首个流式内容时，也会显示 `Thinking...` 加载状态。
当模型流式返回 `reasoning_content` 时，CLI 会在主回答之前及其过程中，以浅灰色预览文本显示这部分内容。
回答会以 Markdown 形式渲染在 CLI 中。

已保存的 session 会写入 `~/.config/coding-agent/sessions/`。可以通过以下命令管理：

```bash
coding-agent sessions list
coding-agent sessions show <session-id>
coding-agent run --resume <session-id> "继续这个会话"
```

交互模式还支持本地斜杠命令：

```bash
/mcp
/mcp add filesystem {"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}
/mcp remove filesystem
/mcp reload
/skills
/skills reload
```

在交互模式中输入 `/` 会自动显示可用的斜杠命令候选项。
`/mcp` 会显示当前生效的 MCP 配置，将基于文件的变更持久化到当前使用的
`mcp.json`，并立即将 MCP 工具重新加载到当前会话中。
当配置了 MCP 服务器时，启动时还会输出初始化是否成功，以及有哪些 MCP 工具可用。
`/skills` 会显示在 `.coding-agent/skills`、`.codex/skills` 和 `.agents/skills`
下发现的、工作区本地已安装技能。系统提示词也会告知这些技能，以便代理在需要时检查匹配的 `SKILL.md`。
`execute_shell` 现在会在当前平台检测到的原生 Shell 中执行任意命令：Windows 使用 PowerShell，macOS/Linux 使用 bash。
在生成大型文件时，代理现在可以先调用 `write_file`，再通过 `append_file` 分段写入，以避免过大的工具调用负载。
对于已有文件的局部编辑，代理会优先先用 `read_file` 读取目标片段，再通过 `replace_text` 做精确替换；如果更适合按行修改，则会使用带 `expected_old_text` 的 `patch_file` 来检测内容漂移并避免误改。

### 单次命令模式

```bash
coding-agent run "解释代码库结构"
coding-agent run -w /path/to/project "解释代码库结构"
coding-agent run --resume <session-id> "继续之前的会话"
```

### 可用命令

```bash
coding-agent --help
coding-agent chat --help
coding-agent run --help
coding-agent sessions --help
```

## 路线图

仓库的成熟度路线图和里程碑规划见 [ROADMAP.md](ROADMAP.md)。

## 项目结构

```text
coding_agent/
├── cli.py              # CLI 入口
├── config.py           # 配置管理
├── agent/              # Agent 核心逻辑
│   ├── core.py         # 主 Agent 循环
│   ├── state.py        # 会话状态
│   └── history.py      # 对话历史
├── llm/                # LLM 客户端
│   ├── client.py       # 统一的 LLM 接口
│   └── tokenizer.py    # Token 计数
├── tools/              # 工具注册表
│   ├── registry.py     # 工具注册
│   ├── file_tools.py   # 文件操作
│   ├── shell_tools.py  # Shell 执行
│   └── code_tools.py   # 代码分析
└── prompts/            # 系统提示词
    └── system.j2       # 主系统提示模板
```

## 开发

```bash
# 运行测试
pytest

# 运行 lint 检查
ruff check .
ruff format .

# 类型检查
mypy coding_agent
```

## 许可证

MIT
