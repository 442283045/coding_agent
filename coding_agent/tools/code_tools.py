"""Code analysis tools using tree-sitter."""

from pathlib import Path
from typing import Any

from coding_agent.config import settings
from coding_agent.tools.registry import registry


def _get_working_dir(ctx: dict[str, Any] | None) -> Path:
    """Get working directory from context or use current directory."""
    if ctx and "working_dir" in ctx:
        return Path(ctx["working_dir"]).resolve()
    return Path.cwd()


def _get_tree_sitter_parser(file_path: Path) -> Any | None:
    """Get appropriate tree-sitter parser for file type."""
    try:
        from tree_sitter import Language, Parser

        suffix = file_path.suffix.lower()

        if suffix == ".py":
            from tree_sitter_python import language as python_lang

            lang = Language(python_lang)
        elif suffix in (".js", ".jsx"):
            from tree_sitter_javascript import language as js_lang

            lang = Language(js_lang)
        elif suffix in (".ts", ".tsx"):
            from tree_sitter_typescript import language_typescript

            lang = Language(language_typescript())
        else:
            return None

        parser = Parser(lang)
        return parser
    except ImportError:
        return None
    except Exception:
        return None


@registry.tool(
    name="get_file_summary",
    description="Get a summary of a code file - shows classes, functions, and their line numbers. "
    "Supports Python, JavaScript, and TypeScript files.",
)
async def get_file_summary(
    path: str,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Get a structural summary of a code file."""
    working_dir = _get_working_dir(ctx)
    file_path = (working_dir / path).resolve()

    if not file_path.exists():
        return f"Error: File '{path}' not found."

    if not file_path.is_file():
        return f"Error: '{path}' is not a file."

    if file_path.stat().st_size > settings.max_file_size:
        return f"Error: File '{path}' is too large."

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
    except UnicodeDecodeError:
        return f"Error: File '{path}' is not a text file."

    # Try tree-sitter parsing
    parser = _get_tree_sitter_parser(file_path)

    if parser is None:
        # Fallback: simple regex-based extraction
        import re

        results = []

        # Python-style definitions
        for i, line in enumerate(lines, 1):
            # Match function/class definitions
            match = re.match(r"^(\s*)(def|class)\s+(\w+)", line)
            if match:
                indent = len(match.group(1))
                kind = match.group(2)
                name = match.group(3)
                level = "  " * (indent // 4)
                icon = "📦" if kind == "class" else "🔧"
                results.append(f"{level}{icon} {name} (line {i})")

        if results:
            return f"Structure of '{path}':\n" + "\n".join(results)
        return f"No class/function definitions found in '{path}'."

    try:
        tree = parser.parse(content.encode("utf-8"))
        root = tree.root_node

        results = []

        def traverse(node: Any, depth: int = 0) -> None:
            indent = "  " * depth

            if node.type in ("function_definition", "function_declaration"):
                # Find function name
                for child in node.children:
                    if child.type == "identifier":
                        name = child.text.decode("utf-8")
                        results.append(f"{indent}🔧 {name} (line {node.start_point[0] + 1})")
                        break

            elif node.type in ("class_definition", "class_declaration"):
                # Find class name
                for child in node.children:
                    if child.type == "identifier" or child.type == "type_identifier":
                        name = child.text.decode("utf-8")
                        results.append(f"{indent}📦 {name} (line {node.start_point[0] + 1})")
                        break

            elif node.type == "method_definition":
                for child in node.children:
                    if child.type == "property_identifier":
                        name = child.text.decode("utf-8")
                        results.append(f"{indent}  🔧 {name} (line {node.start_point[0] + 1})")
                        break

            for child in node.children:
                traverse(
                    child,
                    depth + 1 if node.type in ("class_definition", "class_declaration") else depth,
                )

        traverse(root)

        if results:
            return f"Structure of '{path}':\n" + "\n".join(results)
        return f"No class/function definitions found in '{path}'."

    except Exception as e:
        return f"Error parsing file: {e}"


@registry.tool(
    name="find_symbol",
    description="Search for a symbol (function, class, variable) definition in the codebase. "
    "Returns file paths and line numbers where the symbol is defined.",
)
async def find_symbol(
    name: str,
    path: str = ".",
    ctx: dict[str, Any] | None = None,
) -> str:
    """Find symbol definitions in the codebase."""
    import re

    working_dir = _get_working_dir(ctx)
    search_path = (working_dir / path).resolve()

    if not search_path.exists():
        return f"Error: Path '{path}' not found."

    try:
        # Patterns for different languages
        patterns = {
            ".py": rf"^\s*(def|class)\s+{re.escape(name)}\b",
            ".js": rf"^\s*(function|class|const|let|var)\s+{re.escape(name)}\b",
            ".ts": rf"^\s*(function|class|const|let|var|interface|type)\s+{re.escape(name)}\b",
        }

        matches = []

        for file_path in search_path.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.stat().st_size > settings.max_file_size:
                continue

            suffix = file_path.suffix.lower()
            if suffix not in patterns:
                continue

            pattern = patterns[suffix]

            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()

                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        rel_path = file_path.relative_to(working_dir)
                        matches.append(f"📄 {rel_path}:{i}: {line.strip()}")

            except UnicodeDecodeError, Exception:
                continue

        if not matches:
            return f"No definition found for symbol '{name}'."

        return f"Found '{name}' in {len(matches)} location(s):\n" + "\n".join(matches[:20])

    except Exception as e:
        return f"Error searching for symbol: {e}"


@registry.tool(
    name="get_function_code",
    description="Get the complete code of a specific function or method. "
    "You must specify the file path and function name.",
)
async def get_function_code(
    file_path: str,
    function_name: str,
    ctx: dict[str, Any] | None = None,
) -> str:
    """Extract a specific function's code from a file."""
    import re

    working_dir = _get_working_dir(ctx)
    full_path = (working_dir / file_path).resolve()

    if not full_path.exists():
        return f"Error: File '{file_path}' not found."

    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.splitlines()
    except UnicodeDecodeError:
        return f"Error: File '{file_path}' is not a text file."

    # Find function definition
    func_pattern = rf"^(\s*)(def|class)\s+{re.escape(function_name)}\b"

    start_line: int | None = None
    base_indent: int | None = None

    for i, line in enumerate(lines):
        match = re.match(func_pattern, line)
        if match:
            start_line = i
            base_indent = len(match.group(1))
            break

    if start_line is None:
        return f"Error: Function or class '{function_name}' not found in '{file_path}'."

    # Find the end of the function
    end_line = start_line + 1

    for i in range(start_line + 1, len(lines)):
        line = lines[i]

        # Skip empty lines
        if not line.strip():
            end_line = i + 1
            continue

        # Check indentation
        current_indent = len(line) - len(line.lstrip())

        # If we hit a line with same or less indentation, it's a new top-level definition
        if base_indent is not None and current_indent <= base_indent and line.strip():
            break

        end_line = i + 1

    # Extract the function code
    func_lines = lines[start_line:end_line]
    func_code = "\n".join(func_lines)

    return f"```python\n# {file_path}:{start_line + 1}-{end_line}\n{func_code}\n```"
