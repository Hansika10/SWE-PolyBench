import shlex
from typing import Any, Dict, Iterable, List, Tuple

from software_factory.models import CommandResult
from software_factory.workspace import Workspace

ToolSpec = Dict[str, Any]

TOOL_SPECS: Dict[str, ToolSpec] = {
    "bash": {
        "name": "bash",
        "description": "Run a bash command from the repository root and return its combined output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds (default 300, max 1800)."},
            },
            "required": ["command"],
        },
    },
    "list_files": {
        "name": "list_files",
        "description": "List files tracked by git (plus untracked, non-ignored files) under a path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Default: repository root."}},
        },
    },
    "search": {
        "name": "search",
        "description": "Search file contents with an extended regex (git grep). Returns file:line:text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Limit search to this path."},
            },
            "required": ["pattern"],
        },
    },
    "view_file": {
        "name": "view_file",
        "description": "Show a file with line numbers. Use start_line/end_line for large files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    "str_replace": {
        "name": "str_replace",
        "description": "Replace an exact snippet in a file. old_str must occur exactly once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}

READ_ONLY_TOOLS = ("bash", "list_files", "search", "view_file")
ALL_TOOLS = tuple(TOOL_SPECS)
VIEW_MAX_LINES = 400


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters truncated] ...\n{text[-tail:]}"


class ToolExecutor:
    """Executes agent tool calls against a workspace."""

    def __init__(self, workspace: Workspace, tools: Iterable[str] = ALL_TOOLS, max_output_chars=10000):
        self.workspace = workspace
        self.tools = list(tools)
        self.max_output_chars = max_output_chars

    def specs(self) -> List[ToolSpec]:
        return [TOOL_SPECS[name] for name in self.tools]

    def execute(self, name: str, tool_input: Dict[str, Any]) -> Tuple[str, bool]:
        """Returns ``(output, ok)``. Errors are reported back to the model, never raised."""
        if name not in self.tools:
            return f"Unknown or unavailable tool: {name}", False
        try:
            output, ok = getattr(self, f"_{name}")(**tool_input)
        except TypeError as e:
            return f"Invalid arguments for {name}: {e}", False
        except Exception as e:
            return f"{type(e).__name__}: {e}", False
        return truncate(output, self.max_output_chars), ok

    @staticmethod
    def _format(result: CommandResult) -> str:
        return f"{result.output.rstrip()}\n[exit code: {result.exit_code}]"

    def _bash(self, command: str, timeout: int = 300) -> Tuple[str, bool]:
        result = self.workspace.run(command, timeout=max(1, min(int(timeout), 1800)))
        return self._format(result), result.exit_code == 0

    def _list_files(self, path: str = ".") -> Tuple[str, bool]:
        result = self.workspace.run(
            f"git ls-files --cached --others --exclude-standard -- {shlex.quote(path)} | head -n 1000"
        )
        return result.output or "(no files)", result.exit_code == 0

    def _search(self, pattern: str, path: str = ".") -> Tuple[str, bool]:
        result = self.workspace.run(
            f"git grep -nIE --untracked -e {shlex.quote(pattern)} -- {shlex.quote(path)} | head -n 200"
        )
        return result.output or "(no matches)", True

    def _view_file(self, path: str, start_line: int = 1, end_line: int = 0) -> Tuple[str, bool]:
        lines = self.workspace.read_file(path).splitlines()
        start = max(1, int(start_line))
        end = int(end_line) if end_line else start + VIEW_MAX_LINES - 1
        end = min(end, len(lines))
        body = "\n".join(f"{i:6d}\t{lines[i - 1]}" for i in range(start, end + 1))
        footer = f"\n[lines {start}-{end} of {len(lines)}]"
        return body + footer, True

    def _write_file(self, path: str, content: str) -> Tuple[str, bool]:
        self.workspace.write_file(path, content)
        return f"Wrote {len(content.splitlines())} lines to {path}", True

    def _str_replace(self, path: str, old_str: str, new_str: str) -> Tuple[str, bool]:
        content = self.workspace.read_file(path)
        count = content.count(old_str)
        if count != 1:
            return f"old_str must occur exactly once in {path}, found {count} occurrences.", False
        self.workspace.write_file(path, content.replace(old_str, new_str, 1))
        return f"Edited {path}", True
