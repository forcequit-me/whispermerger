"""WhisperMerger as an MCP server (#36): the actions in actions.py as tools an AI can call.

Runs over stdio. Add it to Claude Code with
    claude mcp add whispermerger -- <repo>\\.venv\\Scripts\\python.exe <repo>\\backend\\mcp_server.py
Tools call the local server's HTTP API directly (starting the server if needed), not the CLI, so
results stay structured. Nothing may print to stdout here: stdout carries the protocol.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import actions  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

READ_ONLY = {actions.add_files, actions.get_progress, actions.get_transcript, actions.get_batch,
             actions.list_models, actions.cache_stats, actions.status}

server = MCPServer(
    "whispermerger",
    instructions=(
        "WhisperMerger transcribes multi-track recordings locally on the GPU and merges each track "
        "into one speaker-labeled transcript with word timings. Typical flow: add_files to see a "
        "file's tracks, transcribe (returns a job id), get_progress until done, get_transcript, "
        "edit_lines to rename speakers or fix lines, export to write SRT/TXT/CSV/JSON. Paths are "
        "local file paths; files are read in place. Transcription can take minutes for long files."),
)


def _tool(fn):
    """The action as a tool. A refusal (ActionError) reaches the model as the reason ("no line 12"),
    since the SDK hides the message of any other exception."""
    @functools.wraps(fn)
    def run(*a, **k):
        try:
            return fn(*a, **k)
        except actions.ActionError as e:
            raise ToolError(str(e)) from None
    return run


for fn in actions.ACTIONS:
    server.add_tool(_tool(fn), annotations=ToolAnnotations(
        readOnlyHint=fn in READ_ONLY, destructiveHint=fn in actions.DESTRUCTIVE))

if __name__ == "__main__":
    server.run("stdio")
