#!/usr/bin/env python3
"""Render a Claude Code session JSONL transcript to a readable markdown file.

User and assistant text is included in full; tool calls are summarized to one
line (tool name + brief inputs); tool results are truncated to keep the file
navigable. The result is a faithful narrative record of the session.

    python scripts/export_transcript.py [transcript.jsonl] [out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TRANSCRIPT = (
    Path.home()
    / ".claude/projects/-Users-patrickbruce-Documents-Speech-Recognition"
    / "f4771e12-6f76-4bee-92fd-5e6ecfffab89.jsonl"
)
DEFAULT_OUT = Path("docs/session-transcript.md")

# Cap on how much of a tool result is shown inline; full results are in the JSONL.
RESULT_CHAR_CAP = 600


def _truncate(text: str, cap: int = RESULT_CHAR_CAP) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n\n_…[{len(text) - cap} more chars truncated]_"


def _tool_input_summary(tool_input: dict) -> str:
    """One-line preview of a tool call's inputs."""
    if not isinstance(tool_input, dict):
        return str(tool_input)
    parts = []
    for key in ("file_path", "path", "command", "description", "url", "query",
                "old_string", "content"):
        if key in tool_input:
            val = str(tool_input[key]).replace("\n", " ")
            if len(val) > 80:
                val = val[:80] + "…"
            parts.append(f"{key}={val!r}")
            break
    if not parts:
        # Fall back to first key.
        for k, v in tool_input.items():
            val = str(v).replace("\n", " ")
            if len(val) > 80:
                val = val[:80] + "…"
            parts.append(f"{k}={val!r}")
            break
    return ", ".join(parts)


def _render_blocks(blocks) -> list[str]:
    """Render an assistant message's content blocks into markdown lines."""
    out: list[str] = []
    if isinstance(blocks, str):
        return [blocks]
    if not isinstance(blocks, list):
        return [str(blocks)]
    for block in blocks:
        if not isinstance(block, dict):
            out.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            out.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "tool")
            summary = _tool_input_summary(block.get("input", {}))
            out.append(f"\n> **tool: {name}**({summary})\n")
        elif btype == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                content = "\n".join(parts)
            text = _truncate(str(content))
            out.append(f"\n```\n{text}\n```\n")
        elif btype == "thinking":
            # Internal model thinking; omit by default to keep the transcript clean.
            continue
        else:
            out.append(f"_[unhandled block type: {btype}]_")
    return out


def render(transcript: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# ASR Robustness — Claude Code session transcript",
        "",
        f"_Rendered from `{transcript}`._",
        "",
        "User and assistant prose are included verbatim; tool calls appear as a "
        "single summary line; tool outputs are truncated to keep the document "
        "readable. The original JSONL has the unabridged record.",
        "",
        "---",
        "",
    ]

    with open(transcript) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            kind = rec.get("type")
            message = rec.get("message") or {}
            role = message.get("role") or rec.get("role") or kind

            if role == "user":
                lines.append("## User")
                lines.append("")
                lines.extend(_render_blocks(message.get("content", rec.get("content", ""))))
                lines.append("")
            elif role == "assistant":
                lines.append("## Assistant")
                lines.append("")
                lines.extend(_render_blocks(message.get("content", rec.get("content", ""))))
                lines.append("")
            elif kind == "summary":
                lines.append(f"_Summary: {rec.get('summary', '')}_")
                lines.append("")
            # Skip system reminders and meta records to keep the doc focused.

    out.write_text("\n".join(lines))
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.0f} KB)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("transcript", nargs="?", type=Path, default=DEFAULT_TRANSCRIPT)
    ap.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if not args.transcript.exists():
        print(f"transcript not found: {args.transcript}", file=sys.stderr)
        return 1
    render(args.transcript, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
