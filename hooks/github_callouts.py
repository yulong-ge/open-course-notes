"""Render GitHub/Obsidian callouts as Material for MkDocs admonitions."""

from __future__ import annotations

import re


CALLOUT_RE = re.compile(
    r"^>\s*\[!(?P<kind>NOTE|TIP|IMPORTANT|WARNING|CAUTION|QUOTE)\]\s*(?P<title>.*)$"
)
CALLOUT_TYPES = {
    "NOTE": "note",
    "TIP": "tip",
    "IMPORTANT": "important",
    "WARNING": "warning",
    "CAUTION": "danger",
    "QUOTE": "quote",
}


def on_page_markdown(markdown: str, **_kwargs) -> str:
    """Translate consecutive GitHub-style blockquotes before Markdown parsing."""
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0

    while index < len(lines):
        match = CALLOUT_RE.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue

        kind = CALLOUT_TYPES[match.group("kind")]
        title = match.group("title").strip()
        title_suffix = f' "{title}"' if title else ""
        output.append(f"!!! {kind}{title_suffix}")
        index += 1

        while index < len(lines):
            line = lines[index]
            if line == ">":
                output.append("    ")
            elif line.startswith("> "):
                output.append(f"    {line[2:]}")
            elif line.startswith(">"):
                output.append(f"    {line[1:]}")
            else:
                break
            index += 1

        output.append("")

    return "\n".join(output)
