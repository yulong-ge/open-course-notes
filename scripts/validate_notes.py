#!/usr/bin/env python3
"""Validate publishable course-note invariants before building the site."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSES_DIR = ROOT / "docs" / "courses"
MKDOCS_CONFIG = ROOT / "mkdocs.yml"
KATEX_BOOTSTRAP = ROOT / "docs" / "javascripts" / "katex.js"

IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
OFFICIAL_SLIDE_RE = re.compile(r"^assets/slides/slide-\d{3}\.jpg$")
TIME_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:--|-|–|—|至)\s*"
    r"\d{1,2}:\d{2}(?::\d{2})?\b"
)
BANNED_RE = re.compile(
    r"\[cite\]|TODO|FIXME|TBD|PLACEHOLDER|\.work/|/Users/|"
    r"\[INAUDIBLE\]|\]\(file://|\]\(source/|`source/"
)


def validate_site_config() -> list[str]:
    """Check the complete KaTeX stack needed for correct browser rendering."""
    errors: list[str] = []
    if not MKDOCS_CONFIG.is_file():
        return ["missing mkdocs.yml"]

    config = MKDOCS_CONFIG.read_text(encoding="utf-8")
    required_config = {
        "pymdownx.arithmatex": "Arithmatex Markdown extension",
        "generic: true": "generic Arithmatex output",
        "hooks/github_callouts.py": "GitHub-style callout hook",
        "javascripts/katex.js": "local KaTeX bootstrap",
        "katex.min.js": "KaTeX runtime",
        "auto-render.min.js": "KaTeX auto-render extension",
        "katex.min.css": "KaTeX stylesheet",
    }
    for marker, description in required_config.items():
        if marker not in config:
            errors.append(f"missing {description}: {marker}")

    if not KATEX_BOOTSTRAP.is_file():
        errors.append("missing docs/javascripts/katex.js")
    else:
        bootstrap = KATEX_BOOTSTRAP.read_text(encoding="utf-8")
        for marker in ("document$.subscribe", "renderMathInElement"):
            if marker not in bootstrap:
                errors.append(f"KaTeX bootstrap missing {marker}")

    return errors


def lesson_pages() -> list[Path]:
    pages: list[Path] = []
    if not COURSES_DIR.is_dir():
        return pages
    for course_dir in sorted(path for path in COURSES_DIR.iterdir() if path.is_dir()):
        for child in sorted(path for path in course_dir.iterdir() if path.is_dir()):
            page = child / "index.md"
            if page.is_file() and (child / "assets").is_dir():
                pages.append(page)
    return pages


def prose_lines(lines: list[str]) -> list[str]:
    """Return Markdown lines outside fenced code blocks."""
    result: list[str] = []
    fence: str | None = None
    for line in lines:
        stripped = line.lstrip()
        marker = "```" if stripped.startswith("```") else "~~~" if stripped.startswith("~~~") else None
        if marker:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is None:
            result.append(line)
    return result


def validate_page(page: Path) -> list[str]:
    errors: list[str] = []
    lesson_dir = page.parent
    text = page.read_text(encoding="utf-8")
    lines = text.splitlines()
    prose = prose_lines(lines)

    if not text.startswith("# "):
        errors.append("must start with exactly one H1 title")
    if sum(line.startswith("# ") for line in prose) != 1:
        errors.append("must contain exactly one H1 title")

    h2_count = sum(line.startswith("## ") for line in prose)
    summary_count = sum(line.strip() == "### 本章小结" for line in prose)
    if "## 总结与延伸" not in text:
        errors.append("missing final section '## 总结与延伸'")
    if h2_count != summary_count:
        errors.append(
            f"every H2 must end with a summary: H2={h2_count}, "
            f"summaries={summary_count}"
        )
    if len(text) < 10_000:
        errors.append(f"too short for a full lecture: {len(text)} characters")

    banned = BANNED_RE.search(text)
    if banned:
        line_number = text.count("\n", 0, banned.start()) + 1
        errors.append(f"line {line_number}: forbidden publish-time marker {banned.group()!r}")

    teaching_figures = 0
    for index, line in enumerate(lines):
        for match in IMAGE_RE.finditer(line):
            raw_path = match.group("path").strip().split()[0].strip("<>")
            if raw_path.startswith(("http://", "https://", "/", "data:")):
                errors.append(
                    f"line {index + 1}: image must be a local relative asset: {raw_path}"
                )
                continue

            target = (lesson_dir / raw_path).resolve()
            if not target.is_relative_to(lesson_dir.resolve()):
                errors.append(f"line {index + 1}: image escapes lesson directory: {raw_path}")
                continue
            if not target.is_file():
                errors.append(f"line {index + 1}: missing image: {raw_path}")

            if "cover" in Path(raw_path).name.lower():
                continue

            teaching_figures += 1
            if OFFICIAL_SLIDE_RE.fullmatch(raw_path):
                continue

            nearby = "\n".join(lines[index : min(len(lines), index + 4)])
            if not TIME_RE.search(nearby):
                errors.append(
                    f"line {index + 1}: teaching figure lacks a nearby video interval: "
                    f"{raw_path}"
                )

    if teaching_figures < 5:
        errors.append(f"too few teaching figures: {teaching_figures}")

    return errors


def main() -> int:
    pages = lesson_pages()
    if not pages:
        print(f"FAIL: no lesson pages found under {COURSES_DIR}", file=sys.stderr)
        return 1

    failed = False
    config_errors = validate_site_config()
    if config_errors:
        failed = True
        print(f"FAIL {MKDOCS_CONFIG.relative_to(ROOT)}")
        for error in config_errors:
            print(f"  - {error}")
    else:
        print(f"PASS {MKDOCS_CONFIG.relative_to(ROOT)}")

    for page in pages:
        errors = validate_page(page)
        relative = page.relative_to(ROOT)
        if errors:
            failed = True
            print(f"FAIL {relative}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {relative}")

    print(f"\nValidated {len(pages)} lecture pages.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
