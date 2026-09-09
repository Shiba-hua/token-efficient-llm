#!/usr/bin/env python3
"""Check published Markdown structure using only the Python standard library.

Scope: README.md, CONTRIBUTING.md, and Markdown under docs/, meta/, assets/, sources/.
Private research/report/tmp directories are excluded, including nested ones.
Inline links, reference definitions, and HTML href/src targets are checked.
Fenced and inline code are ignored for links but checked for local user paths.
This is not a full Markdown parser, anchor checker, network link checker, or
review of formulas, source claims, licenses, or image appearance.
"""

from __future__ import annotations

import html
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {"research", "report", "reports", "tmp", "temp"}
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
USER_PATH = re.compile(
    r"(?<![\w:])(?:/(?:Users|home)/[^\s/]+|[A-Za-z]:[\\/]+Users[\\/]+)",
    re.IGNORECASE,
)
REFERENCE = re.compile(r"^ {0,3}\[[^\]\n]+\]:\s*(.*)$", re.MULTILINE)
INLINE_START = re.compile(r"(?<!\\)\]\(")
INLINE_CODE = re.compile(r"(?<![\\`])(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)


def published_markdown(root: Path) -> list[Path]:
    paths = {root / "README.md", root / "CONTRIBUTING.md", root / "VERSIONS.md"}
    for directory in ("docs", "meta", "assets", "sources"):
        paths.update(
            path for path in (root / directory).rglob("*.md")
            if not PRIVATE_DIRS.intersection(path.relative_to(root).parts[:-1])
        )
    return sorted(paths)


def blank(text: str) -> str:
    """Preserve offsets and line numbers while masking Markdown code."""
    return "".join("\n" if character == "\n" else " " for character in text)


def prose_only(text: str) -> tuple[str, list[tuple[int, str]]]:
    output = []
    errors = []
    opening = None
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        # A fence can also be nested in a block quote.
        candidate = re.sub(r"^(?: {0,3}> ?)+", "", line)
        match = FENCE.match(candidate)
        if opening is not None:
            marker, start = opening
            if (match and match[1][0] == marker[0]
                    and len(match[1]) >= len(marker) and not match[2].strip()):
                opening = None
            output.append(blank(line))
        elif match and not (match[1][0] == "`" and "`" in match[2]):
            opening = (match[1], number)
            output.append(blank(line))
        else:
            output.append(line)
    if opening is not None:
        errors.append((opening[1], "unclosed code fence"))
    prose = INLINE_CODE.sub(lambda match: blank(match[0]), "".join(output))
    return prose, errors


def destination(text: str, start: int) -> tuple[str, int]:
    """Read one Markdown destination, allowing escapes and balanced brackets."""
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    angled = index < len(text) and text[index] == "<"
    if angled:
        index += 1
    value = []
    depth = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and index + 1 < len(text):
            value.append(text[index + 1])
            index += 2
            continue
        if angled:
            if character == ">":
                break
        else:
            if character.isspace() or (character == ")" and depth == 0):
                break
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
        value.append(character)
        index += 1
    return "".join(value), index


class HTMLLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append((self.getpos()[0], value))


def links_in(prose: str) -> list[tuple[int, str]]:
    links = []
    for match in INLINE_START.finditer(prose):
        target, _ = destination(prose, match.end())
        links.append((prose.count("\n", 0, match.start()) + 1, target))
    for match in REFERENCE.finditer(prose):
        target, _ = destination(prose, match.start(1))
        links.append((prose.count("\n", 0, match.start()) + 1, target))
    parser = HTMLLinks()
    parser.feed(prose)
    links.extend(parser.links)
    return links


def check_document(path: Path, root: Path) -> tuple[list[tuple[int, str]], int]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [(1, f"cannot read Markdown: {error}")], 0
    prose, errors = prose_only(text)
    for number, line in enumerate(text.splitlines(), 1):
        if USER_PATH.search(unquote(html.unescape(line))):
            errors.append((number, "absolute local user path; use a repository-relative path"))
    count = 0
    for number, raw in links_in(prose):
        target = html.unescape(raw)
        try:
            url = urlsplit(target)
        except ValueError:
            errors.append((number, f"invalid link target: {raw!r}"))
            continue
        if url.scheme or url.netloc or not url.path:
            continue
        count += 1
        relative = Path(unquote(url.path))
        if relative.is_absolute():
            errors.append((number, f"local link must be relative: {raw!r}"))
            continue
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(root):
            errors.append((number, f"local link escapes repository: {raw!r}"))
        elif not resolved.exists():
            errors.append((number, f"missing local target: {raw!r}"))
    return sorted(set(errors)), count


def main() -> int:
    documents = published_markdown(ROOT)
    errors = 0
    targets = 0
    for path in documents:
        findings, count = check_document(path, ROOT)
        targets += count
        for number, message in findings:
            print(f"{path.relative_to(ROOT)}:{number}: {message}", file=sys.stderr)
        errors += len(findings)
    if errors:
        print(f"Content checks failed: {errors} issue(s) in {len(documents)} Markdown files.",
              file=sys.stderr)
        return 1
    print(f"Content checks passed: {len(documents)} Markdown files, {targets} local targets.")
    print("Checked file existence, fences, and local user paths; external URLs, anchors, "
          "and content meaning were not validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
