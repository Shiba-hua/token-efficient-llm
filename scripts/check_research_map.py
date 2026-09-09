"""Check map navigation and publication records, never judge prose coverage.

Heading fragments use the subset of GFM headings used by this repository.
External links and scientific correctness are outside this mechanical check.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

from check_content import ROOT, links_in, prose_only, published_markdown
from research_registry import validate


def anchors(text: str) -> set[str]:
    prose, _ = prose_only(text)
    found = set(re.findall(r'<a\s+id=["\']([^"\']+)', prose))
    counts: dict[str, int] = {}
    # Use original heading text: masking inline code would delete method names.
    in_fence = False
    for line in text.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
        if in_fence:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
        value = re.sub(r"<[^>]+>", "", html.unescape(value)).lower()
        value = re.sub(r"[^\w\-\s]", "", value).replace(" ", "-")
        count = counts.get(value, 0)
        counts[value] = count + 1
        found.add(value + (f"-{count}" if count else ""))
    return found


def main() -> int:
    metadata = ROOT / "meta/research-map-v3"
    registry = json.loads((ROOT / "sources/papers.json").read_text())
    errors = validate(registry)
    chapters = json.loads((metadata / "chapters.json").read_text())
    slugs = {chapter["slug"] for chapter in chapters}
    if len(chapters) != 13 or len(slugs) != 13:
        errors.append("expected 13 unique chapter destinations")
    for chapter in chapters:
        if not (ROOT / chapter["page"]).is_file():
            errors.append(f"missing chapter: {chapter['page']}")
    published = [p for p in registry["papers"] if p["status"] == "published"]
    artifacts = []
    for paper in published:
        pid = paper["paper_id"]
        expected = f"docs/chapters/{paper['primary_chapter']}/reference/{pid}.md"
        if paper.get("artifact") != expected or not (ROOT / expected).is_file():
            errors.append(f"invalid primary artifact: {pid}")
        artifacts.append(expected)
        for field in ("reading_version", "read_scope", "reading_prompt_version",
                      "lifecycle_tags", "mechanism_tags", "task_history"):
            if not paper.get(field):
                errors.append(f"missing {field}: {pid}")
        if not set(paper.get("related_chapters", [])).issubset(slugs):
            errors.append(f"unknown related chapter: {pid}")
        if paper.get("active_task"):
            errors.append(f"published paper still has active reading: {pid}")
    physical = {str(p.relative_to(ROOT)) for p in
                (ROOT / "docs/chapters").glob("*/reference/*.md")}
    if set(artifacts) != physical or len(artifacts) != len(set(artifacts)):
        errors.append("reference files and unique publication records differ")
    queue = json.loads((metadata / "reading-queue.json").read_text())
    expected_ids = {p["paper_id"] for p in queue} | {"2602.20945", "src-7cc3fedc37b4"}
    if expected_ids != {p["paper_id"] for p in published}:
        errors.append("calibration/production queue and integrated papers differ")
    if json.loads((metadata / "pending-figures.json").read_text()):
        errors.append("unresolved figure mappings remain")
    for item in json.loads((metadata / "generated-maps.json").read_text()):
        image = ROOT / item["file"]
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"map asset changed since record: {item['slug']}")
        if item["status"] != "visual_pass":
            errors.append(f"map review incomplete: {item['slug']}")
    # This checks navigability, not the quality of every paper's body.
    cache: dict[Path, set[str]] = {}
    fragment_count = 0
    for source in published_markdown(ROOT):
        if not source.exists():
            continue
        prose, _ = prose_only(source.read_text())
        for line, raw in links_in(prose):
            url = urlsplit(html.unescape(raw))
            if url.scheme or url.netloc or not url.fragment:
                continue
            target = (source.parent / unquote(url.path)).resolve() if url.path else source
            if target.suffix != ".md" or not target.exists():
                continue
            fragment_count += 1
            if target not in cache:
                cache[target] = anchors(target.read_text())
            fragment = unquote(url.fragment)
            if fragment not in cache[target]:
                errors.append(f"{source.relative_to(ROOT)}:{line}: missing fragment {raw}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(json.dumps({"chapters": len(chapters), "unique_readings": len(published),
                      "local_markdown_fragments": fragment_count,
                      "errors": 0, "prose_quality_audit": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
