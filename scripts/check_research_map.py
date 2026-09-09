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
from check_math import collect as collect_math


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
    metadata = ROOT / "meta"
    registry = json.loads((ROOT / "sources/papers.json").read_text())
    errors = validate(registry)
    chapters = json.loads((metadata / "chapters.json").read_text())
    slugs = {chapter["slug"] for chapter in chapters}
    for forbidden in ['docs/chapters', 'docs/technical', 'meta/research-map-v3',
                      'assets/maps-v3', 'assets/papers-v3', 'assets/plots', 'assets/generated']:
        if (ROOT / forbidden).exists():
            errors.append(f'obsolete directory recreated: {forbidden}')
    for legacy in (ROOT / 'docs').glob('[0-9][0-9]-*.md'):
        if legacy.name != '00-overview.md':
            errors.append(f'obsolete numbered compatibility page: {legacy.name}')
    if len(chapters) != 13 or len(slugs) != 13:
        errors.append("expected 13 unique chapter destinations")
    for chapter in chapters:
        if not (ROOT / chapter["page"]).is_file():
            errors.append(f"missing chapter: {chapter['page']}")
    published = [p for p in registry["papers"] if p["status"] == "published"]
    for paper in registry['papers']:
        if paper.get('artifact') and not (ROOT / paper['artifact']).is_file():
            errors.append(f'missing reading/survey artifact: {paper["paper_id"]}')
        if paper.get('qc_record'):
            path, _, record_id = paper['qc_record'].partition('#')
            location = ROOT / path
            if not location.is_file():
                errors.append(f'missing QC file: {paper["paper_id"]}')
            elif record_id and record_id not in {r.get('id') for r in json.loads(location.read_text())}:
                errors.append(f'missing QC record: {paper["paper_id"]}')
    artifacts = []
    for paper in published:
        pid = paper["paper_id"]
        expected = f"docs/{paper['primary_chapter']}/reference/{pid}.md"
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
                (ROOT / "docs").glob("*/reference/*.md")}
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
    for ledger, fields in [('map-specifications.json', ('prompt_file', 'target')),
                          ('paper-assets.json', ('file', 'preview')),
                          ('main-figure-placements.json', ('page', 'file'))]:
        for record in json.loads((metadata / ledger).read_text()):
            for field in fields:
                if record.get(field) and not (ROOT / record[field]).is_file():
                    errors.append(f'{ledger}: missing {field}: {record[field]}')
    for item in json.loads((metadata / 'paper-assets.json').read_text()):
        for path_field, hash_field in [('file', 'sha256'), ('preview', 'preview_sha256')]:
            asset = ROOT / item[path_field]
            if asset.is_file() and item.get(hash_field) and hashlib.sha256(asset.read_bytes()).hexdigest() != item[hash_field]:
                errors.append(f'paper asset changed: {item[path_field]}')
    # A changed formula requires new real-render evidence, not just a known macro.
    # Prose-only edits do not invalidate a formula receipt.
    receipt = metadata / 'github-render-audit.json'
    if receipt.exists():
        report = json.loads(receipt.read_text())
        checked = {page['path']: page for page in report['pages']}
        formulas: dict[str, list[dict]] = {}
        for expression in collect_math():
            formulas.setdefault(expression['file'], []).append(
                {'kind': expression['kind'], 'sha256': expression['sha256']})
        for path, expressions in formulas.items():
            signature = hashlib.sha256(json.dumps(expressions, sort_keys=True).encode()).hexdigest()
            page = checked.get(path, {})
            if not page.get('pass') or page.get('math_signature') != signature:
                errors.append(f'missing or stale GitHub formula rendering evidence: {path}')
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
