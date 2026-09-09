"""Main-agent publication helper: copy existing reading drafts and licensed assets.

It checks file/format consistency, not scientific coverage. Unresolved figures stay
in an explicit pending ledger until the main agent resolves them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from check_content import prose_only
from check_math import extract, fragment_errors

ROOT = Path(__file__).resolve().parents[1]


def math_format(text: str) -> str:
    text = re.sub(r'\\\[\s*\n?', '\n```math\n', text)
    text = re.sub(r'\n?\s*\\\]', '\n```\n', text)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: '$' + m[1] + '$', text, flags=re.S)
    def latex_typos(value: str) -> str:
        return re.sub(r'(?<!\\)\b(qquad|mathcal|mathbf|alpha|beta|gamma|theta|lambda|sigma)\b', r'\\\1', value)
    text = re.sub(r'((?:```|~~~)math\n)(.*?)(\n(?:```|~~~))', lambda m: m[1] + latex_typos(m[2]) + m[3], text, flags=re.S)
    text = re.sub(r'(?<!\$)\$([^$\n]+)\$(?!\$)', lambda m: '$' + latex_typos(m[1]) + '$', text)
    return text


def markdown_hard_breaks(text: str) -> str:
    """Keep intentional Markdown line breaks without trailing whitespace."""
    output, fenced = [], False
    for line in text.splitlines():
        if re.match(r'^\s*(```|~~~)', line):
            fenced = not fenced
        if not fenced and line.endswith('  ') and line.strip():
            line = line.rstrip() + ' <br>'
        elif not fenced:
            line = line.rstrip()
        output.append(line)
    return '\n'.join(output).rstrip() + '\n'


def github_inline_math(text: str) -> str:
    """Protect TeX escapes and Chinese punctuation with GitHub's $`...`$ form.

    Fenced examples and inline code are excluded; existing protected math is
    masked as code and retained. This changes delimiters, not formula content.
    """
    prose, _ = prose_only(text)
    pattern = re.compile(r'(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)')
    replacements = []
    for match in pattern.finditer(prose):
        if not match[1].strip():
            continue
        start, end = match.span()
        original = text[start + 1:end - 1]
        if '`' not in original:
            replacements.append((start, end, '$`' + original + '`$'))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', type=Path, required=True)
    ap.add_argument('--replace-published', action='store_true', help='Explicitly replace integrated readings; otherwise preserve the current mother text.')
    args = ap.parse_args()
    registry_path = ROOT / 'sources/papers.json'
    data = json.loads(registry_path.read_text())
    previous_path = ROOT / 'meta/paper-assets.json'
    previous_assets = {a['file']: a for a in json.loads(previous_path.read_text())} if previous_path.exists() else {}
    pending_path = ROOT / 'meta/pending-figures.json'
    assets = list(previous_assets.values())
    pending = json.loads(pending_path.read_text()) if pending_path.exists() else []
    published = []
    for paper in data['papers']:
        pid = paper['paper_id']
        chapter = paper.get('primary_chapter')
        packet = args.scratch / pid
        if paper['status'] == 'published' and not args.replace_published:
            continue
        if not chapter or paper['status'] not in ('draft_ready', 'calibration_pass', 'published'):
            continue
        if not (packet / 'draft.md').exists() or not (packet / 'handoff.json').exists():
            continue
        handoff = json.loads((packet / 'handoff.json').read_text())
        text = math_format((packet / 'draft.md').read_text())
        math_errors = [(f.line, fragment_errors(f)) for f in extract(github_inline_math(text)) if fragment_errors(f)]
        if math_errors:
            raise SystemExit(f'{pid}: draft needs math repair before publication: {math_errors}')
        assets = [a for a in assets if a['paper_id'] != pid]
        pending = [a for a in pending if a['paper_id'] != pid]
        text = re.sub(r'`(\{\{FIG:[^}]+\}\})`', r'\1', text)
        page = ROOT / 'docs' / chapter / 'reference' / (pid + '.md')
        page.parent.mkdir(parents=True, exist_ok=True)
        resolved = {}
        source_targets, digest_targets = {}, {}
        expanded = []
        for entry in handoff.get('figures', []):
            local = entry.get('local_file')
            if isinstance(local, list):
                for n, item in enumerate(local, 1):
                    child = dict(entry)
                    child['logical_id'] = entry['id']
                    child['id'] = entry['id'] + '-part-' + str(n)
                    child['local_file'] = item
                    expanded.append(child)
            else:
                expanded.append(entry)
        for figure in expanded:
            fid = figure.get('id', figure.get('figure_id', ''))
            raw = figure.get('local_file')
            if not fid or not raw:
                continue
            original = Path(raw) if Path(raw).is_absolute() else packet / raw
            permission = figure.get('reuse_allowed')
            allowed = permission is True or figure.get('citation_use') is True or (isinstance(permission, str) and permission.lower().startswith('yes'))
            if not allowed or not original.is_file() or original.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.pdf', '.svg'):
                continue
            safe = re.sub(r'[^A-Za-z0-9._-]+', '-', fid)
            target_dir = ROOT / 'assets/papers' / pid
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / (safe + original.suffix.lower())
            shutil.copyfile(original, target)
            source_targets[str(original.resolve())] = target
            digest_targets[hashlib.sha256(original.read_bytes()).hexdigest()] = target
            preview = target
            if target.suffix == '.pdf':
                preview = target.with_suffix('.png')
                if not preview.exists():
                    subprocess.run(['pdftoppm', '-f', '1', '-singlefile', '-scale-to', '2200', '-png', str(target), str(preview.with_suffix(''))], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            caption = figure.get('caption', fid)
            logical_id = figure.get('logical_id', fid)
            part = f'![{caption}]({os.path.relpath(preview, page.parent)})'
            resolved[logical_id] = resolved.get(logical_id, '') + ('\n\n' if logical_id in resolved else '') + part
            assets.append({'paper_id': pid, 'figure_id': fid, 'logical_id': figure.get('logical_id', fid), 'caption': caption, 'file': str(target.relative_to(ROOT)), 'preview': str(preview.relative_to(ROOT)), 'source_url': figure.get('source_url'), 'license': figure.get('license'), 'use_basis': 'scholarly figure quotation; original rights retained, no relicensing' if figure.get('citation_use') else 'source license/attribution retained', 'transformation': 'source-derived rendering; see source/license note' if 'adaptation' in str(figure.get('license')) or 'cropped' in str(figure.get('license')) else 'format-only full-asset rendering' if preview != target else 'unmodified copy', 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})

        def replace_figure(match: re.Match) -> str:
            marker = match[1].split('|', 1)[0].strip()
            if marker in handoff.get('non_image_markers', {}):
                return '' if marker == '...' else '[原始表格/补充示例](https://arxiv.org/abs/' + pid + ')'
            if marker in resolved:
                return resolved[marker]
            pending.append({'paper_id': pid, 'marker': marker, 'page': str(page.relative_to(ROOT))})
            return '<!-- FIGURE_PENDING:' + pid + ':' + marker + ' -->'

        text = re.sub(r'\{\{FIG:([^}]+)\}\}', replace_figure, text)
        def packet_link(match: re.Match) -> str:
            raw = match[2]
            if not raw.startswith(('assets/', 'source/', '/private/tmp/token-efficient-map-v3/')):
                return match[0]
            path = Path(raw) if Path(raw).is_absolute() else packet / raw
            target = source_targets.get(str(path.resolve()))
            if target is None and path.is_file():
                target = digest_targets.get(hashlib.sha256(path.read_bytes()).hexdigest())
            if target is not None:
                if match[1].startswith('![') and target.suffix == '.pdf' and target.with_suffix('.png').exists():
                    target = target.with_suffix('.png')
                return match[1] + '(' + os.path.relpath(target, page.parent) + ')'
            return match[1] + '(' + (paper.get('versions') or ['https://arxiv.org/abs/' + pid])[0] + ')'
        text = re.sub(r'(!?\[[^\]\n]*\])\(([^)\s]+)\)', packet_link, text)
        # Source-packet paths are provenance bookkeeping, not public download links.
        text = re.sub(r'/private/tmp/token-efficient-map-v3/[^\s)\]`]+', '[temporary reading packet]', text)
        lines = text.splitlines()
        lines[1:1] = ['', '[所属章节](../index.md) · [来源与阅读状态](' + os.path.relpath(registry_path, page.parent) + ')' , '']
        text = '\n'.join(lines).rstrip() + '\n'
        if pid == '2404.19737':
            text = text.replace('它不是在没有中间 token 的情况下直接建模一个严格的联合概率 $P(x_{t+1},\\ldots,x_{t+n}\\mid x_{t:1})$。', '这些边缘分布的乘积可以定义带条件独立假设的模型联合分布，但不保证恢复真实未来 token 的相关性，也不等同于主自回归模型逐步条件化得到的联合分布。')
        page.write_text(github_inline_math(markdown_hard_breaks(text)))
        paper['artifact'] = str(page.relative_to(ROOT))
        paper['reading_version'] = handoff.get('version')
        paper['read_scope'] = handoff.get('read_scope')
        paper['status'] = 'published'
        published.append(pid)
    registry_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    folder = ROOT / 'meta'
    for asset in assets:
        if asset['paper_id'] not in published:
            continue
        old = previous_assets.get(asset['file'], {})
        if old.get('sha256') == asset['sha256']:
            for key in ('use_basis', 'transformation', 'preview_transformation'):
                if key in old:
                    asset[key] = old[key]
        asset['preview_sha256'] = hashlib.sha256((ROOT / asset['preview']).read_bytes()).hexdigest()
    (folder / 'paper-assets.json').write_text(json.dumps(assets, ensure_ascii=False, indent=2) + '\n')
    (folder / 'pending-figures.json').write_text(json.dumps(pending, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'published_readings': len(published), 'resolved_assets': len(assets), 'pending_figure_markers': len(pending)}))


if __name__ == '__main__':
    main()
