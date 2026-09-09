"""Download public arXiv reading materials to an explicitly selected scratch root.

This does not read a paper, verify its claims, or grant permission to reuse figures.
Never executes downloaded TeX. Existing files are retained.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def fetch_one(pid: str, scratch: Path) -> dict:
    requested = re.fullmatch(r'(\d{4}\.\d{4,5})(v\d+)?', pid)
    if not requested:
        raise ValueError('expected an arXiv work id')
    requested_version = pid if requested.group(2) else None
    pid = requested.group(1)
    folder = scratch / pid / 'source'
    folder.mkdir(parents=True, exist_ok=True)
    result = {'paper_id': pid, 'errors': []}

    def download(url: str, target: Path) -> bytes:
        if target.exists():
            return target.read_bytes()
        request = urllib.request.Request(url, headers={'User-Agent': 'ResearchMapSourceCache/1.0'})
        data = urllib.request.urlopen(request, timeout=40).read()
        target.write_bytes(data)
        return data

    try:
        page = download('https://arxiv.org/abs/' + pid, folder / 'abstract.html').decode('utf-8')
        matches = re.findall(re.escape(pid) + r'v(\d+)', page)
        version = requested_version or (pid + 'v' + str(max(map(int, matches))) if matches else pid)
        title = re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', page)
        license_match = re.search(r'<a[^>]+href="([^"]+)"[^>]*>\s*(?:view license|View license)', page)
        if not license_match:
            license_match = re.search(r'href="(https?://creativecommons.org/licenses/[^\"]+)"', page)
        result.update(version=version, title=html.unescape(title.group(1)) if title else None,
                      license_url=license_match.group(1) if license_match else None,
                      abstract_url='https://arxiv.org/abs/' + version)
        data = download('https://arxiv.org/pdf/' + version, folder / 'paper.pdf')
        if not data.startswith(b'%PDF'):
            raise ValueError('downloaded PDF has an unexpected signature')
        if not (folder / 'paper.txt').exists():
            subprocess.run(['pdftotext', '-layout', str(folder / 'paper.pdf'), str(folder / 'paper.txt')], check=True)
        try:
            download('https://arxiv.org/src/' + version, folder / 'latex-source.tar')
            destination = folder / 'tex'
            destination.mkdir(exist_ok=True)
            with tarfile.open(folder / 'latex-source.tar') as archive:
                for item in archive.getmembers():
                    if item.issym() or item.islnk() or not (destination / item.name).resolve().is_relative_to(destination.resolve()):
                        raise ValueError('unsafe tar member')
                archive.extractall(destination, filter='data')
            result['tex_available'] = True
        except Exception as exc:
            result['errors'].append('tex: ' + str(exc))
    except Exception as exc:
        result['errors'].append(str(exc))
    (folder / 'source-manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(lambda pid: fetch_one(pid, args.scratch), args.ids):
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
