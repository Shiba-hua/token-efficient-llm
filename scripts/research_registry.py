"""Single-writer literature registry. Structural checks do not judge paper prose."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'sources' / 'papers.json'


def normalize_identifier(value: str) -> str:
    value = unquote(value.strip())
    match = re.search(r'(?:arxiv(?:\.org/(?:abs|html|pdf)/|:))?(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?', value, re.I)
    if match:
        return 'arxiv:' + match.group(1)
    match = re.search(r'(10\.\d{4,9}/[^\s?#]+)', value, re.I)
    if match:
        return 'doi:' + match.group(1).rstrip('/').lower()
    if value.startswith(('https://', 'http://')):
        parsed = urlsplit(value)
        return 'url:' + parsed.netloc.lower() + parsed.path.rstrip('/')
    return value.lower()


def validate(data: dict) -> list[str]:
    errors, papers, aliases, tasks = [], {}, {}, set()
    for p in data['papers']:
        pid = p['paper_id']
        if pid in papers:
            errors.append(f'duplicate paper_id: {pid}')
        papers[pid] = p
        for value in p.get('identifiers', []):
            key = normalize_identifier(value)
            if key in aliases and aliases[key] != pid:
                errors.append(f'identifier collision: {key}: {aliases[key]}, {pid}')
            aliases[key] = pid
        active = p.get('active_task')
        if active:
            if active['task_id'] in tasks:
                errors.append(f'duplicate active task: {active["task_id"]}')
            tasks.add(active['task_id'])
        if p.get('qc_status') == 'sample_pass' and not p.get('qc_record'):
            errors.append(f'sample_pass without record: {pid}')
    return errors


def claim(data: dict, paper_id: str, task_id: str, owner: str, scope: str, prompt: str) -> None:
    p = next(p for p in data['papers'] if p['paper_id'] == paper_id)
    if p.get('active_task'):
        raise ValueError(f'{paper_id} already has active task {p["active_task"]["task_id"]}')
    if any(x.get('active_task', {}).get('task_id') == task_id for x in data['papers']):
        raise ValueError(f'task_id already active: {task_id}')
    p['active_task'] = {'task_id': task_id, 'owner': owner, 'scope': scope, 'prompt_version': prompt}
    p['status'] = 'reading'
    p['prompt_version'] = prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['check', 'claim', 'finish'])
    parser.add_argument('--registry', type=Path, default=DEFAULT)
    parser.add_argument('--paper')
    parser.add_argument('--task')
    parser.add_argument('--owner')
    parser.add_argument('--scope', default='main_body')
    parser.add_argument('--prompt', default='luna-reading-v1')
    parser.add_argument('--artifact')
    args = parser.parse_args()
    data = json.loads(args.registry.read_text())
    if args.operation == 'claim':
        if not all([args.paper, args.task, args.owner]):
            parser.error('claim requires --paper, --task and --owner')
        claim(data, args.paper, args.task, args.owner, args.scope, args.prompt)
    elif args.operation == 'finish':
        p = next(p for p in data['papers'] if p['paper_id'] == args.paper)
        active = p.get('active_task')
        if not active or active['task_id'] != args.task:
            parser.error('finish requires the matching active task')
        p.setdefault('task_history', []).append(active)
        p.pop('active_task')
        p['status'] = 'draft_ready'
        p['qc_status'] = 'not_sampled'
        if args.artifact:
            p['artifact'] = args.artifact
    errors = validate(data)
    if errors:
        raise SystemExit('\n'.join(errors))
    if args.operation != 'check':
        args.registry.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'papers': len(data['papers']), 'active_tasks': sum(bool(p.get('active_task')) for p in data['papers']), 'structural_errors': 0}))


if __name__ == '__main__':
    main()
