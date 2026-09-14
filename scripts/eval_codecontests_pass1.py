#!/usr/bin/env python3
"""Evaluate CodeContests pass@1 for one or more result directories."""
import json
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, '/data/ppnm/EasyOPD-baseline')
from verl.utils.reward_score.prime_code import compute_score


def extract_code(text: str) -> str:
    if not text:
        return ''
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    blocks = re.findall(r'(?:^|\n)-{10,}\n(.*?)(?:\n-{10,}|$)', text, re.DOTALL)
    code_blocks = [
        b.strip() for b in blocks
        if any(k in b for k in ('input()', 'def ', 'import ', 'print('))
    ]
    return max(code_blocks, key=len) if code_blocks else ''


def eval_dir(results_dir: str, label: str) -> dict:
    files = sorted(
        glob.glob(os.path.join(results_dir, '[0-9]*.json')),
        key=lambda p: int(Path(p).stem),
    )
    passed = 0
    per_sample = []
    for fp in files:
        with open(fp, encoding='utf-8') as f:
            d = json.load(f)
        tests = list(d.get('public_tests') or []) + list(d.get('private_tests') or [])
        if not tests:
            per_sample.append(0.0)
            continue
        in_outs = {
            'inputs': [t['input'] for t in tests],
            'outputs': [t['output'] for t in tests],
        }
        ok_any = False
        ok_count = 0
        n_comp = len(d.get('completions') or [])
        for c in d.get('completions') or []:
            text = (c.get('text') or '') + '\n' + (c.get('reasoning_content') or '')
            code = extract_code(text)
            if not code:
                continue
            try:
                success, _ = compute_score(
                    f'```python\n{code}\n```', in_outs, continuous=False,
                )
                if success:
                    ok_any = True
                    ok_count += 1
            except Exception:
                pass
        passed += int(ok_any)
        per_sample.append(ok_count / n_comp if n_comp else 0.0)

    n = len(files)
    missing = sorted(set(range(165)) - {int(Path(f).stem) for f in files})
    summary = {
        'label': label,
        'dir': results_dir,
        'n_problems': n,
        'pass_at_1_pct': round(passed / n * 100, 2) if n else 0.0,
        'avg_sample_pass_rate_pct': round(sum(per_sample) / n * 100, 2) if n else 0.0,
        'missing_indices': missing,
    }
    out = os.path.join(results_dir, 'code_pass_at_1.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        raise SystemExit('Usage: eval_codecontests_pass1.py <dir> <label> [...]')
    all_results = {}
    for i in range(0, len(args), 2):
        d, label = args[i], args[i + 1]
        all_results[label] = eval_dir(d, label)
        print(json.dumps(all_results[label], indent=2))
    print('\n=== SUMMARY ===')
    print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
