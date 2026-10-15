#!/usr/bin/env python3
"""生成挽单测试集（合成数据，test split）。

  python3 eval/gen_testset.py                      # eval/testset_v1.jsonl
  python3 eval/gen_testset.py --ablate profile     # eval/testset_v1_no_profile.jsonl，做数据消融
  python3 eval/gen_testset.py --ablate item
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import DEFAULT_TEST_PLAN, generate, render  # noqa: E402

VERSION = 'v1'
SEED = 20261016


def build(ablate=(), seed=SEED):
    cases = generate('test', seed, DEFAULT_TEST_PLAN)
    for c in cases:
        c['query'] = render(c, ablate)
    return cases


def path_for(ablate=()):
    suffix = ''.join(f'_no_{x}' for x in sorted(ablate))
    return Path(__file__).parent / f'testset_{VERSION}{suffix}.jsonl'


def write(cases, out):
    with open(out, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablate', default='', help='逗号分隔：profile,item')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out')
    a = ap.parse_args()
    ablate = tuple(x for x in a.ablate.split(',') if x)
    out = Path(a.out) if a.out else path_for(ablate)
    cases = build(ablate, a.seed)
    write(cases, out)
    print(f'{len(cases)} cases -> {out}')


if __name__ == '__main__':
    main()
