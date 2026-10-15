#!/usr/bin/env python3
"""给预测结果打分。

预测文件为 jsonl，每行：
  {"id": "C001", "act": true, "primary": "价格", "secondary": "比价",
   "confidence": 0.7, "evidence": [3, 5], "action": "定向券", "uncertain": false}

用法：python3 eval/score.py eval/testset_v0.jsonl preds.jsonl
"""
import json
import sys
from collections import defaultdict


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def score(cases, preds):
    pm = {p['id']: p for p in preds}
    groups = defaultdict(lambda: defaultdict(list))
    missing = 0
    for c in cases:
        g, p = c['gold'], pm.get(c['id'])
        if p is None:
            missing += 1
            p = {}
        n_ev = len(c['events'])
        m = {}
        m['出手判断'] = p.get('act') == g['act']
        if g['act'] or g.get('score_primary'):
            m['主顾虑 top1'] = p.get('primary') == g['primary']
            m['主顾虑 top2'] = g['primary'] in (p.get('primary'), p.get('secondary'))
        if g['act']:
            m['动作正确'] = p.get('action') in g['action_ok']
        ev = p.get('evidence') or []
        m['证据编号合法'] = all(isinstance(i, int) and 1 <= i <= n_ev for i in ev)
        if g['evidence'] and g['act']:
            m['证据召回'] = len(set(ev) & set(g['evidence'])) / len(g['evidence'])
        m['踩禁区'] = p.get('primary') in g['must_not'] or p.get('action') in g['must_not']
        m['误发券'] = p.get('action') == '定向券' and '定向券' in g['must_not']
        if g['allow_uncertain']:
            m['稀疏样本过度自信'] = (not p.get('uncertain')) and (p.get('confidence') or 0) > 0.7
        keys = ['全部', c['layer']] + [f'标签:{t}' for t in c['tags']]
        for k in keys:
            for name, v in m.items():
                groups[k][name].append(float(v))
    return groups, missing


def main():
    cases, preds = load(sys.argv[1]), load(sys.argv[2])
    groups, missing = score(cases, preds)
    if missing:
        print(f'⚠ {missing} 条用例没有预测，按全错计')
    cols = ['出手判断', '主顾虑 top1', '主顾虑 top2', '动作正确', '证据召回', '证据编号合法',
            '踩禁区', '误发券', '稀疏样本过度自信']
    print('分组'.ljust(16) + '样本'.rjust(5) + ''.join(c.rjust(10) for c in cols))
    order = ['全部'] + sorted(k for k in groups if k != '全部' and not k.startswith('标签:')) \
        + sorted(k for k in groups if k.startswith('标签:'))
    for k in order:
        g = groups[k]
        n = len(g['出手判断'])
        row = k[:15].ljust(16) + str(n).rjust(5)
        for c in cols:
            v = g.get(c)
            row += (f'{sum(v) / len(v):.0%}' if v else '-').rjust(10)
        print(row)
    print('\n踩禁区、误发券、稀疏样本过度自信越低越好，其余越高越好。')


if __name__ == '__main__':
    main()
