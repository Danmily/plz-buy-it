#!/usr/bin/env python3
"""给预测结果逐条打分并汇总。

预测 jsonl 每行（touchpoint、reply 可选，给了才评）：
  {"id": "T0001", "act": true, "primary": "价格", "secondary": "比价", "confidence": 0.7,
   "evidence": [3, 5], "action": "定向券", "touchpoint": "push", "uncertain": false, "reply": "..."}

用法：python3 eval/score.py eval/testset_v1.jsonl preds.jsonl
"""
import json
import re
import sys
from collections import defaultdict

LOWER_IS_BETTER = {'踩禁区', '误发券', '稀疏样本过度自信', '回复暴露推断或营销腔', '不该回复却回复', '输出无法解析'}
METRICS = ['出手判断', '主顾虑 top1', '主顾虑 top2', '动作正确', '触点正确', '证据召回', '证据编号合法',
           '踩禁区', '误发券', '稀疏样本过度自信', '回复数字可溯源', '回复暴露推断或营销腔', '不该回复却回复',
           '输出无法解析']
# 软指标不计入「全对」：证据召回低于一半才算错
SOFT = {'主顾虑 top2', '触点正确', '证据召回', '稀疏样本过度自信'}
EXPOSE = ['犹豫', '看了很久', '注意到', '检测到', '发现您', '我们看到', '纠结']
HYPE = ['亲', '专属', '错过', '史上最低', '全网最低', '闭眼入', '！！', '赶紧', '手慢无']


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def check_reply(reply, query):
    nums = re.findall(r'\d+(?:\.\d+)?', reply)
    grounded = all(n in query for n in nums)
    bad = [w for w in EXPOSE + HYPE if w in reply]
    if '最低价' in reply and '最低' not in query:
        bad.append('最低价')
    return grounded, bad


def score_case(c, p):
    g = c['gold']
    m, fails = {}, []
    parsed = bool(p) and not p.get('_error')
    m['输出无法解析'] = not parsed
    p = p or {}
    m['出手判断'] = p.get('act') == g['act']
    if g['act'] or g.get('score_primary'):
        m['主顾虑 top1'] = p.get('primary') == g['primary']
        m['主顾虑 top2'] = g['primary'] in (p.get('primary'), p.get('secondary'))
    if g['act']:
        m['动作正确'] = p.get('action') in g['action_ok']
        if 'touchpoint' in p:
            m['触点正确'] = p.get('touchpoint') in g['touch_ok']
    ev = p.get('evidence') or []
    n_ev = len(c['events'])
    m['证据编号合法'] = all(isinstance(i, int) and 1 <= i <= n_ev for i in ev)
    if g['act'] and g['evidence']:
        m['证据召回'] = len(set(ev) & set(g['evidence'])) / len(g['evidence'])
    m['踩禁区'] = p.get('primary') in g['must_not'] or p.get('action') in g['must_not']
    m['误发券'] = p.get('action') == '定向券' and '定向券' in g['must_not']
    if g['allow_uncertain']:
        m['稀疏样本过度自信'] = (not p.get('uncertain')) and (p.get('confidence') or 0) > 0.7
    reply = (p.get('reply') or '').strip()
    if reply:
        if not g['act']:
            m['不该回复却回复'] = True
        grounded, bad = check_reply(reply, c['query'])
        m['回复数字可溯源'] = grounded
        m['回复暴露推断或营销腔'] = bool(bad)
    elif not g['act'] and 'reply' in p:
        m['不该回复却回复'] = False
    for k, v in m.items():
        if k in SOFT:
            bad_v = k == '证据召回' and v < 0.5
        else:
            bad_v = (v is True) if k in LOWER_IS_BETTER else (v is False)
        if bad_v:
            fails.append(k)
    return m, fails


def group_keys(c):
    g = c['gold']
    keys = ['全部', f'层:{c["layer"]}', f'场景:{c["scenario"].split(":")[0]}', f'渠道:{c["channel"]}',
            f'品类:{c["item"]["category"]}', f'活跃度:{c["profile"]["tier"]}']
    keys.append(f'顾虑:{g["primary"]}' if g['act'] else f'不出手:{g["act_reason"]}')
    keys += [f'标签:{t}' for t in c['tags']]
    return keys


def score(cases, preds):
    pm = {p.get('id'): p for p in preds}
    results = []
    agg = defaultdict(lambda: defaultdict(list))
    for c in cases:
        p = pm.get(c['id'])
        m, fails = score_case(c, p)
        results.append(dict(id=c['id'], scenario=c['scenario'], layer=c['layer'], channel=c['channel'],
                            gold=c['gold'], pred=p, metrics=m, fails=fails))
        for k in group_keys(c):
            agg[k]['_n'].append(1)
            agg[k]['_全对'].append(float(not fails))
            for name, v in m.items():
                agg[k][name].append(float(v))
    summary = {k: {name: (sum(v) / len(v) if name != '_n' else len(v)) for name, v in g.items()} for k, g in agg.items()}
    return results, summary


def confusion(results):
    conf = defaultdict(lambda: defaultdict(int))
    for r in results:
        g, p = r['gold'], r['pred'] or {}
        if g['act'] or g.get('score_primary'):
            conf[g['primary']][p.get('primary', '缺失')] += 1
    return conf


def act_breakdown(results):
    out = defaultdict(lambda: [0, 0])
    for r in results:
        key = r['gold']['act_reason']
        out[key][0] += 1
        out[key][1] += int((r['pred'] or {}).get('act') == r['gold']['act'])
    return out


def print_table(summary, keys=None):
    cols = [c for c in METRICS if any(c in summary[k] for k in summary)]
    keys = keys or sorted(summary, key=lambda k: (k != '全部', k))
    print('分组'.ljust(22) + '样本'.rjust(5) + '全对'.rjust(7) + ''.join(c[:6].rjust(9) for c in cols))
    for k in keys:
        s = summary[k]
        row = k[:21].ljust(22) + str(int(s['_n'])).rjust(5) + f'{s["_全对"]:.0%}'.rjust(7)
        for c in cols:
            row += (f'{s[c]:.0%}' if c in s else '-').rjust(9)
        print(row)


def main():
    cases, preds = load(sys.argv[1]), load(sys.argv[2])
    _, summary = score(cases, preds)
    print_table(summary, [k for k in sorted(summary, key=lambda k: (k != '全部', k)) if not k.startswith(('品类', '活跃度', '标签'))])
    print('\n踩禁区、误发券、过度自信、暴露推断、不该回复、无法解析越低越好，其余越高越好。')


if __name__ == '__main__':
    main()
