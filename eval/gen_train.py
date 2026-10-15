#!/usr/bin/env python3
"""生成训练数据（train split，与测试集商品、表述不重叠）。

按某次评测的错误分布加量：
  python3 eval/gen_train.py --from-run eval/runs/<run> --n 2000
不看评测结果，均匀生成：
  python3 eval/gen_train.py --n 2000
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import scenarios as S  # noqa: E402

HERE = Path(__file__).parent

REPLY = {
    '定向券': '这款现在 ¥{price}，结算前可以看看店铺券和到手价。',
    '价格事实': '这款当前 ¥{price}，近 30 天均价 ¥{avg}、最低 ¥{low}。',
    '分期信息': '这款支持分期付款，结算时可以选择期数。',
    '价保': '可以先设置开售提醒，开售时第一时间通知您。',
    '凑单推荐': '可以去凑单专区看看，凑够门槛就能用满减。',
    '对比卡片': '可以用对比功能把两款的到手价、评分和发货时间放在一起看。',
    '好评与保障': '这款评分 {rating}，可以看看带图好评和店铺资质。',
    '尺码与规格助手': '可以用规格推荐工具，参考买过的人的反馈来选。',
    '时效与运费险': '这款{ship}，退换货规则可以在商详查看。',
    '支付引导': '支付没成功的话，可以换一种支付方式再试一次。',
}


def answer_with_reply(case):
    a = S.gold_answer(case)
    a['reply'] = ''
    if case['channel'] == 'chat' and a['act']:
        it = case['item']
        a['reply'] = REPLY[a['action']].format(price=S.fmt_price(it['price']), avg=S.fmt_price(it['avg']),
                                               low=S.fmt_price(it['low']), rating=it['rating'], ship=it['ship'])
    return a


def plan_from_summary(summary, suggestions, n_total):
    """基础量 1 份，按场景错误率和诊断建议加量。"""
    weights = {}
    for key in S.DEFAULT_TEST_PLAN:
        name, _, arg = key.partition(':')
        err = 1 - summary.get(f'场景:{name}', {}).get('_全对', 1)
        if arg:
            err = max(err, 1 - summary.get(f'顾虑:{arg}', {}).get('_全对', 1))
        weights[key] = 1 + 6 * err
    pair_w = {p: 0.5 for p in S.PAIRS}
    for s in suggestions or []:
        for a in s['augment']:
            if a in weights:
                weights[a] += 4 * min(1.0, s['severity'])
            elif a in pair_w:
                pair_w[a] += 4 * min(1.0, s['severity'])
    return weights, pair_w


def allocate(weights, pair_w, n_total, pair_share=0.15):
    n_pairs_total = int(n_total * pair_share)
    n_single = n_total - n_pairs_total
    tot = sum(weights.values())
    plan = {k: max(1, round(n_single * w / tot)) for k, w in weights.items()}
    ptot = sum(pair_w.values())
    pairs = {k: max(1, round(n_pairs_total / 2 * w / ptot)) for k, w in pair_w.items()}
    return plan, pairs


def leak_check(train_cases):
    names = {it['name'] for cat in S.ITEMS['test'].values() for it in cat}
    brands = set(S.ALT_BRANDS['test'])
    hits = [c['id'] for c in train_cases if any(n in c['query'] for n in names | brands)]
    return '通过' if not hits else f'有 {len(hits)} 条包含测试集商品或品牌'


def make_training(out_dir, n_total=2000, seed=None, summary=None, suggestions=None):
    seed = seed if seed is not None else random.randrange(1 << 30)
    weights, pair_w = plan_from_summary(summary or {}, suggestions, n_total)
    plan, pairs = allocate(weights, pair_w, n_total)
    cases = S.generate('train', seed, plan, prefix='R')
    pr = random.Random(seed + 1)
    pair_cases = []
    for pname, k in pairs.items():
        sub = S.generate_pairs('train', pr.randrange(1 << 30), k, prefix=f'P{pname[5:8]}')
        pair_cases += [c for c in sub if pname in c['tags']]
    allc = cases + pair_cases
    for c in allc:
        c['query'] = S.render(c)
    system = (HERE / 'prompt.md').read_text(encoding='utf-8')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sft.jsonl', 'w', encoding='utf-8') as f:
        for c in allc:
            msg = [{'role': 'system', 'content': system}, {'role': 'user', 'content': c['query']},
                   {'role': 'assistant', 'content': json.dumps(answer_with_reply(c), ensure_ascii=False)}]
            f.write(json.dumps({'id': c['id'], 'messages': msg}, ensure_ascii=False) + '\n')
    with open(out / 'cases.jsonl', 'w', encoding='utf-8') as f:
        for c in allc:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    leak = leak_check(allc)
    by_scen = Counter(c['scenario'] for c in allc)
    info = dict(dir=str(out), total=len(allc), pairs=len(pair_cases), seed=seed, leak_check=leak,
                plan=plan, pair_plan=pairs, top_alloc=by_scen.most_common(5))
    (out / 'plan.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-run', help='评测结果目录（读取 metrics.json）')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--out')
    a = ap.parse_args()
    summary = suggestions = None
    if a.from_run:
        m = json.loads((Path(a.from_run) / 'metrics.json').read_text(encoding='utf-8'))
        summary, suggestions = m['summary'], m['suggestions']
    out = a.out or (Path(a.from_run) / 'train_data' if a.from_run else HERE / 'train_data' / 'uniform')
    info = make_training(out, a.n, a.seed, summary, suggestions)
    print(json.dumps({k: info[k] for k in ('dir', 'total', 'pairs', 'leak_check', 'top_alloc')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
