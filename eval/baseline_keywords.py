#!/usr/bin/env python3
"""关键词规则基线：只看 query 文本，给模型提供一个下限。

用法：python3 eval/baseline_keywords.py eval/testset_v0.jsonl > preds_baseline.jsonl
"""
import json
import re
import sys

KW = {
    '价格': ['优惠明细', '降价', '价格区', '领取'],
    '等大促': ['双11', '大促', '开售提醒'],
    '凑单': ['满减', '凑单'],
    '比价': ['同类商品', '哪个好', '来回切换', '另一品牌'],
    '品质信任': ['差评', '买家秀', '资质', '正品'],
    '规格适配': ['对照表', '切换为', '怎么选', '相关评价'],
    '物流售后': ['配送时效', '退换货', '收货地址切换', '多久能到'],
    '流程摩擦': ['支付失败', '超出配送范围', '反复点击支付'],
}
ACTION = {'价格': '定向券', '等大促': '价保', '凑单': '凑单推荐', '比价': '对比卡片', '品质信任': '好评与保障',
          '规格适配': '尺码与规格助手', '物流售后': '时效与运费险', '流程摩擦': '支付引导'}


def predict(case):
    hits = {c: [] for c in KW}
    for line in case['query'].splitlines():
        m = re.match(r'#(\d+) ', line)
        if not m:
            continue
        for c, kws in KW.items():
            if any(k in line for k in kws):
                hits[c].append(int(m.group(1)))
    ranked = sorted(hits, key=lambda c: -len(hits[c]))
    top, sec = ranked[0], ranked[1]
    if not hits[top]:
        return dict(id=case['id'], act=False, primary='无', secondary=None, confidence=0.5,
                    evidence=[], action='无', uncertain=True)
    return dict(id=case['id'], act=True, primary=top, secondary=sec if hits[sec] else None,
                confidence=min(0.9, 0.4 + 0.15 * len(hits[top])), evidence=hits[top],
                action=ACTION[top], uncertain=len(hits[top]) < 2)


def main():
    with open(sys.argv[1], encoding='utf-8') as f:
        for l in f:
            print(json.dumps(predict(json.loads(l)), ensure_ascii=False))


if __name__ == '__main__':
    main()
