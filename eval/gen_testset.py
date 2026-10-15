#!/usr/bin/env python3
"""生成智能挽单「出手判断 + 顾虑归因」测试集（合成数据）。

每条用例 = 一段截至决策时刻的会话 + 画像 + 商品上下文，附带标准答案。
用法：
  python3 eval/gen_testset.py                       # 输出 eval/testset_v0.jsonl
  python3 eval/gen_testset.py --ablate profile      # 去掉画像块，做数据消融
  python3 eval/gen_testset.py --ablate item --out eval/testset_v0_no_item.jsonl
"""
import argparse
import json
import random
from pathlib import Path

CONCERNS = ['价格', '等大促', '凑单', '比价', '品质信任', '规格适配', '物流售后', '流程摩擦']
ACTIONS = ['定向券', '价格事实', '价保', '凑单推荐', '对比卡片', '好评与保障',
           '尺码与规格助手', '时效与运费险', '支付引导', '无']
ACTION_OK = {
    '价格': ['定向券', '价格事实'],
    '等大促': ['价保'],
    '凑单': ['凑单推荐'],
    '比价': ['对比卡片'],
    '品质信任': ['好评与保障'],
    '规格适配': ['尺码与规格助手'],
    '物流售后': ['时效与运费险'],
    '流程摩擦': ['支付引导'],
}

ITEMS = {
    '服饰鞋包': [
        dict(name='轻量缓震跑鞋', price=529, avg=539, low=499, rating=4.7, ship='48 小时发货', kind='鞋码', opts=['40', '41', '42']),
        dict(name='羊毛混纺大衣', price=899, avg=869, low=799, rating=4.5, ship='72 小时发货', kind='尺码', opts=['S', 'M', 'L']),
        dict(name='通勤双肩包', price=259, avg=249, low=229, rating=4.8, ship='24 小时发货', kind='颜色', opts=['黑色', '灰色', '藏青']),
        dict(name='加绒卫衣', price=199, avg=189, low=169, rating=4.6, ship='48 小时发货', kind='尺码', opts=['M', 'L', 'XL']),
    ],
    '3C数码': [
        dict(name='降噪蓝牙耳机', price=399, avg=372, low=359, rating=4.6, ship='24 小时发货', kind='版本', opts=['标准版', 'Pro 版']),
        dict(name='65W 氮化镓充电器', price=129, avg=125, low=109, rating=4.8, ship='24 小时发货', kind='颜色', opts=['白色', '黑色']),
        dict(name='轻薄笔记本电脑', price=5999, avg=5899, low=5699, rating=4.7, ship='48 小时发货', kind='配置', opts=['16G+512G', '32G+1T']),
        dict(name='智能手表', price=1299, avg=1259, low=1199, rating=4.5, ship='48 小时发货', kind='表盘尺寸', opts=['41mm', '45mm']),
    ],
    '快消': [
        dict(name='浓缩洗衣液 3kg', price=59.9, avg=59.9, low=55.9, rating=4.9, ship='次日达', kind='规格', opts=['3kg']),
        dict(name='抽纸 24 包', price=49.9, avg=52.9, low=45.9, rating=4.8, ship='次日达', kind='规格', opts=['24 包']),
    ],
    '药品': [
        dict(name='感冒颗粒 10 袋', price=29.9, avg=29.9, low=27.9, rating=4.7, ship='24 小时发货', kind='规格', opts=['10 袋']),
    ],
}
NOISE_ITEMS = ['保温杯', '手机壳', '零食大礼包', '桌面收纳盒', '运动袜 5 双']
CITIES = ['拉萨', '乌鲁木齐', '喀什', '西宁']


def fmt_price(p):
    return f'{p:g}'


def alt_of(it):
    return dict(name='另一品牌' + it['name'], price=round(it['price'] * 0.85))


# 每个顾虑的信号模板：返回 (page, 描述)。
def concern_signals(c, it, r):
    alt = alt_of(it)
    o = it['opts']
    a, b = (o[0], o[1]) if len(o) > 1 else (o[0], o[0])
    pool = {
        '价格': [
            ('商详', f'展开优惠明细，领取 ¥{r.choice([5, 10, 20])} 店铺券但未使用'),
            ('商详', f'离开后再次回到该商品，停留在价格区 {r.randint(8, 30)} 秒'),
            ('搜索', f'搜索「{it["name"]} 降价」'),
            ('商详', '点击「降价提醒」按钮'),
        ],
        '等大促': [
            ('搜索', f'搜索「双11 {it["name"]}」'),
            ('会场', f'浏览双11 大促会场预告页 {r.randint(20, 90)} 秒'),
            ('商详', '查看「大促到手价」预告浮层'),
            ('商详', '点击「开售提醒」'),
        ],
        '凑单': [
            ('购物车', '查看店铺满减规则（满 300 减 30）'),
            ('购物车', '反复勾选、取消购物车里的另一件小商品'),
            ('凑单页', f'进入「凑单专区」浏览 {r.randint(15, 60)} 秒后返回购物车'),
        ],
        '比价': [
            ('商详', f'进入同类商品「{alt["name"]}」（¥{alt["price"]}）商详，停留 {r.randint(20, 80)} 秒'),
            ('搜索', f'搜索「{it["name"]} 和 {alt["name"]} 哪个好」'),
            ('商详', '在两款同类商品商详之间来回切换 3 次'),
            ('购物车', f'把「{alt["name"]}」也加入购物车'),
        ],
        '品质信任': [
            ('商详', f'评价区筛选「差评」，浏览 {r.randint(5, 20)} 条'),
            ('商详', f'查看买家秀图片 {r.randint(6, 25)} 张'),
            ('店铺', '点击店铺资质与店铺评分'),
            ('商详', '在「问大家」搜索「是不是正品」'),
        ],
        '规格适配': [
            ('商详', f'选择{it["kind"]} {a} → 切换为 {b} → 又切回 {a}'),
            ('商详', f'打开{it["kind"]}对照表，停留 {r.randint(15, 40)} 秒'),
            ('商详', f'评价区筛选「{it["kind"]}」相关评价'),
            ('商详', f'在「问大家」提问「{it["kind"]}怎么选」'),
        ],
        '物流售后': [
            ('商详', f'查看配送时效，显示预计 {r.randint(4, 7)} 天送达'),
            ('商详', '查看退换货与运费险说明'),
            ('结算', f'把收货地址切换到{r.choice(CITIES)}'),
            ('商详', '在「问大家」搜索「多久能到」'),
        ],
        '流程摩擦': [
            ('结算', '提交订单后支付失败（银行卡验证未通过）'),
            ('结算', '切换支付方式后再次支付失败'),
            ('结算', '修改收货地址，提示超出配送范围'),
            ('结算', f'结算页停留 {r.randint(40, 120)} 秒，反复点击支付按钮'),
        ],
    }
    return pool[c]


def noise_events(r, k):
    pool = [
        ('首页', f'浏览首页推荐 {r.randint(5, 40)} 秒'),
        ('商详', f'点进无关商品「{r.choice(NOISE_ITEMS)}」，停留 {r.randint(3, 15)} 秒'),
        ('订单', '查看一笔已有订单的物流'),
        ('推荐流', f'观看短视频 {r.randint(10, 60)} 秒'),
    ]
    return [(p, d, None) for p, d in r.sample(pool, k)]


def make_profile(r, tier=None, price_sens=None, extras=None):
    tier = tier or r.choice(['高活跃', '中活跃', '低活跃', '新客'])
    return dict(tier=tier, price_sens=price_sens or r.choice(['高', '中', '低']), extras=extras or [])


EXIT = {
    '商详': ('商详', '离开商详，未加购', '商详离开未加购'),
    '购物车': ('购物车', '离开购物车，未勾选结算', '购物车离开'),
    '结算': ('结算', '退出结算页，订单未支付', '结算页退出'),
}


def build_flow(r, it, groups, stage, noise_k=0, add_cart=True):
    """groups: [(concern, n_signals)]；stage: 决策时刻所在页面。"""
    ev = [(r.choice(['搜索', '推荐流', '直播间']), f'进入「{it["name"]}」商详', None),
          ('商详', f'浏览商详 {r.randint(20, 90)} 秒', None)]
    sig = []
    for c, n in groups:
        for p, d in r.sample(concern_signals(c, it, r), n):
            sig.append((p, d, c))
    r.shuffle(sig)
    # 结算类、购物车类信号要放在进入对应页面之后
    pre = [e for e in sig if e[0] not in ('购物车', '结算', '凑单页')]
    post = [e for e in sig if e[0] in ('购物车', '结算', '凑单页')]
    ev += pre
    if stage in ('购物车', '结算') or post:
        if add_cart:
            ev.append(('商详', '加入购物车', None))
        ev.append(('购物车', f'进入购物车，停留 {r.randint(5, 30)} 秒', None))
        if stage == '结算' or any(e[0] == '结算' for e in post):
            ev.append(('结算', '进入结算页', None))
            stage = '结算'
        elif stage == '商详':
            stage = '购物车'
    ev += post
    for n in noise_events(r, noise_k):
        ev.insert(r.randint(2, len(ev)), n)
    p, d, dp = EXIT[stage]
    ev.append((p, d, None))
    return ev, dp


def finalize(r, cid, cat, it, profile, ev, dp, gold, layer, tags, extra_ctx=None):
    h, m = r.randint(8, 23), r.randint(0, 40)
    events, evidence = [], {}
    for i, (p, d, c) in enumerate(ev, 1):
        m += r.randint(0, 2)
        events.append(dict(i=i, ts=f'{h:02d}:{min(m, 59):02d}', page=p, desc=d))
        if c:
            evidence.setdefault(c, []).append(i)
    gold['evidence'] = evidence.get(gold['primary'], [])
    gold['evidence_by_concern'] = evidence
    ctx = dict(category=cat, **{k: it[k] for k in ('name', 'price', 'avg', 'low', 'rating', 'ship')})
    ctx['extra'] = extra_ctx or []
    return dict(id=cid, layer=layer, tags=tags, profile=profile, item=ctx,
                events=events, decision_point=dp, gold=gold)


def render(case, ablate):
    lines = []
    if 'profile' not in ablate:
        p = case['profile']
        s = f'活跃度：{p["tier"]}；价格敏感度：{p["price_sens"]}'
        if p['extras']:
            s += '；' + '；'.join(p['extras'])
        lines += ['【用户画像】', s, '']
    if 'item' not in ablate:
        it = case['item']
        s = (f'{it["name"]}（{it["category"]}），当前价 ¥{fmt_price(it["price"])}，近 30 天均价 ¥{fmt_price(it["avg"])}，'
             f'近 30 天最低 ¥{fmt_price(it["low"])}；评分 {it["rating"]}；{it["ship"]}')
        if it['extra']:
            s += '；' + '；'.join(it['extra'])
        lines += ['【商品上下文】', s, '']
    lines.append(f'【会话行为】截至决策时刻，共 {len(case["events"])} 条')
    lines += [f'#{e["i"]} {e["ts"]} [{e["page"]}] {e["desc"]}' for e in case['events']]
    lines += ['', f'【决策时刻】#{len(case["events"])} {case["decision_point"]}']
    return '\n'.join(lines)


def gold_act(primary, secondary_ok=(), action_ok=None, must_not=None, allow_uncertain=False):
    must = list(must_not or [])
    if primary != '价格' and '定向券' not in must:
        must.append('定向券')
    return dict(act=True, act_reason='顾虑可挽', primary=primary, secondary_ok=list(secondary_ok),
                action_ok=action_ok or ACTION_OK[primary], must_not=must, allow_uncertain=allow_uncertain)


def gold_no(reason, primary='无', secondary_ok=(), score_primary=False):
    return dict(act=False, act_reason=reason, primary=primary, secondary_ok=list(secondary_ok),
                action_ok=['无'], must_not=['定向券'], allow_uncertain=False, score_primary=score_primary)


def pick_item(r, c):
    if c == '规格适配':
        cat = r.choice(['服饰鞋包', '3C数码'])
        return cat, dict(r.choice([i for i in ITEMS[cat] if len(i['opts']) > 1]))
    cat = r.choice(['服饰鞋包', '3C数码'])
    return cat, dict(r.choice(ITEMS[cat]))


def extra_ctx_for(c, it, r):
    if c == '等大促':
        return [f'距双11 开售还有 {r.randint(3, 9)} 天']
    if c == '凑单':
        return ['店铺满 300 减 30', f'购物车合计 ¥{r.randint(262, 292)}']
    if c == '物流售后':
        return ['支持运费险']
    return []


def stage_for(c, r):
    if c == '流程摩擦':
        return '结算'
    if c == '凑单':
        return '购物车'
    return r.choice(['商详', '购物车', '购物车'])


def generate(seed):
    r = random.Random(seed)
    cases, n = [], 0

    def nid():
        nonlocal n
        n += 1
        return f'C{n:03d}'

    # L1 清晰单顾虑 / L2 加干扰 / L4 稀疏
    for c in CONCERNS:
        for layer, cnt in (('L1_清晰', 6), ('L2_干扰', 6), ('L4_稀疏', 2)):
            for _ in range(cnt):
                cat, it = pick_item(r, c)
                st = stage_for(c, r)
                if layer == 'L4_稀疏':
                    prof = make_profile(r, tier=r.choice(['低活跃', '新客']))
                    ev, dp = build_flow(r, it, [(c, 1)], st)
                    ev = [e for e in ev if not e[1].startswith('浏览商详')]
                    g = gold_act(c, allow_uncertain=True)
                else:
                    ps = '高' if c == '价格' else None
                    prof = make_profile(r, price_sens=ps)
                    groups = [(c, r.randint(2, 3))]
                    sec = []
                    if layer == 'L2_干扰' and r.random() < 0.5:
                        other = r.choice([x for x in CONCERNS if x not in (c, '流程摩擦', '凑单')])
                        groups.append((other, 1))
                        sec = [other]
                    ev, dp = build_flow(r, it, groups, st, noise_k=2 if layer == 'L2_干扰' else 0)
                    g = gold_act(c, secondary_ok=sec)
                cases.append(finalize(r, nid(), cat, it, prof, ev, dp, g, layer, [c],
                                      extra_ctx_for(c, it, r)))

    # L3 双顾虑：主顾虑信号数严格多于次顾虑
    pairs = [(a, b) for a in CONCERNS for b in CONCERNS
             if a != b and '流程摩擦' not in (a, b) and '凑单' not in (a, b)]
    for a, b in r.sample(pairs, 30):
        cat, it = pick_item(r, '规格适配' if '规格适配' in (a, b) else a)
        prof = make_profile(r)
        ev, dp = build_flow(r, it, [(a, 3), (b, 1)], stage_for(a, r), noise_k=r.randint(0, 1))
        cases.append(finalize(r, nid(), cat, it, prof, ev, dp, gold_act(a, secondary_ok=[b]),
                              'L3_双顾虑', [a, b], extra_ctx_for(a, it, r) + extra_ctx_for(b, it, r)))

    # 不出手：必买
    for _ in range(14):
        it = dict(r.choice(ITEMS['快消']))
        k = r.randint(4, 8)
        prof = make_profile(r, tier='高活跃', price_sens='低',
                            extras=[f'该品牌复购 {k} 次，平均间隔 {r.randint(30, 50)} 天，距上次购买 {r.randint(28, 48)} 天'])
        ev = [('搜索', f'搜索常买品牌的「{it["name"]}」', None),
              ('商详', f'直接进入常买款商详，停留 {r.randint(3, 8)} 秒', None),
              ('商详', '加入购物车', None), ('结算', '进入结算页', None),
              ('结算', f'结算页停留 {r.randint(2, 5)} 秒后退出，订单未支付', None)]
        cases.append(finalize(r, nid(), '快消', it, prof, ev, '结算页退出', gold_no('必买'), 'N_不出手', ['必买']))

    # 不出手：弱需求
    for _ in range(14):
        cat, it = pick_item(r, '价格')
        prof = make_profile(r)
        ev = [('推荐流', f'从推荐流点进「{it["name"]}」商详', None),
              ('商详', f'浏览商详 {r.randint(2, 6)} 秒', None)]
        ev += noise_events(r, r.randint(0, 1))
        ev.append(('商详', '离开商详，未加购', None))
        cases.append(finalize(r, nid(), cat, it, prof, ev, '商详离开未加购', gold_no('弱需求'), 'N_不出手', ['弱需求']))

    # 不出手：已买替代品
    for _ in range(10):
        cat, it = pick_item(r, '比价')
        alt = alt_of(it)
        prof = make_profile(r)
        ev, _ = build_flow(r, it, [('比价', 2)], '购物车')
        ev = ev[:-1] + [('结算', f'对「{alt["name"]}」下单并支付成功', None),
                        ('购物车', f'回到购物车，「{it["name"]}」仍未结算', None),
                        ('购物车', '离开购物车', None)]
        cases.append(finalize(r, nid(), cat, it, prof, ev, '购物车离开',
                              gold_no('已购替代', secondary_ok=['比价']), 'N_不出手', ['已购替代']))

    # 不出手：频控（仍需识别顾虑）
    for _ in range(10):
        c = r.choice(['价格', '比价', '品质信任', '规格适配'])
        cat, it = pick_item(r, c)
        prof = make_profile(r, extras=[f'近 7 天收到挽留触达 {r.randint(3, 5)} 次，全部点击关闭'])
        ev, dp = build_flow(r, it, [(c, 2)], stage_for(c, r))
        cases.append(finalize(r, nid(), cat, it, prof, ev, dp,
                              gold_no('频控', primary=c, score_primary=True), 'N_不出手', ['频控', c]))

    # 不出手：敏感类目
    for _ in range(6):
        it = dict(ITEMS['药品'][0])
        prof = make_profile(r)
        ev, dp = build_flow(r, it, [('价格', 2)], '购物车')
        cases.append(finalize(r, nid(), '药品', it, prof, ev, dp, gold_no('敏感类目'), 'N_不出手', ['敏感类目']))

    # L5 陷阱：画像说价格敏感，行为全是规格
    for _ in range(10):
        cat, it = pick_item(r, '规格适配')
        prof = make_profile(r, price_sens='高', extras=['近 30 天领券 6 次、核销 4 次'])
        ev, dp = build_flow(r, it, [('规格适配', 3)], stage_for('规格适配', r))
        g = gold_act('规格适配', must_not=['价格'])
        cases.append(finalize(r, nid(), cat, it, prof, ev, dp, g, 'L5_陷阱', ['画像与行为冲突']))

    # L5 陷阱：已是 30 天最低价
    for _ in range(8):
        cat, it = pick_item(r, '价格')
        it['price'] = it['low']
        prof = make_profile(r, price_sens='高')
        ev, dp = build_flow(r, it, [('价格', 3)], stage_for('价格', r))
        g = gold_act('价格', action_ok=['价格事实'], must_not=['定向券'])
        cases.append(finalize(r, nid(), cat, it, prof, ev, dp, g, 'L5_陷阱', ['已是低价']))

    # L5 陷阱：加购等券型
    for _ in range(8):
        cat, it = pick_item(r, '价格')
        prof = make_profile(r, price_sens='高', extras=[f'近 90 天有 {r.randint(4, 7)} 次「加购后收到挽留券才下单」'])
        ev, dp = build_flow(r, it, [('价格', 2)], '购物车')
        g = gold_act('价格', action_ok=['价格事实', '价保'], must_not=['定向券'])
        cases.append(finalize(r, nid(), cat, it, prof, ev, dp, g, 'L5_陷阱', ['加购等券']))

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--ablate', default='', help='逗号分隔：profile,item')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ablate = {x for x in a.ablate.split(',') if x}
    suffix = ''.join(f'_no_{x}' for x in sorted(ablate))
    out = Path(a.out or Path(__file__).parent / f'testset_v0{suffix}.jsonl')
    cases = generate(a.seed)
    with out.open('w', encoding='utf-8') as f:
        for c in cases:
            c['query'] = render(c, ablate)
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f'{len(cases)} cases -> {out}')


if __name__ == '__main__':
    main()
