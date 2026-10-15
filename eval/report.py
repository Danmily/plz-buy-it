"""评测诊断与 Markdown 报告。"""
import math
from collections import defaultdict

from score import LOWER_IS_BETTER, METRICS

# (分组, 指标) → 诊断。augment 是训练数据要加量的场景（gen_train 使用）。
RULES = [
    ('场景:no_substitute', '出手判断', '没认出"已经买了替代品"',
     '模型没把会话里对其他商品的支付和焦点商品联系起来，买完了还在挽留',
     {'规则': '决策前加硬规则：本次会话已支付同类商品 → 停止该商品的所有挽留',
      '特征': '增加"会话内同类目已支付"特征',
      '训练数据': '增补已购替代样本和"买了 / 没买"最小对比对'}, ['no_substitute', 'pair_substitute']),
    ('场景:no_freq', '出手判断', '频控没生效',
     '画像里写了近期多次关闭挽留，模型仍然出手',
     {'规则': '频控放到规则层前置执行，不交给模型判断',
      '训练数据': '增补频控样本和"有 / 无频控"最小对比对'}, ['no_freq', 'pair_freq']),
    ('场景:no_sensitive', '出手判断', '敏感类目仍在挽留',
     '药品、母婴类被当成普通商品处理，可能带来合规风险',
     {'规则': '敏感类目白名单前置过滤，命中直接不出手', '训练数据': '增补敏感类目样本'}, ['no_sensitive']),
    ('场景:chat_optout', '出手判断', '用户说了不要，仍在挽留',
     '对话中的拒绝表达没有被当作最高优先级',
     {'规则': '拒绝类表达（不需要、别推、不用了）命中即停止，进入冷却期',
      '训练数据': '增补拒绝样本和"拒绝 / 再想想"最小对比对'}, ['chat_optout', 'pair_optout']),
    ('场景:chat_nonshop', '出手判断', '把售后咨询当成购买犹豫',
     '用户在问已有订单、发票、退款，模型却给出挽留',
     {'规则': '对话先做意图分类，非购物意图不进入挽留流程', '训练数据': '增补非购物意图样本'}, ['chat_nonshop']),
    ('场景:no_mustbuy', '出手判断', '对必买用户出手',
     '复购周期内的老客短暂退出也被挽留，券会白发',
     {'特征': '打分模型增加复购次数、复购间隔、距上次购买天数', '训练数据': '增补必买样本'}, ['no_mustbuy']),
    ('场景:no_weak', '出手判断', '对弱需求用户出手',
     '停留几秒就离开的会话也被挽留，属于打扰',
     {'特征': '停留时长和深度行为数量作为出手门槛', '训练数据': '增补弱需求样本'}, ['no_weak']),
    ('场景:no_hoarder', '出手判断', '对囤货型用户出手',
     '购物车当收藏夹的用户被频繁挽留',
     {'特征': '增加"购物车加购 / 结算比"特征', '训练数据': '增补囤货型样本'}, ['no_hoarder']),
    ('场景:trap_lowest_price', '动作正确', '已是低价还在发券',
     '当前价已经是 30 天最低，模型没用价格事实，而是继续发券',
     {'Prompt': '明确"当前价 ≤ 近 30 天最低时只给价格事实"',
      '数据': '确保商品上下文里有 30 天均价和最低价',
      '训练数据': '增补"低价 / 非低价"最小对比对'}, ['trap_lowest_price', 'pair_lowest']),
    ('场景:trap_coupon_waiter', '误发券', '给"加购等券"用户发券',
     '画像显示用户习惯等挽留券才下单，模型仍然发券，会强化等券习惯',
     {'特征': '从券核销历史计算"加购后等券"模式', '训练数据': '增补等券型样本'}, ['trap_coupon_waiter']),
    ('场景:trap_live_expiring', '动作正确', '直播限时价没用上',
     '直播专享价即将失效是最好的事实，模型却发券或给了别的动作',
     {'Prompt': '商品上下文出现限时价时优先用价格事实，触点用 push', '训练数据': '增补直播限时样本'},
     ['trap_live_expiring']),
    ('场景:trap_installment', '动作正确', '高客单分期顾虑没识别',
     '用户在看分期方案，模型给了普通券或别的动作',
     {'Prompt': '高客单价 + 查看分期 → 分期信息', '训练数据': '增补分期样本'}, ['trap_installment']),
    ('场景:trap_profile_conflict', '主顾虑 top1', '画像压过了行为',
     '画像写着价格敏感，但这次会话全是尺码行为，模型仍判成价格',
     {'Prompt': '强调本次行为优先，画像只作先验', '训练数据': '增补画像与行为冲突样本'}, ['trap_profile_conflict']),
    ('场景:trap_focus_item', '主顾虑 top1', '把其他商品的行为算到焦点商品上',
     '会话里看了多个商品，模型把别的商品上的价格行为当成焦点商品的顾虑',
     {'数据': '事件级别标注所属商品', 'Prompt': '只看焦点商品的行为', '训练数据': '增补多商品会话样本'},
     ['trap_focus_item']),
    ('场景:chat_conflict', '主顾虑 top1', '没把用户亲口说的顾虑放在第一位',
     '用户在对话里明说了顾虑，模型仍按行为推断',
     {'Prompt': '对话中的显式表达优先于行为推断', '训练数据': '增补对话与行为冲突样本'}, ['chat_conflict']),
    ('场景:chat_implicit', '主顾虑 top1', '没听懂隐含表达',
     '"学生党预算有限""之前退过一次"这类话没有映射到对应顾虑',
     {'训练数据': '增补隐含表达样本，覆盖更多说法'}, [f'chat_implicit:{c}' for c in
                                         ['价格', '等大促', '凑单', '比价', '品质信任', '规格适配', '物流售后', '流程摩擦']]),
    ('场景:app_dual', '主顾虑 top1', '主次顾虑排反',
     '两类顾虑同时出现时，没有按证据强弱排序',
     {'Prompt': '按证据数量和强度排序主次', '训练数据': '增补双顾虑样本'}, ['app_dual']),
    ('层:L4_稀疏', '稀疏样本过度自信', '证据很少却很自信',
     '只有一条信号时仍给出高置信度，下游会按确定结论执行',
     {'Prompt': '证据少于 2 条时 uncertain=true、confidence ≤ 0.6', '训练数据': '增补稀疏样本'},
     ['app_sparse:价格', 'app_sparse:比价', 'app_sparse:规格适配', 'chat_think']),
    ('全部', '证据编号合法', '引用了不存在的事件编号',
     '证据编号超出范围，说明在编造证据',
     {'规则': '输出后处理：校验编号范围，不合法直接判无效并降级为规则动作'}, []),
    ('全部', '证据召回', '证据列得不全',
     '结论对了，但支撑它的事件只列了一部分，可解释性不够',
     {'Prompt': '要求列出所有支撑主顾虑的事件编号'}, []),
    ('全部', '输出无法解析', '输出格式不稳定',
     '部分输出不是合法 JSON',
     {'工程': '开启 JSON mode 或用 response_format 约束；失败时重试一次'}, []),
    ('全部', '回复数字可溯源', '回复里出现了输入中没有的数字',
     '对话回复编造了价格、天数等数字',
     {'工程': '回复改成"模板 + 事实槽位"，数字只能从商品上下文填入'}, []),
    ('全部', '回复暴露推断或营销腔', '回复暴露推断或用营销腔',
     '回复里出现"看您犹豫""亲""专属"等表述',
     {'工程': '回复过一层禁用词过滤', 'Prompt': '列出禁用表述'}, []),
    ('全部', '触点正确', '触点选择不对',
     '对话场景没用对话回复，或出手时触点与顾虑不匹配',
     {'Prompt': '写清楚各顾虑对应的触点'}, []),
]


def is_bad(metric, value, thresholds):
    t = thresholds.get(metric)
    if t is None or value is None:
        return False
    return value > t if metric in LOWER_IS_BETTER else value < t


def gap(metric, value, t):
    return (value - t) if metric in LOWER_IS_BETTER else (t - value)


def diagnose(summary, conf, thresholds):
    out = []
    for key, metric, title, cause, actions, augment in RULES:
        s = summary.get(key)
        if not s or metric not in s:
            continue
        v, t = s[metric], thresholds.get(metric)
        if not is_bad(metric, v, thresholds):
            continue
        n = int(s['_n'])
        sev = gap(metric, v, t) * math.log(n + 2)
        if metric in ('误发券', '出手判断') and key.startswith('场景:no_') or key == '场景:chat_optout':
            sev += 0.5
        out.append(dict(title=title, group=key, metric=metric, value=v, threshold=t, n=n, cause=cause,
                        actions=actions, augment=augment, severity=round(sev, 3)))
    # 按顾虑的混淆给出区分建议
    for g, row in conf.items():
        total = sum(row.values())
        wrong = {p: k for p, k in row.items() if p != g}
        if total < 3 or not wrong:
            continue
        acc = row.get(g, 0) / total
        if acc >= thresholds.get('主顾虑 top1', 0.85):
            continue
        p, k = max(wrong.items(), key=lambda x: x[1])
        aug = [f'app_clear:{g}', f'app_noisy:{g}', f'chat_explicit:{g}'] if g in ACTIONABLE else []
        out.append(dict(title=f'「{g}」常被判成「{p}」', group=f'顾虑:{g}', metric='主顾虑 top1', value=acc,
                        threshold=thresholds.get('主顾虑 top1'), n=total,
                        cause=f'{total} 条「{g}」里有 {k} 条被判成「{p}」',
                        actions={'Prompt': f'写清「{g}」和「{p}」的区分信号', '训练数据': f'增补「{g}」样本，并加入与「{p}」的难区分样本'},
                        augment=aug, severity=round((1 - acc) * math.log(total + 2), 3)))
    out.sort(key=lambda x: -x['severity'])
    return out


ACTIONABLE = {'价格', '等大促', '凑单', '比价', '品质信任', '规格适配', '物流售后', '流程摩擦'}


def pct(v):
    return '-' if v is None else f'{v:.0%}'


def delta(cur, prev, metric):
    if cur is None or prev is None:
        return ''
    d = cur - prev
    if abs(d) < 0.005:
        return '持平'
    good = (d < 0) if metric in LOWER_IS_BETTER else (d > 0)
    return f'{"↑" if d > 0 else "↓"}{abs(d):.0%}{"（变好）" if good else "（变差）"}'


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def render(meta, summary, results, conf, act_bd, thresholds, suggestions, baseline=None, previous=None,
           ablation=None, train_info=None, cases_by_id=None):
    L = []
    al = summary['全部']
    core = [m for m in METRICS if m in al]
    passed = [m for m in core if m in thresholds and not is_bad(m, al[m], thresholds)]
    judged = [m for m in core if m in thresholds]
    L.append(f'# 挽单评测报告 · {meta["name"]}')
    L.append('')
    L.append(table(['项目', '值'], [
        ['时间', meta['time']], ['被测', meta['backend_desc']], ['Prompt', meta.get('prompt_id', '-')],
        ['测试集', f'{meta["testset"]}，共 {al["_n"]:.0f} 条'], ['耗时', meta.get('elapsed', '-')],
        ['平均延迟', meta.get('latency', '-')], ['解析失败', meta.get('parse_fail', 0)],
    ]))
    L += ['', '## 1. 结论', '']
    L.append(f'- 全部判对 **{al["_全对"]:.0%}**；核心指标达标 **{len(passed)}/{len(judged)}**。')
    if baseline:
        L.append(f'- 关键词基线全部判对 {baseline["全部"]["_全对"]:.0%}，本次{delta(al["_全对"], baseline["全部"]["_全对"], "全对")}。')
    if previous:
        L.append(f'- 相比上次（{meta.get("previous_name", "上次")}）：全部判对 {delta(al["_全对"], previous["全部"]["_全对"], "全对")}。')
    if suggestions:
        L.append('- 最需要先解决：' + '；'.join(f'{s["title"]}（{s["group"]} {s["metric"]} {pct(s["value"])}）'
                                         for s in suggestions[:3]) + '。')
    else:
        L.append('- 没有指标低于及格线。')
    L += ['', '## 2. 核心指标', '']
    rows = []
    for m in core:
        t = thresholds.get(m)
        status = '-' if t is None else ('✅' if not is_bad(m, al[m], thresholds) else '❌')
        rows.append([m, pct(al[m]), '-' if t is None else (('≤ ' if m in LOWER_IS_BETTER else '≥ ') + pct(t)), status,
                     pct(baseline['全部'].get(m)) if baseline else '-',
                     delta(al[m], previous['全部'].get(m), m) if previous else '-'])
    L.append(table(['指标', '本次', '及格线', '状态', '关键词基线', '对比上次'], rows))
    L.append('')
    L.append('> 及格线为团队提议值，在 `eval/config.json` 修改。')

    def group_table(prefix, sort_by_acc=False):
        keys = [k for k in summary if k.startswith(prefix)]
        keys.sort(key=(lambda k: summary[k]['_全对']) if sort_by_acc else None)
        cols = ['出手判断', '主顾虑 top1', '动作正确', '证据召回', '误发券']
        rows = [[k.split(':', 1)[1], int(summary[k]['_n']), pct(summary[k]['_全对'])] +
                [pct(summary[k].get(c)) for c in cols] +
                ([delta(summary[k]['_全对'], previous[k]['_全对'], '全对') if previous and k in previous else '-'])
                for k in keys]
        return table(['分组', '样本', '全对'] + cols + ['全对 vs 上次'], rows)

    L += ['', '## 3. 分层表现', '', group_table('层:')]
    L += ['', '## 4. 场景表现（按全对率从低到高）', '', group_table('场景:', True)]
    L += ['', '## 5. 渠道与品类', '', group_table('渠道:'), '', group_table('品类:')]

    L += ['', '## 6. 顾虑混淆矩阵', '', '行是标准答案，列是模型判断。', '']
    labels = sorted({g for g in conf} | {p for r in conf.values() for p in r})
    L.append(table(['答案 \\ 判断'] + labels, [[g] + [conf[g].get(p, '') for p in labels] for g in sorted(conf)]))

    L += ['', '## 7. 出手判断分解', '']
    L.append(table(['原因（标准答案）', '样本', '出手判断正确'],
                   [[k, v[0], pct(v[1] / v[0])] for k, v in sorted(act_bd.items(), key=lambda x: x[1][1] / x[1][0])]))

    if ablation:
        L += ['', '## 8. 数据消融', '', '去掉一类输入后重跑同一模型。下降越多，说明这类数据越必要。', '']
        cols = ['_全对', '出手判断', '主顾虑 top1', '动作正确', '误发券']
        rows = [['完整输入'] + [pct(al.get(c)) for c in cols]]
        for name, s in ablation.items():
            rows.append([name] + [f'{pct(s["全部"].get(c))}（{delta(s["全部"].get(c), al.get(c), c).split("（")[0]}）'
                                  for c in cols])
        L.append(table(['版本', '全对', '出手判断', '主顾虑 top1', '动作正确', '误发券'], rows))

    L += ['', '## 9. 改进建议（按优先级）', '']
    if not suggestions:
        L.append('暂无。')
    for i, s in enumerate(suggestions, 1):
        L.append(f'### 9.{i} {s["title"]}')
        L.append('')
        L.append(f'- **现象**：{s["group"]} 的「{s["metric"]}」为 {pct(s["value"])}（及格线 {pct(s["threshold"])}，{s["n"]} 条）')
        L.append(f'- **原因判断**：{s["cause"]}')
        for k, v in s['actions'].items():
            L.append(f'- **{k}**：{v}')
        L.append('')

    L += ['## 10. 典型错误样例', '']
    by_scen = defaultdict(list)
    for r in results:
        if r['fails']:
            by_scen[r['scenario'].split(':')[0]].append(r)
    for scen, rs in sorted(by_scen.items(), key=lambda x: -len(x[1]))[:10]:
        r = rs[0]
        g, p = r['gold'], r['pred'] or {}
        L.append(f'**{r["id"]} · {scen}**（该场景错 {len(rs)} 条）未通过：{"、".join(r["fails"])}')
        L.append('')
        L.append(table(['', '出手', '主顾虑', '动作', '证据'], [
            ['答案', g['act'], g['primary'], '/'.join(g['action_ok']), g['evidence']],
            ['模型', p.get('act'), p.get('primary'), p.get('action'), p.get('evidence')]]))
        if cases_by_id:
            L += ['', '<details><summary>输入</summary>', '', '```', cases_by_id[r['id']]['query'], '```', '', '</details>']
        L.append('')

    if train_info:
        L += ['## 11. 已生成的训练数据', '']
        L.append(f'- 位置：`{train_info["dir"]}`（`sft.jsonl` 用于微调，`cases.jsonl` 带结构化字段与答案，可做打分模型特征）')
        L.append(f'- 共 {train_info["total"]} 条，其中最小对比对 {train_info["pairs"]} 条；与测试集商品和表述不重叠（泄漏检查：{train_info["leak_check"]}）')
        L.append('- 加量最多的场景：' + '、'.join(f'{k} {v} 条' for k, v in train_info['top_alloc']))
        L.append('')

    L += ['## 12. 说明与局限', '',
          '- 测试集是合成数据，用来检查决策规则、输出格式和典型陷阱；真实准确率要用人工标注的真实会话金标集衡量。',
          '- 训练数据和测试集用不同的商品和表述生成，测试分数不会因为训练数据而虚高；但两者出自同一套场景设计，测试集上的提升不等于真实线上提升。',
          '- 回复类指标只检查数字溯源和禁用表述，不评价说服力。']
    return '\n'.join(L) + '\n'
