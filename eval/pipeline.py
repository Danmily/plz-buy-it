#!/usr/bin/env python3
"""一条命令完成：调用模型 → 打分 → 诊断 → 报告 → 生成下一轮训练数据。

  # 关键词基线（不需要模型，用来确认流程能跑通）
  python3 eval/pipeline.py --backend baseline

  # OpenAI 兼容接口（方舟、内部网关、vLLM 等），API key 放环境变量
  LLM_API_KEY=xxx python3 eval/pipeline.py --backend openai --base-url https://.../v1 --model <模型或接入点>

  # 自己的程序：stdin 读用例 jsonl（不含答案），stdout 输出预测 jsonl
  python3 eval/pipeline.py --backend cmd --cmd "python3 my_model/predict.py"

  # 已有预测文件
  python3 eval/pipeline.py --backend preds --preds my_preds.jsonl

常用参数：--name 本次名称；--limit 20 冒烟测试；--ablate 加跑数据消融；--train-n 0 不生成训练数据；
--compare <run 目录> 指定对比对象（默认自动对比上一次）。
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import backends  # noqa: E402
import gen_testset  # noqa: E402
import report  # noqa: E402
from gen_train import make_training  # noqa: E402
from score import act_breakdown, confusion, score  # noqa: E402

RUNS = HERE / 'runs'


def load_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def ensure_testset(ablate=()):
    p = gen_testset.path_for(ablate)
    if not p.exists():
        gen_testset.write(gen_testset.build(ablate), p)
    return p


def predict(args, cases, cfg, system_prompt, log):
    if args.backend == 'baseline':
        return backends.run_baseline(cases)
    if args.backend == 'preds':
        return backends.run_preds(args.preds)
    if args.backend == 'cmd':
        return backends.run_cmd(cases, args.cmd)
    o = dict(cfg['backend']['openai'])
    for k in ('base_url', 'model', 'api_key_env', 'workers'):
        v = getattr(args, k)
        if v:
            o[k] = v
    return backends.run_openai(cases, base_url=o['base_url'], model=o['model'], api_key_env=o['api_key_env'],
                               system_prompt=system_prompt, temperature=o['temperature'],
                               max_tokens=o['max_tokens'], timeout=o['timeout'], workers=int(o['workers']), log=log)


def previous_run(exclude, testset_label):
    """同一测试集、同样条数的最近一次评测。"""
    if not RUNS.exists():
        return None
    for d in sorted(RUNS.iterdir(), reverse=True):
        f = d / 'metrics.json'
        if d != exclude and f.exists():
            if json.loads(f.read_text(encoding='utf-8'))['meta'].get('testset') == testset_label:
                return d
    return None


def backend_desc(args, cfg):
    if args.backend == 'openai':
        return f'openai · {args.model or cfg["backend"]["openai"]["model"]}'
    if args.backend == 'cmd':
        return f'cmd · {args.cmd}'
    if args.backend == 'preds':
        return f'preds · {args.preds}'
    return '关键词基线'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', choices=['openai', 'cmd', 'preds', 'baseline'], default='baseline')
    ap.add_argument('--base-url')
    ap.add_argument('--model')
    ap.add_argument('--api-key-env')
    ap.add_argument('--workers', type=int)
    ap.add_argument('--cmd')
    ap.add_argument('--preds')
    ap.add_argument('--prompt', default=str(HERE / 'prompt.md'))
    ap.add_argument('--testset', help='默认使用 eval/testset_v1.jsonl，不存在时自动生成')
    ap.add_argument('--name', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ablate', action='store_true', help='加跑去掉画像、去掉商品上下文两组消融')
    ap.add_argument('--compare', help='对比的 run 目录；默认上一次')
    ap.add_argument('--train-n', type=int, default=2000)
    a = ap.parse_args()

    cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    system_prompt = Path(a.prompt).read_text(encoding='utf-8')
    prompt_id = f'{Path(a.prompt).name}#{hashlib.md5(system_prompt.encode()).hexdigest()[:8]}'
    ts_path = Path(a.testset) if a.testset else ensure_testset()
    cases = load_jsonl(ts_path)
    if a.limit:
        cases = cases[:a.limit]
    name = a.name or a.backend
    run_dir = RUNS / f'{datetime.now():%Y%m%d-%H%M%S}_{name}'
    run_dir.mkdir(parents=True)
    log = print

    log(f'▶ 评测 {len(cases)} 条 · {backend_desc(a, cfg)}')
    t0 = time.time()
    preds, raws = predict(a, cases, cfg, system_prompt, log)
    elapsed = time.time() - t0
    write_jsonl(run_dir / 'preds.jsonl', preds)
    if raws:
        write_jsonl(run_dir / 'raw_outputs.jsonl', raws)

    results, summary = score(cases, preds)
    conf, act_bd = confusion(results), act_breakdown(results)
    _, base_summary = score(cases, backends.run_baseline(cases)[0])
    th = cfg['thresholds']
    suggestions = report.diagnose(summary, conf, th)

    ablation = None
    if a.ablate and a.backend in ('openai', 'cmd'):
        ablation = {}
        for ab, label in ((('profile',), '去掉画像'), (('item',), '去掉商品上下文')):
            ac = load_jsonl(ensure_testset(ab))
            if a.limit:
                ac = ac[:a.limit]
            log(f'▶ 消融：{label}')
            ap_, _ = predict(a, ac, cfg, system_prompt, log)
            ablation[label] = score(ac, ap_)[1]

    ts_label = f'{ts_path.name}' + (f'（前 {a.limit} 条）' if a.limit else '')
    prev_dir = Path(a.compare) if a.compare else previous_run(run_dir, ts_label)
    prev = json.loads((prev_dir / 'metrics.json').read_text(encoding='utf-8'))['summary'] if prev_dir else None

    train_info = None
    if a.train_n > 0:
        log(f'▶ 按错误分布生成 {a.train_n} 条训练数据')
        train_info = make_training(run_dir / 'train_data', a.train_n, summary=summary, suggestions=suggestions)

    lat = [p.get('_latency') for p in preds if p.get('_latency') is not None]
    meta = dict(name=name, time=f'{datetime.now():%Y-%m-%d %H:%M}', backend_desc=backend_desc(a, cfg),
                prompt_id=prompt_id if a.backend == 'openai' else '-',
                testset=ts_label,
                elapsed=f'{elapsed:.0f} 秒', latency=f'{sum(lat) / len(lat):.2f} 秒' if lat else '-',
                parse_fail=sum(1 for p in preds if p.get('_error')),
                previous_name=prev_dir.name if prev_dir else None)
    md = report.render(meta, summary, results, conf, act_bd, th, suggestions, baseline=base_summary,
                       previous=prev, ablation=ablation, train_info=train_info,
                       cases_by_id={c['id']: c for c in cases})
    (run_dir / 'report.md').write_text(md, encoding='utf-8')
    (run_dir / 'metrics.json').write_text(json.dumps(dict(meta=meta, summary=summary, suggestions=suggestions,
                                                          ablation=ablation), ensure_ascii=False, indent=1),
                                          encoding='utf-8')
    write_jsonl(run_dir / 'results.jsonl', results)

    al = summary['全部']
    log('')
    log(f'✔ 全部判对 {al["_全对"]:.0%} · 出手判断 {al["出手判断"]:.0%} · 主顾虑 top1 {al.get("主顾虑 top1", 0):.0%}'
        f' · 动作正确 {al.get("动作正确", 0):.0%} · 误发券 {al["误发券"]:.0%}')
    for s in suggestions[:3]:
        log(f'  · {s["title"]}（{s["group"]} {s["metric"]} {s["value"]:.0%}）')
    log(f'报告：{run_dir / "report.md"}')
    if train_info:
        log(f'训练数据：{train_info["dir"]}（{train_info["total"]} 条，泄漏检查 {train_info["leak_check"]}）')


if __name__ == '__main__':
    main()
