"""调用被测模型，产出预测。四种后端：

- openai：任何 OpenAI 兼容的 /chat/completions 接口（方舟、内部网关、vLLM 等）
- cmd：自己的程序，stdin 读用例 jsonl（不含答案），stdout 输出预测 jsonl；适合打分模型或本地推理脚本
- preds：已经跑好的预测文件
- baseline：关键词规则基线
"""
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import baseline_keywords  # noqa: E402

PUBLIC_FIELDS = ('id', 'channel', 'multi_item', 'profile', 'item', 'events', 'decision_point', 'query')


def public(case):
    """去掉标准答案和场景名，防止泄漏给被测程序。"""
    return {k: case[k] for k in PUBLIC_FIELDS if k in case}


def extract_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text[text.find('\n') + 1:] if '\n' in text else text
    i = text.find('{')
    if i < 0:
        raise ValueError('输出里没有 JSON')
    obj, _ = json.JSONDecoder().raw_decode(text[i:])
    return obj


def _chat(base_url, model, api_key, system, user, temperature, max_tokens, timeout):
    body = json.dumps(dict(model=model, temperature=temperature, max_tokens=max_tokens,
                           messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]))
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions', data=body.encode('utf-8'),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data['choices'][0]['message']['content'], data.get('usage', {})


def run_openai(cases, *, base_url, model, api_key_env='LLM_API_KEY', system_prompt, temperature=0,
               max_tokens=600, timeout=60, workers=8, retries=2, log=print):
    api_key = os.environ.get(api_key_env, '')
    if not base_url or not model:
        raise SystemExit('openai 后端需要 --base-url 和 --model（或在 eval/config.json 里配置）')
    if not api_key:
        log(f'⚠ 环境变量 {api_key_env} 为空，按无鉴权请求')

    def one(c):
        t0, err, raw, usage = time.time(), None, '', {}
        for attempt in range(retries + 1):
            try:
                raw, usage = _chat(base_url, model, api_key, system_prompt, c['query'], temperature, max_tokens, timeout)
                pred = extract_json(raw)
                pred['id'] = c['id']
                pred['_latency'] = round(time.time() - t0, 2)
                pred['_usage'] = usage
                return pred, raw
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as e:
                err = f'{type(e).__name__}: {e}'
                time.sleep(1.5 * (attempt + 1))
        return dict(id=c['id'], _error=err, _latency=round(time.time() - t0, 2)), raw

    out, raws, done = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for pred, raw in ex.map(one, cases):
            out.append(pred)
            raws.append(dict(id=pred['id'], raw=raw))
            done += 1
            if done % 20 == 0 or done == len(cases):
                log(f'  已完成 {done}/{len(cases)}')
    return out, raws


def run_cmd(cases, cmd, timeout=3600):
    payload = ''.join(json.dumps(public(c), ensure_ascii=False) + '\n' for c in cases)
    proc = subprocess.run(shlex.split(cmd), input=payload, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise SystemExit(f'命令执行失败（退出码 {proc.returncode}）：\n{proc.stderr[-2000:]}')
    preds = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return preds, [dict(stderr=proc.stderr[-5000:])]


def run_preds(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()], []


def run_baseline(cases):
    return [baseline_keywords.predict(public(c)) for c in cases], []
