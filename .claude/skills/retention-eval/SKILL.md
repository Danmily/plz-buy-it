---
name: retention-eval
description: 智能挽单（plz-buy-it）模型的自动化评测、分析报告、改进建议和训练数据生成。当用户说「跑一下挽单评测」「评测一下模型 / 新 prompt」「测一下这个接入点」「看看模型哪里错了」「出评测报告」「生成下一轮训练数据」「做数据消融」时使用。
---

# 挽单评测

一条命令完成：调用被测模型 → 逐条打分 → 诊断 → Markdown 报告 → 按错误分布生成下一轮训练数据。

## 定位仓库

仓库根目录 = 本文件真实路径往上三级：

```bash
REPO="$(cd "$(dirname "$(readlink -f "<本 skill 的 Base directory>/SKILL.md")")/../../.." && pwd)"
```

找不到时，在用户工作目录下查找 `plz-buy-it/eval/pipeline.py`。后续命令都在 `$REPO` 下执行。

## 第一步：确定被测对象

按用户的话选后端，缺什么只问一次：

| 用户说 | 后端 | 需要 |
|---|---|---|
| 某个模型、接入点、endpoint、新 prompt | `openai` | `--base-url`、`--model`；先看 `eval/config.json` 的 `backend.openai` 有没有存 |
| 自己的脚本、打分模型、本地推理 | `cmd` | 一条命令：stdin 读用例 jsonl，stdout 写预测 jsonl |
| 已经有预测文件 | `preds` | 文件路径 |
| 只想看流程 / 没给模型 | `baseline` | 无 |

- API key 只走环境变量（默认 `LLM_API_KEY`，可用 `--api-key-env` 改名）。变量不存在时，请用户在自己的 shell 配置里 export 后重开会话；**不要让用户把 key 贴进对话，也不要打印它**。
- 用户说「换个 prompt 测」时，把新 prompt 写成文件，用 `--prompt <文件>` 传入，不要直接改 `eval/prompt.md`，除非用户要求。

## 第二步：运行

```bash
python3 eval/pipeline.py --backend <后端> [参数] --name <简短名称>
```

- 先用 `--limit 20` 冒烟，确认能解析输出，再跑全量。
- openai 后端跑全量（320 条）可能超过 2 分钟，放到后台运行。
- 用户要看"哪些数据有用"时加 `--ablate`（会额外调用两轮模型）。
- 不需要训练数据时加 `--train-n 0`；要更多就调大 `--train-n`。
- 指定对比对象用 `--compare eval/runs/<目录>`；默认自动对比同一测试集的上一次评测。

## 第三步：汇报

读 `eval/runs/<本次>/report.md`，用几句话告诉用户：

1. 全部判对率、核心指标达标几项、相比上次和关键词基线的变化
2. 前 3 条改进建议（报告第 9 节），每条说清现象和建议动作
3. 训练数据的位置、条数、泄漏检查结果
4. 报告文件路径

不要把整份报告贴进对话。用户想分享时，可以用 lark-doc 把报告同步成飞书文档。

## 其他请求

| 用户说 | 做法 |
|---|---|
| 只生成训练数据 | `python3 eval/gen_train.py --from-run eval/runs/<最近一次> --n <条数>` |
| 不看评测，均匀生成训练数据 | `python3 eval/gen_train.py --n <条数>` |
| 改及格线 | 改 `eval/config.json` 的 `thresholds` |
| 加新场景 | 在 `eval/scenarios.py` 写 builder、注册到 `BUILDERS` 和 `DEFAULT_TEST_PLAN`，并把 `gen_testset.py` 的 `VERSION` 加一 |

## 规则

- 测试集只用来评测。训练数据一律用 `gen_train.py` 生成（train split，商品和表述与测试集不重叠），不要把测试用例或 `results.jsonl` 里的样本拿去训练。
- 改动测试集内容必须升级 `VERSION`，否则新旧分数不可比。
- 合成测试集检查的是决策规则、格式和典型陷阱；汇报时提醒用户，真实准确率要用真实会话的人工金标集衡量。
