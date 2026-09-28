# Aurora-X · Agentic Observer 提交项目

GOSIM 2026 智能巡天黑客松 · 队伍 **Aurora-X**（GitHub：kkkkikun）的正式提交项目。
平台 `/compete` 页使用本仓库 URL 提交：平台会固定源码版本与启动配置、跑公开场景接口测试，由队伍确认版本后评测。

## 目录结构

```
observer.project.json   # 启动清单（observer-project-v1）
agent/
  minimal_agent.py      # 入口：JSONL 协议、异常兜底、跨 cwd 导入自解析
  my_strategy.py        # 策略主文件（v6.11 + 合约读取泛化）
  decision_graph.py     # 决策图 / 候选排序基础设施
  anomaly_detection.py  # 异常检测（分层归一化）
  scoring_preview.py    # 评分预览（REQUIRED 可完成性校验）
  protocol.py / state.py / model_factory.py / reference_strategy.py
  requirements.txt      # 仅可选 LLM 模式需要；确定性模式零依赖
  .env.example          # 环境变量模板（不含任何密钥）
```

## 运行方式

```json
{
  "schema_version": "observer-project-v1",
  "protocol": "jsonl-v2",
  "image": "python:3.12-slim",
  "working_directory": ".",
  "build": [],
  "run": ["python3", "-u", "agent/minimal_agent.py"]
}
```

常驻进程，stdin/stdout 每行一个 JSON 对象：`initialize`（不回复）→ 每个决策点一条 `decision_request` → 回一条 `decision_response`（`observe` 或 `wait`）。诊断日志走 stderr，每次响应立即 flush。

**确定性模式零依赖**：不需要任何模型 key、网络或额外 pip 包即可运行；`requirements.txt` 只为可选的 LLM 扩展准备。

## 策略概览

三层结构，按场景类型自动切换：

1. **未观测 REQUIRED 硬优先** —— 消除终局 `-1000` 漏拍风险，配合末班车游标预留处理短窗口天区。
2. **正式赛型（coverage_bonus_weight > 0）** —— 覆盖感知边际排序：估计收益 + 覆盖均匀度（Jain）的边际贡献，均匀分布的覆盖项在 demand > 100% 时约占总分五分之一。
3. **宽松场景（W = 0）** —— 质量择时（等到高度/时角进入高质量段再拍）+ 松弛门，日历压力大时自动退回默认排序。

**异常检测**（分层归一化）：读数比相对当夜全场中位数与分区滚动中位数判定仪器故障与天区标签（nova / reddening），支持确认调度与上报；整个检测器与决策函数均有兜底，任何异常只丢一次检测机会，绝不终止运行。

**任务卡泛化**：分区清单、项目加成表、异常因子全部从平台 `initialize` 合约读取（`tile_catalog.region_ids` / `score_config.program_bonus` / `score_config.anomaly_tags.*_factor`），不硬编码分区数与评分常数，可直接跑未知任务卡。

## 本地复现与验证

使用官方 starter kit 的 `local_runner.py`（平台同款 `JsonLineAgentProcess` 传输），从任意工作目录运行：

```bash
python3 <starter-kit>/local_runner.py \
  --scenario <starter-kit>/scenarios/demo-week \
  --agent <本仓库>/agent --out run_output
```

2026-09-28 四路冒烟（均 `survey_complete`，agent.log 零错误，`decisions.csv` md5 一致）：

| 跑法 | 场景 | 总分 |
|---|---|---|
| 项目目录直跑（跨 cwd） | demo-week | 5909.570277 |
| **ZIP 解包直跑（cwd=/tmp）** | demo-week | 5909.570277 |
| 策略原件直跑 | demo-week | 5909.570277 |
| 项目目录直跑 | comp-like（30 夜压力场） | 204616.091206 |
| 策略原件直跑 | comp-like | 204616.091206 |

## 安全

- 仓库与清单中**不含任何凭据**；`.env` 已被 `.gitignore` 排除，模型 key 只在评测页内存中填写，不入库、不进日志。
- 平台按已提交动作评分，全链路兜底，评测中不会因策略异常整场终止。
