# Agentic Observer — GOSIM 智能巡天黑客松

队伍 **Aurora-X**（GitHub 领队 KKKKIKUN）参赛工作目录。比赛：2026-10-05 00:00 – 10-07 23:59（北京时间），正式赛只收完整项目（`observer.project.json`）。

## 目录结构

| 路径 | 说明 | 入库 |
|---|---|---|
| `agent-observer-starter-kit/agent/` | **策略代码（核心资产）**——`my_strategy.py` 是主战场，v6.11 定版 | ✅ |
| `agentic-observer-project/` | 正式赛提交项目（manifest + agent） | ✅ |
| `agent-observer-project.zip` | 实际提交的 ZIP（由上一行构建） | ✅ |
| `agentic-observer-project/README.md` + `sync_submit_repo.sh` | 对外提交仓库的权威源与同步脚本 → `github.com/kkkkikun/aurora-x-obp`（public，练习赛/正式赛提交链接） | ✅ |
| `agent-observer-starter-kit.zip` | 官方入门包原件（策略之外的 kit 状态由此锁定） | ✅ |
| `比赛调研报告.md` / `下一步行动指南.md` / `正式赛备战计划.md` / `项目提交预演清单.md` | 调研与作战文档 | ✅ |
| `reference/` | 官网规则/文档快照（2026-09-25，比 GitHub 仓库新） | ✅ |
| `agent-observer-starter-kit.zip` 之外的平台/克隆资料（`platform/`、`observer-project-example/`、`platform_mirror/`） | 官方仓库与示例的本地副本，可重新获取 | ❌ |
| 基准输出（`*_output/`、`wf_out/`）与生成场景（`scenarios/comp-*`、`dev-fortnight`） | 可再生：输出按轮丢弃，场景按种子由 `make_scenario.py` 复现 | ❌ |

## 策略版本线（my_strategy.py）

| 版本 | 内容 | 三场景总分 |
|---|---|---|
| v6.4 | 覆盖感知边际排序 + REQUIRED 硬优先 + 松弛门质量择时 | 587,588.2 |
| v6.7 | + 故障报告等待期分区规避（工作流第 1 轮接受） | 587,859.9 |
| v6.8 | + REQUIRED 末班车游标预留（第 2 轮接受） | 588,704.3 |
| v6.11 | + 中天爬升加成（仅非 mechanics 场景，第 6 轮接受） | **588,798.9** |
| P0 泛化 | 硬编码 8 区 / 加成表 / 异常因子 → 改从 `initialize` 合约读取（9/28，应对未知任务卡 E/F/G/H） | 588,798.9（逐位不变） |

详细实验记录（含负结果与测量墓碑）：`下一步行动指南.md`。

## 提交纪律

每次重要修改（策略版本被接受、文档更新、包重建）单独 commit 并打 `strategy/*` 标签；提交包以 git 内的 zip 为准。
