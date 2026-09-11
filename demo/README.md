# DriveScene Agent 最小离线 Demo

该 Demo 使用真实的 Plan-and-Execute 运行时和三条内置事件索引，展示“中文请求 → 结构化计划 → 确定性工具 → 完成度判断 → 最终回答”的完整链路。

不需要：

- API Key；
- 完整 Argoverse 2 数据；
- 已生成的人工复核资产；
- 外部网络服务。

## 安装

在仓库根目录创建 Python 3.11 虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## CLI 演示

使用默认问题：

```bash
python scripts/run_demo.py
```

传入自己的问题：

```bash
python scripts/run_demo.py --question "找 2 个有效急刹案例"
```

输出 JSON，便于 CI 或脚本检查：

```bash
python scripts/run_demo.py --json
```

CLI 输出包括：

1. Planner 生成的步骤；
2. `EvidenceStore.query_index` 的参数和确定性结果；
3. Execution Digest；
4. Completion Evaluator 状态；
5. 最终中文回答。

## Streamlit 演示

```bash
python -m streamlit run scripts/demo_app.py
```

页面展示聊天消息、计划表、事件指标和执行摘要，适合面试时快速说明 Agent 的主执行链路。

![最小离线 Demo](../docs/assets/drivescene-demo.png)

## 推荐面试演示话术

> 这个 Demo 没有调用模型，但走的是正式 Plan-and-Execute Runtime。这里用确定性 Planner 替代在线 LLM，是为了让任何人克隆仓库后都能复现执行、状态和完成度判断链路。正式 Agent 只替换 Planner，不会绕过 Tool Registry、Executor 或安全策略。

## Demo 的边界

- 内置数据只有三条，目的是验证运行链路，不代表完整数据规模。
- Demo Planner 是离线规则入口，不用于证明 LLM 规划能力。
- 86.7% Planner 指标来自独立的 120 条模型主评测，不来自该三条 Demo 数据。
- 删除、覆盖和完整多步骤工作流应在正式 CLI/UI 中演示，不在最小 Demo 中执行真实文件副作用。

返回 [项目首页](../README.md)，或查看 [完整面试讲解](../docs/interview_guide.md)。
