# DriveScene Agent Demo

该 Demo 使用正式的 Plan-and-Execute Runtime 和三条内置事件索引，展示“中文请求 → 结构化计划 → 确定性工具 → 完成度判断 → 最终回答”的完整链路。推荐使用 API Key 调用真实模型；没有密钥时也可切换到确定性 Planner。

两种模式都不需要：

- 完整 Argoverse 2 数据；
- 已生成的人工复核资产。

在线模式需要模型 API Key 和网络访问；离线模式两者都不需要。

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

## 在线 CLI：真实模型规划

复制模型配置并通过环境变量注入密钥：

```powershell
Copy-Item config/model.example.yml config/model.yml
$env:OPENAI_API_KEY = "<your-api-key>"
python scripts/run_demo.py --online --question "找 2 个有效急刹案例并给出动画路径"
```

macOS / Linux：

```bash
cp config/model.example.yml config/model.yml
export OPENAI_API_KEY="<your-api-key>"
python scripts/run_demo.py --online --question "找 2 个有效急刹案例并给出动画路径"
```

在线模式中，Planner 与 Reporter 调用 `config/model.yml` 中的真实模型；事件查询、参数绑定、工具执行和完成度判断仍由确定性 Runtime 完成。默认示例配置使用 DeepSeek，也可以换成其他 OpenAI-compatible 服务。

## 离线 CLI

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

## Streamlit 界面

```bash
python -m streamlit run scripts/demo_app.py
```

页面展示聊天消息、计划表、事件指标和执行摘要。进程检测到 `OPENAI_API_KEY` 时默认启用在线模式，也可以在侧栏切换模式和模型配置。

![DriveScene Agent 实际运行录制](../docs/assets/drivescene-runtime-demo.gif)

## 实现说明

在线与离线模式共享 Tool Registry、Executor、状态传递、完成度判断和安全策略。两者仅在 Planner 与 Reporter 实现上不同，因此离线模式可以验证运行链路，但不能用于证明模型规划效果。

## Demo 的边界

- 内置数据只有三条，目的是验证运行链路，不代表完整数据规模。
- 离线 Planner 是确定性规则入口，不用于证明 LLM 规划能力；模型效果以在线模式和独立评测为准。
- 86.7% Planner 指标来自独立的 120 条模型主评测，不来自该三条 Demo 数据。
- 删除、覆盖和完整多步骤工作流应在正式 CLI/UI 中演示，不在最小 Demo 中执行真实文件副作用。

返回[项目首页](../README.md)，或查看[系统架构](../docs/architecture.md)与[评测报告](../docs/agent_evaluation.md)。
