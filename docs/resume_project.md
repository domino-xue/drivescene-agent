# DriveScene Agent 简历项目稿

面向岗位：AI Agent 工程、AI 应用开发、LLM 应用工程。

## 推荐项目名称

**DriveScene Agent｜自动驾驶轨迹分析智能体**

技术栈：Python、LangGraph、LangChain、DeepSeek、Pandas、PyArrow、Streamlit、Pytest

## 一行版

基于 LangGraph 构建自动驾驶轨迹分析 Plan-and-Execute Agent，以富工具契约、Typed Blackboard、完成度评估和高风险确认实现可审计的中文多工具任务执行。

## 推荐三点版

- 设计并实现面向 Argoverse 2 轨迹数据的 Plan-and-Execute Agent，将 LLM 规划与确定性事件检测、索引检索、证据生成、表格转换和文件操作解耦，通过统一 Tool Registry 约束可执行边界。
- 构建 Typed Blackboard 参数绑定、Execution Digest、任务完成度评估与有界重规划机制，解决多步骤结果引用脆弱、工具成功但任务未完成等问题；对删除、覆盖和原地修改增加人工确认 Risk Gate。
- 建立 200 条中文分层 Planner 评测集和自动化消融框架；在 120 条 heldout/adversarial/safety 主评测上取得 **86.7%（104/120）严格 whole-case 通过率，Wilson 95% CI 79.4%–91.6%**，富契约提示包较 schema-only 提升 **65.0pp**。

## 更短的两点版

- 构建自动驾驶轨迹分析 Plan-and-Execute Agent，以 Tool Registry、Typed Blackboard、Completion Evaluator 和 Risk Gate 支撑可审计的中文多工具执行。
- 设计 200 条分层评测与消融实验，在 120 条独立主评测上达到 86.7% 严格 whole-case 通过率；富契约提示包较 schema-only 提升 65.0pp。

## 项目经历长版

DriveScene Agent 面向自动驾驶场景挖掘工作流。用户可以用中文查询急刹、近距离跟车、切入、换道等事件，要求重新计算运动学指标、获取动画和轨迹证据，或者导出表格与文件。

系统没有让 LLM 直接读取原始轨迹并猜测结论，而是将职责分为三层：LLM Planner 负责意图理解和工具编排，Plan-and-Execute Runtime 负责计划校验、状态绑定、完成度判断与风险控制，确定性 Python 工具负责轨迹计算和副作用操作。跨步骤输出被归一化为 review IDs、evidence paths、export directory 等 Typed Blackboard 槽位，降低了模型直接拼接 JSON 路径带来的失败。

为验证 Planner，项目构建了 200 条带 development、regression、heldout、adversarial、safety 分层的中文任务，并对工具名、schema、富契约和格式纠错机制进行配对消融。Full 配置在 120 条主评测上通过 104 条；schema-only 通过 26 条。项目同时记录数据集 hash、模型版本、代码 revision、逐例输出、Wilson 区间和失败 case，便于复现与审计。

## 简历指标应该怎样解释

### 86.7%是什么

它是 120 条 heldout/adversarial/safety 有限契约任务上的 **严格 whole-case 通过率**。一个 case 的步数、工具选择、依赖、参数和确认行为必须全部正确，才算通过。

### 65.0pp是什么

它是 Full 富契约提示包与 schema-only 配置在同模型、同任务上的绝对通过率差。Full 同时包含输入输出契约、使用边界、示例、领域规则和状态引用规则，因此不能表述为“某一个 Tool Contract 字段单独提升 65pp”。

### 为什么不主推启发式 +79.2pp

当前启发式只覆盖少量关键词意图，9 条通过均来自 retrieval 和 evidence，在另外四类任务中为 0。它适合作为离线调试下界，不是与 LLM Planner 等能力的竞争基线。

### 16条失败怎样回答

事后审计显示，4 条较明确属于模型错误：3 条有效性过滤字段映射错误，1 条过度规划。其余 12 条涉及题面未提供标准答案要求的源路径/列名，或省略与运行时默认值等价的参数。

正式结果仍保持 104/120；不能把争议项直接改判后宣传 96.x%。正确做法是修订 benchmark、冻结新 hash 并重跑全部配置。

## 面试安全表述

推荐说：

> 我评测的是 Agent 规划链路，而不是基础模型通用能力。120 条任务来自未参与提示词开发的 heldout、对抗和安全分层；每条任务只有在工具、参数、依赖和确认行为全部正确时才通过。

> Full 和 schema-only 使用同一个模型、相同样本和指标，所以 65pp 比 LLM 与启发式规则的对比更有解释力；但它仍是多个契约组件的组合消融，下一步还需要继续拆分。

> 我保留了失败 case。复盘后发现部分 table/safety 题面缺少标准答案所需字段，因此没有事后修改数字，而是把它记录为下一版 benchmark 的数据质量问题。

避免说：

- “模型准确率达到 86.7%。”
- “Tool Contract 单项让模型提升 65%。”
- “安全模块达到 100%，所以系统绝对安全。”
- “性能比规则系统提升十倍。”
- “已经在真实自动驾驶生产环境验证。”

## 面试官追问时的证据入口

| 追问 | 代码或文档 |
| --- | --- |
| 为什么不是套壳 | [架构职责划分](architecture.md) |
| Planner 如何防止幻觉工具 | [`tool_registry.py`](../src/drivescene/agent/tool_registry.py) |
| 跨步骤参数如何传递 | [`plan_execute.py`](../src/drivescene/agent/plan_execute.py) |
| 如何判断任务真正完成 | [`evaluator.py`](../src/drivescene/agent/evaluator.py) |
| 删除和覆盖如何保护 | [`risk.py`](../src/drivescene/ops/risk.py) |
| 86.7%如何复现 | [评测说明](agent_evaluation.md) |
| 如何现场演示 | [面试讲解](interview_guide.md) |

## 可继续增强的方向

- 修订题面缺字段和默认值等价匹配问题后重跑主评测。
- 增加同模型原生 tool calling / ReAct 竞争基线。
- 将富契约包拆成 contract、usage rule、example、domain rule、state rule 增量消融。
- 在真实用户问题上补充澄清率、执行成功率、安全误触发率和端到端延迟。

继续阅读：[项目首页](../README.md) · [系统架构](architecture.md) · [完整面试话术](interview_guide.md)
