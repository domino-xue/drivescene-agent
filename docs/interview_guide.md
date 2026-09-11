# DriveScene Agent 面试讲解指南

这份文档用于项目介绍、现场演示和技术追问。核心原则是先讲问题和设计选择，再讲框架名称；指标必须带样本和口径。

## 30 秒版本

> DriveScene Agent 是一个面向自动驾驶轨迹数据的 Plan-and-Execute Agent。它让 LLM 负责理解中文任务和规划工具，但轨迹计算、事件检测、文件导出都由确定性 Python 工具完成。为了让多步骤执行可靠，我实现了富工具契约、Typed Blackboard、任务完成度评估、有界重规划和高风险操作确认。项目还包含 200 条分层 Planner 评测，在 120 条主评测上取得 86.7% 的严格 whole-case 通过率。

## 2 分钟版本

> 这个项目解决的是自动驾驶数据分析中的多步骤任务编排。比如用户要求找 5 个有效急刹事件、获取每个事件的动画证据并导出。这里既有自然语言理解，也有检索、跨步骤传值、文件操作和安全控制。
>
> 我没有让 LLM 直接计算轨迹指标，而是把系统拆成三层：Planner 生成结构化步骤；Runtime 校验计划、绑定状态、检查风险和完成度；Python 工具负责真正计算和执行。所有工具通过 Tool Registry 注册，Planner 不能发明函数。工具结果进入 Typed Blackboard，后续步骤读取 review IDs、evidence paths 等语义槽位，不依赖模型拼接脆弱 JSON 路径。
>
> 工具执行成功后，我还会检查用户目标是否真的完成。例如只创建了目录但没复制证据，系统会有界重规划，而不是直接回答成功。删除、覆盖和原地修改由确定性 Risk Gate 拦截，必须人工确认。
>
> 评测方面，我构建了 200 条 development、regression、heldout、adversarial 和 safety 分层任务。Full 配置在 120 条主评测上严格通过 104 条，schema-only 只通过 26 条。这个 65pp 是整个富契约提示包的组合增益，不是基础模型通用准确率。

## 5 分钟架构讲解顺序

### 第一分钟：业务任务为什么不是单工具调用

用一个请求开场：

> “找 5 个有效急刹案例，返回动画证据并导出到指定目录。”

拆出隐含步骤：检索事件、取得 review ID、查询证据、创建目录、复制文件、检查数量和路径是否满足。强调这是一个状态化、多工具、有副作用的任务。

### 第二分钟：为什么选择 Plan-and-Execute

- ReAct 更灵活，但过程难以提前审计。
- 本项目先生成完整结构化计划，再校验工具、参数、依赖和步数。
- Executor 一步一步运行，失败信息和业务状态分离。
- 重规划被限制为一次，防止无限循环。

### 第三分钟：Typed Blackboard 如何解决跨步骤传值

- 工具输出统一成 state-transform envelope。
- Runtime 将输出归一化为 `review_ids`、`evidence_paths`、`export_dirs` 等槽位。
- Planner 使用 `{"$from_state":"evidence_paths"}` 这样的结构化引用。
- Executor 解析引用并在调用前重新验证参数。

### 第四分钟：怎样判断完成和控制风险

- Completion Evaluator 从 Execution Digest 检查交付物。
- 工具返回 `ok=true` 不代表用户目标已经满足。
- Risk Gate 独立于模型，按工具和参数确定风险。
- 删除、overwrite、input=output 的原地修改会暂停并请求确认。

### 第五分钟：如何证明设计有效

- 200 条分层 benchmark，主结果只用 120 条 heldout/adversarial/safety。
- Whole-case 要求工具、参数、依赖、步数和确认全部通过。
- Full 104/120，schema-only 26/120，绝对差 65pp。
- No-retry 102/120，与 Full 差异不显著，因此不夸大重试收益。
- 主动讲失败审计和下一步竞争基线，体现工程判断。

## 三分钟现场 Demo

### 准备

```bash
python -m pip install -e ".[dev]"
```

Demo 不需要 API Key 或完整 Argoverse 2 数据。

### 演示一：单步检索

```bash
python scripts/run_demo.py --question "找 2 个有效急刹案例"
```

讲解观察点：

1. Planner 生成 `EvidenceStore.query_index` 步骤。
2. 参数含 `event_type=hard_braking`、`is_valid_event=true` 和 `limit=2`。
3. 工具返回结构化事件数据。
4. Completion Evaluator 确认数量和类型满足。

### 演示二：界面展示

```bash
python -m streamlit run scripts/demo_app.py
```

重点展示聊天消息、计划表、工具结果、Execution Digest 和最终回答。不要把时间花在 UI 样式上，回到执行链路和可靠性设计。

### 如果面试官要求看完整 Agent

```bash
python -m streamlit run scripts/agent_app.py
```

完整 Agent 需要本地模型配置和 API Key。不要现场输入真实密钥到共享屏幕；提前用环境变量配置。

## 常见技术追问

### 1. 为什么不用纯 ReAct

回答重点：不是否定 ReAct，而是任务包含依赖、多文件副作用和确认边界。Plan-and-Execute 能在执行前验证整个计划，并把重规划限制在明确次数内。仓库保留一个最小 ReAct 学习示例，但正式入口使用 Plan-and-Execute。

### 2. 怎样防止 LLM 发明工具

Planner 只看到 Tool Registry 中的工具，计划解析后再次验证 `tool_name`。Executor 只从注册表取函数；未知名称在执行前失败，不会进入动态 import 或任意代码执行。

### 3. Schema 已经有参数，为什么还要 Rich Tool Contract

Schema 只能描述形状，不能充分说明什么时候用、什么时候不能用、输出如何被下一步消费，以及领域字段的精确语义。当前 schema-only 消融只有 21.7%，但 Full 一次加入多个组件，因此后续还要拆分每部分贡献。

### 4. Typed Blackboard 与普通 context dict 有什么区别

Blackboard 中的字段是由成功工具输出按语义归一化得到的稳定槽位，并且与 diagnostics 分开。它不是把所有历史消息和原始 JSON 无差别塞进上下文；后续工具可以按类型和语义绑定参数。

### 5. 如果工具返回成功，但只完成一半怎么办

Execution Digest 汇总实际交付物，Completion Evaluator 对照原始用户请求。如果缺少证据、导出文件或目标数量，返回 `needs_replan`。系统最多重规划一次，仍不满足时明确报告缺失项。

### 6. 风险判断为什么不用 LLM

安全边界不应依赖模型概率输出。Risk Gate 使用确定性规则读取工具名和参数，例如 delete、overwrite 或 input/output 同路径，结果可测试、可审计。

### 7. 86.7%是什么准确率

不是通用模型准确率。它是 120 条有限契约任务上的严格 whole-case 通过率；工具选择、参数、依赖、步数和确认行为全部正确才通过，并报告 Wilson 95% 区间。

### 8. 启发式基线是否太弱

是，所以不把 +79.2pp 作为主要结论。启发式是部分覆盖的离线调试下界，只有 retrieval/evidence 出现通过。更可信的当前证据是同模型 Full 与 schema-only 的配对消融；下一步会增加同模型原生 tool calling / ReAct 基线。

### 9. 为什么 table 类只有 46.2%

原始严格分数确实是 6/13。逐例审计发现 7 条失败的题面没有给出标准答案所要求的源路径或列名，因此既有模型行为问题，也有 benchmark 标注问题。项目没有事后调高分数，而是记录修订后重跑的计划。

### 10. 如果重新做一次，最先改什么

先修评测质量：显式补全题面字段、让默认参数等价匹配、给“请求澄清”增加安全可接受标签。然后增加同模型竞争基线，最后拆分富契约组件并重复多次模型运行。

## 可以主动讲的失败与反思

最能体现工程成熟度的不是“全都做对”，而是说明失败如何被发现和处理：

- 3 条模型把 `is_valid_event=true` 错写为 `review_status=true`，说明布尔有效性与复核状态需要更清晰的字段契约。
- 1 条简单 manifest 请求被过度规划，说明 Planner 还需要最小充分计划约束。
- 11 条题面缺少源表或列名，说明程序生成 benchmark 也需要语义完整性检查。
- 1 条省略默认 `overwrite=false` 被判错，说明评测器需要默认值归一化。

结论：保持正式分数不变，修订数据集后完整重跑，而不是通过事后口径抬高结果。

## 面试结束时的总结

> 这个项目真正想展示的不是我调用了某个 Agent 框架，而是我如何把不稳定的模型规划放进一个可验证、可审计、有状态、有安全边界的执行系统；并且用分层评测和失败分析验证这些设计。

继续阅读：[项目首页](../README.md) · [架构说明](architecture.md) · [评测说明](agent_evaluation.md) · [简历项目稿](resume_project.md)
