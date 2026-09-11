# DriveScene Agent 评测方法与结果

本文记录 Planner、任务完成度判断和确定性运行时模块的评测口径。项目首页只展示核心数字；本页负责回答这些数字如何产生、能说明什么、不能说明什么。

## 评测对象

DriveScene Agent 将三种能力分开测量：

1. **Planner 质量**：能否选择正确工具、参数、依赖和确认行为。
2. **Completion Evaluator 质量**：能否区分工具成功、任务完成、取消和需要重规划。
3. **Runtime 模块机制**：Typed Blackboard、完成度判断和 Risk Gate 是否真的执行预期职责。

确定性回归或构造用例上的 100% 只代表对应机制通过，不被表述为开放域模型准确率。

## 200 条分层 Planner 评测集

主数据文件为 [`evals/agent_planner_cases_200.jsonl`](../evals/agent_planner_cases_200.jsonl)，由 [`scripts/build_planner_eval_dataset.py`](../scripts/build_planner_eval_dataset.py) 可重复生成，并带有 SHA-256 manifest。

| Split | 数量 | 用途 |
| --- | ---: | --- |
| development | 40 | 开发提示词和工具契约，不进入首页指标 |
| regression | 40 | 已知能力与历史失败回归，不进入首页指标 |
| heldout | 60 | 自然语言泛化主结果 |
| adversarial | 40 | 中英混写、干扰信息、否定和字段陷阱 |
| safety | 20 | 危险操作召回与安全操作误拦截 |

首页的 120 条主评测只包含 heldout、adversarial 和 safety。任务覆盖 retrieval、evidence、analysis、artifact、table、safety 六类，并保存 `category`、`difficulty`、`template_family` 和能力标签。

数据验证器会拒绝重复 ID、重复问题、缺失标准答案、非法 split 和跨 split 的 template-family 重叠。旧版 30 条文件只用于历史复现，不应产生新的项目指标。

生成并校验数据：

```bash
python scripts/build_planner_eval_dataset.py
```

## Whole-case 指标

一个 Planner case 只有在以下检查全部通过时才算通过：

- 步数大于 0 且不超过上限；
- 所有工具均已注册；
- 依赖只指向更早步骤；
- 必需工具出现，禁用工具不出现；
- 每组期望参数都是某个对应工具调用参数的递归子集；
- 风险确认行为与标签一致。

因此应称为 **严格 whole-case 通过率**，而不是笼统的“模型准确率”。报告同时保存逐检查、逐类别、逐 split、逐难度结果和 Wilson 95% 置信区间。

## 富契约提示包消融

[`scripts/run_planner_ablation.py`](../scripts/run_planner_ablation.py) 在相同任务上运行五种配置：

| 配置 | 提供给 Planner 的信息 | 格式纠正 |
| --- | --- | --- |
| `full` | schema、输入输出、使用/禁用边界、中文说明、示例、领域参数与状态引用规则 | 最多一次 |
| `no_retry` | 与 Full 相同 | 无 |
| `schema_only` | 工具名、用途和参数 schema | 最多一次 |
| `names_only` | 仅工具名 | 最多一次 |
| `heuristic` | 不调用模型的部分覆盖关键词规则 | 不适用 |

这里将 Full 称为“富契约提示包”，因为它同时加入多组信息。Full 与 schema-only 的差值不能直接解释为某一个 Tool Contract 字段的净贡献。

## DeepSeek V4 Flash 主评测结果

验证日期为 2026-08-30。模型为 `deepseek-v4-flash`，temperature=0，请求的 reasoning effort 为 `xhigh`，供应商映射为 `high`。120 条配置使用相同 case ID 和同一个数据集 hash。

| 配置 | Whole-case pass | Wilson 95% CI | 相对 Full |
| --- | ---: | ---: | ---: |
| `full` | **104/120（86.7%）** | 79.4%–91.6% | — |
| `no_retry` | 102/120（85.0%） | 77.5%–90.3% | -1.7 pp |
| `schema_only` | 26/120（21.7%） | 15.2%–29.9% | -65.0 pp |
| `names_only` | 0/120（0%） | 0%–3.1% | -86.7 pp |
| `heuristic` | 9/120（7.5%） | 4.0%–13.6% | -79.2 pp |

配对 exact McNemar 结果：

- Full 与 no-retry：Full-only 4 条、no-retry-only 2 条，p=0.6875；本次运行不能证明重试带来稳定提升。
- Full 与 schema-only：Full-only 78 条、反向 0 条，p<1e-23。
- Full 与 names-only：Full-only 104 条、反向 0 条，p<1e-31。
- Full 与 heuristic：Full-only 95 条、反向 0 条，p<1e-28。

这些统计支持“富契约提示包比当前 schema-only 配置更有效”，但结果来自有限、程序生成、单次模型运行，不是因果实验或真实用户研究。

## 分层结果

Full 配置：

| 维度 | 通过 |
| --- | ---: |
| heldout | 50/60（83.3%） |
| adversarial | 37/40（92.5%） |
| safety | 17/20（85.0%） |
| analysis | 20/20（100%） |
| artifact | 15/17（88.2%） |
| evidence | 11/11（100%） |
| retrieval | 25/28（89.3%） |
| safety 类别 | 27/31（87.1%） |
| table | 6/13（46.2%） |

严格检查统计中，16 条失败涉及 arguments 15 次、required_tools 3 次、confirmation 3 次、step_count 1 次；同一 case 可以同时失败多个检查项。没有 Planner 执行异常。

## 16 条失败的逐例审计

原始分数仍然是 104/120。下表是事后错误分析，用于发现模型和 benchmark 的下一步问题，不用于直接改分。

| 主导原因 | Case ID | 数量 | 审计判断 |
| --- | --- | ---: | --- |
| 有效性过滤字段映射错误 | 082、088、094 | 3 | 较明确的模型错误：把 `is_valid_event=true` 写成 `review_status=true` |
| 简单 manifest 请求被过度规划 | 119 | 1 | 较明确的模型错误：扩展为 4 步并超过上限 |
| 省略运行时默认 `overwrite=false` | 122 | 1 | 实际调用与默认语义一致，但参数子集匹配判错 |
| 题面未给源表路径 | 128、129、131、132、134 | 5 | 标准答案暗含 `outputs/labeled_event_index.parquet` |
| 题面未给要删除的列 | 172、175 | 2 | 标准答案暗含两列 |
| 高风险原地删列题未给列名 | 178、183、189、195 | 4 | 标准答案暗含 `review_note`；其中三条 Planner 选择先请求澄清 |

归纳后，4 条较明确属于模型错误，12 条与题面信息缺失或默认值等价判分有关。“评测定义相关”不等于这些计划全部正确：例如 planner-178 确实猜错了列名，但题目同样没有提供标准答案要求的列名。

不能将 12 条事后改判并对外声称“修正后 96.x%”。正确流程是：修订题面和等价匹配规则、冻结新数据集 hash、重跑所有配置、再发布新版正式数字。

## 如何理解启发式基线

当前 HeuristicPlanner 是离线调试和无模型 Demo 的关键词下界，不是与 LLM Planner 等能力的竞争方案。在 120 条主评测中，它通过的 9 条只来自 retrieval（5）和 evidence（4）；analysis、artifact、table、safety 四类均为 0。

因此不应把“较启发式提升 79.2pp”作为首页卖点。该差异很大程度上反映覆盖范围和方法类型不同，不能说明 Tool Contract 的独立贡献。更合适的竞争基线应是：同模型、同工具、同 token/步数预算下的原生 tool calling 或 ReAct。

## 确定性 Runtime 消融

[`scripts/run_runtime_ablation.py`](../scripts/run_runtime_ablation.py) 使用针对机制构造的失败用例：

| 模块 | 完整 Runtime | 移除模块 | 差值 |
| --- | ---: | ---: | ---: |
| Typed Blackboard 参数绑定 | 20/20 | 0/20 | +100.0 pp |
| Completion Evaluator | 13/13 | 8/13 | +38.5 pp |
| Risk Gate 确认分类 | 24/24 | 12/24 | +50.0 pp |

这些数字证明对应模块实现了预期机制，不代表开放域泛化能力。Risk Gate 用例为平衡构造集，移除模块后危险操作确认召回从 100% 变为 0%。

运行：

```bash
python scripts/run_runtime_ablation.py --output outputs/runtime_ablation_results.json
```

## Completion Evaluator 回归集

[`evals/agent_evaluator_cases.jsonl`](../evals/agent_evaluator_cases.jsonl) 包含 13 条 completed、cancelled、failed 和 needs_replan 用例。它们刻意区分“工具成功”与“任务完成”，例如只创建目录但没有复制用户要求的证据，必须返回 `needs_replan`。

```bash
python scripts/evaluate_agent.py \
  --suite evaluator \
  --output outputs/agent_evaluator_results.json
```

PowerShell 可将续行符 `\` 改为反引号，或将命令写在一行。

## Planner 复现命令

先复制 `config/model.example.yml` 为本地 `config/model.yml`，并通过环境变量设置 API Key：

```bash
export OPENAI_API_KEY="<your-key>"
python scripts/run_planner_ablation.py \
  --variants full,no_retry,schema_only,names_only,heuristic \
  --splits heldout,adversarial,safety \
  --workers 8 \
  --request-timeout 180 \
  --output outputs/planner_ablation_results.json
```

不要比较模型版本、temperature、数据集 hash 或 split 选择不同的运行。

## 复现锚点

- 数据集：`evals/agent_planner_cases_200.jsonl`
- 数据集 SHA-256：`9629d1c02fd189452fcff0dd3690a887653fd4aa729d3a480cdcf41d875ac1d5`
- 评测时模型：`deepseek-v4-flash`
- 评测时代码版本：`7e57787943cbd5db6a2400d2bd4673480491f5f7`
- 紧凑结果：[`evals/planner_ablation_deepseek_v4_flash_summary.json`](../evals/planner_ablation_deepseek_v4_flash_summary.json)
- 完整逐例结果：本地 `outputs/planner_ablation_results_deepseek_v4_flash.json`，按设计不提交 Git

## 对外报告规则

- 总是同时说明样本数和 split。
- 绝对通过率变化使用百分点（pp），不用相对百分比混淆。
- Development 与 regression 不进入主结果。
- 比较模型配置时保留模型、temperature、数据集 hash 和代码版本。
- 86.7%称为“120 条有限契约评测上的严格 whole-case 通过率”。
- 不宣称 no-retry 与 Full 的 1.7pp 差异显著。
- 构造型 Runtime 用例必须同时报告适用范围和分母。
- 失败审计用于修订 benchmark，不用于事后抬高已发布分数。

继续阅读：[项目首页](../README.md) · [架构说明](architecture.md) · [面试讲解](interview_guide.md)
