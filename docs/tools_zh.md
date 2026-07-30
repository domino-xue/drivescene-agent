# DriveScene Agent Tools

本文档说明当前暴露给 LLM/Agent 的工具接口。重构后，Agent 只看到 4 个通用参数化工具；急刹、近距离跟车、文件操作和表格操作都通过 `operation`、`event_type`、`filters` 等参数表达，避免为同类任务不断增加新工具名。

工具描述采用英文主说明加中文备注，结构包含 `Input`、`Output`、`When to use`、`Do not use when` 和 examples。中文备注只用于帮助中文请求映射参数，不作为唯一语义来源。

## Tool Registry

代码入口：

```python
from drivescene.agent.tool_registry import (
    build_tool_registry,
    execute_registered_tool,
    preflight_tool_call,
    tool_descriptions,
)
```

当前 agent-facing 工具白名单只有：

```text
EvidenceStore.query_index
AnalysisTools.run_analysis
ArtifactOps.manage_artifacts
TableOps.transform_table
```

执行工具前仍然先调用风险预检：

```python
preflight = preflight_tool_call(
    registry,
    {
        "tool": "ArtifactOps.manage_artifacts",
        "args": {"operation": "delete", "paths": ["outputs/old"]},
    },
)
```

如果 `preflight["risk"]["requires_confirmation"] is True`，Plan-and-Execute agent 会暂停并等待人工确认。

## 1. EvidenceStore.query_index

用途：查询已经构建好的事件索引或场景索引，不重新读取原始 parquet 做检测。

签名：

```python
store.query_index(
    target: str,
    operation: str,
    filters: dict | None = None,
    review_ids: list[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    limit: int | None = None,
)
```

参数：

- `target`: `"events"` 或 `"scenarios"`。
- `operation`: 对 `events` 支持 `"search"`、`"summary"`、`"detail"`、`"evidence"`；对 `scenarios` 支持 `"search"`。
- `filters`: 搜索条件，如 `{"event_type": "hard_braking", "is_valid_event": True}`。
- `review_ids`: 查询事件详情或证据路径时使用。
- `sort_by` / `ascending` / `limit`: 排序和截断。

示例：

```python
store.query_index(
    target="events",
    operation="search",
    filters={"event_type": "close_following", "is_valid_event": True},
    limit=5,
)
```

```python
store.query_index(target="events", operation="summary")
```

```python
store.query_index(
    target="events",
    operation="evidence",
    review_ids=["000094"],
)
```

## 2. AnalysisTools.run_analysis

用途：读取原始 scenario parquet，运行确定性检测器或计算运动学/相对运动指标。

签名：

```python
tools.run_analysis(
    operation: str,
    event_type: str | None = None,
    scenario_ids: list[str] | None = None,
    scenario_paths: list[str | Path] | None = None,
    scenario_glob: str | None = None,
    track_id: str | int | None = None,
    actor_id: str | int | None = None,
    window: dict[str, int] | None = None,
    parameters: dict | None = None,
)
```

输入源必须三选一：

- `scenario_ids`
- `scenario_paths`
- `scenario_glob`

支持的 `operation`：

- `"detect_events"`: 需要 `event_type`，支持 `"hard_braking"`、`"close_following"`、
  `"cut_in"`、`"stopped_vehicle_ahead"` 和 `"lane_change"`。
- `"compute_track_kinematics"`: 需要 `track_id`。
- `"compute_relative_motion"`: 可选 `actor_id`。
- `"compare_velocity_position_evidence"`: 需要 `track_id` 和 `window={"start_timestep": ..., "end_timestep": ...}`。

批量输出统一格式：

```python
{
    "ok": False,
    "operation": "detect_events",
    "summary": {"total": 2, "succeeded": 1, "failed": 1},
    "items": [
        {"input": "scene-1", "ok": True, "result": [...]},
        {
            "input": "missing-scene",
            "ok": False,
            "error_type": "FileNotFoundError",
            "error": "...",
        },
    ],
}
```

示例：

```python
tools.run_analysis(
    operation="detect_events",
    event_type="hard_braking",
    scenario_ids=["scene-1", "scene-2"],
)
```

```python
tools.run_analysis(
    operation="detect_events",
    event_type="close_following",
    scenario_glob="data/val/*/scenario_*.parquet",
)
```

## 3. ArtifactOps.manage_artifacts

用途：创建导出目录、复制证据文件、删除文件/文件夹、写 manifest。

签名：

```python
artifact_ops.manage_artifacts(
    operation: str,
    paths: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    manifest: dict | None = None,
    preserve_structure: bool = False,
    base_dir: str | Path | None = None,
    overwrite: bool = False,
    missing_ok: bool = True,
)
```

支持的 `operation`：

- `"create_folder"`
- `"copy"`
- `"delete"`
- `"write_manifest"`

风险规则：

- `delete` 是 high risk，需要人工确认。
- `copy` 且 `overwrite=True` 是 high risk。
- `create_folder` 和普通 `copy` 是 low risk。
- `write_manifest` 默认 low risk，覆盖写按风险策略处理。

示例：

```python
artifact_ops.manage_artifacts(
    operation="copy",
    paths=["outputs/review_assets/000094/animation.gif"],
    output_dir="outputs/exports/demo_cases",
)
```

## 4. TableOps.transform_table

用途：对 parquet/csv 表做派生转换，如过滤、删列、格式转换。

签名：

```python
table_ops.transform_table(
    operation: str,
    input_path: str | Path,
    output_path: str | Path,
    columns: list[str] | None = None,
    filters: dict | None = None,
    overwrite: bool = False,
)
```

支持的 `operation`：

- `"drop_columns"`
- `"filter_rows"`
- `"convert_format"`

风险规则：

- 输出到新文件是 medium risk。
- `overwrite=True` 或 `input_path == output_path` 是 high risk，需要人工确认。

示例：

```python
table_ops.transform_table(
    operation="filter_rows",
    input_path="outputs/labeled_event_index.parquet",
    output_path="outputs/exports/valid_hard_braking.parquet",
    filters={"event_type": "hard_braking", "is_valid_event": True},
)
```

## Planner Usage Pattern

用户问“找 5 个有效急刹案例”时，planner 应生成：

```json
{
  "tool_name": "EvidenceStore.query_index",
  "args": {
    "target": "events",
    "operation": "search",
    "filters": {
      "event_type": "hard_braking",
      "is_valid_event": true
    },
    "limit": 5
  }
}
```

用户问“批量重跑近距离跟车检测”时，planner 应生成：

```json
{
  "tool_name": "AnalysisTools.run_analysis",
  "args": {
    "operation": "detect_events",
    "event_type": "close_following",
    "scenario_glob": "data/val/*/scenario_*.parquet"
  }
}
```
