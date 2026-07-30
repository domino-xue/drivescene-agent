from dataclasses import dataclass

import pytest

from drivescene.agent.plan_execute import LLMJsonPlanner
from drivescene.agent.tool_registry import ToolSpec


@dataclass
class Response:
    content: object


class SequenceModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Response:
        self.prompts.append(prompt)
        return Response(self.responses.pop(0))


def _registry():
    def query_index(**kwargs):
        return kwargs

    spec = ToolSpec(
        name="EvidenceStore.query_index",
        category="retrieval",
        func=query_index,
        args_schema={"target": "str", "operation": "str"},
        purpose="query",
        input_contract="input",
        output_contract="output",
        when_to_use="query",
        do_not_use_when="analysis",
        zh_note="查询",
        examples=[],
    )
    return {spec.name: spec}


def _valid_payload() -> str:
    return """
    {
      "steps": [
        {
          "step_id": "step_1",
          "goal": "summarize events",
          "tool_name": "EvidenceStore.query_index",
          "args": {"target": "events", "operation": "summary"},
          "depends_on": [],
          "status": "pending"
        }
      ]
    }
    """


def test_planner_accepts_json_fence_and_validates_plan() -> None:
    model = SequenceModel([f"```json\n{_valid_payload()}\n```"])
    planner = LLMJsonPlanner(model, _registry())

    plan = planner.create_plan("汇总事件")

    assert [step.step_id for step in plan] == ["step_1"]
    assert plan[0].tool_name == "EvidenceStore.query_index"


def test_planner_retries_after_schema_validation_error() -> None:
    invalid = _valid_payload().replace("EvidenceStore.query_index", "Unknown.tool")
    model = SequenceModel([invalid, _valid_payload()])
    planner = LLMJsonPlanner(model, _registry(), max_attempts=2)

    plan = planner.create_plan("汇总事件")

    assert len(model.prompts) == 2
    assert "Unknown tool" in model.prompts[1]
    assert plan[0].step_id == "step_1"


def test_planner_rejects_forward_dependency() -> None:
    payload = """
    {"steps":[
      {"step_id":"step_1","goal":"first","tool_name":null,"args":{},
       "depends_on":["step_2"],"status":"pending"},
      {"step_id":"step_2","goal":"second","tool_name":null,"args":{},
       "depends_on":[],"status":"pending"}
    ]}
    """
    planner = LLMJsonPlanner(SequenceModel([payload]), _registry(), max_attempts=1)

    with pytest.raises(ValueError, match="must appear earlier"):
        planner.create_plan("test")


def test_replan_may_depend_on_completed_external_step() -> None:
    payload = """
    {"steps":[
      {"step_id":"step_2","goal":"follow up","tool_name":"EvidenceStore.query_index",
       "args":{"target":"events","operation":"summary"},
       "depends_on":["step_1"],"status":"pending"}
    ]}
    """
    request = (
        "The previous plan did not fully satisfy the task.\n"
        '{"existing_steps":[{"step_id":"step_1","status":"completed"}]}'
    )
    planner = LLMJsonPlanner(SequenceModel([payload]), _registry(), max_attempts=1)

    plan = planner.create_plan(request)

    assert plan[0].depends_on == ["step_1"]
