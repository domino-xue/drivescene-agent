from drivescene.agent.digest import ExecutionDigest
from drivescene.agent.reporting import LLMReporter


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return type("Response", (), {"content": self.content})()


class FailingModel:
    def invoke(self, prompt: str):
        raise RuntimeError("model unavailable")


def test_llm_reporter_uses_digest_prompt_and_returns_model_content() -> None:
    model = FakeModel("已找到 5 条急刹事件，并已复制到 outputs/exports/hard_braking。")
    digest = ExecutionDigest(
        task="查找急刹并复制证据",
        status="completed",
        events_found=5,
        event_types=["hard_braking"],
        files_copied=15,
        output_paths=["outputs/exports/hard_braking"],
        execution_summary=["检索急刹事件", "复制证据文件"],
    )

    answer = LLMReporter(model=model).generate(digest)

    assert answer == "已找到 5 条急刹事件，并已复制到 outputs/exports/hard_braking。"
    assert "ExecutionDigest" in model.prompts[0]
    assert "outputs/exports/hard_braking" in model.prompts[0]
    assert "不要编造" in model.prompts[0]


def test_llm_reporter_falls_back_to_deterministic_digest_answer() -> None:
    digest = ExecutionDigest(
        task="查找急刹并复制证据",
        status="completed",
        events_found=5,
        event_types=["hard_braking"],
        files_copied=15,
        output_paths=["outputs/exports/hard_braking"],
        execution_summary=["检索急刹事件", "复制证据文件"],
    )

    answer = LLMReporter(model=FailingModel()).generate(digest)

    assert "已完成：查找急刹并复制证据" in answer
    assert "共找到 5 条事件" in answer
    assert "已复制 15 个文件" in answer
    assert "outputs/exports/hard_braking" in answer


def test_llm_reporter_fallback_includes_event_ids_and_paths() -> None:
    digest = ExecutionDigest(
        task="给我一个急刹的事件",
        status="completed",
        events_found=1,
        event_types=["hard_braking"],
        output_paths=["outputs/review_assets/000123/animation.gif"],
        execution_summary=["查找急刹事件案例"],
    )
    digest.events = [
        {
            "review_id": "000123",
            "event_type": "hard_braking",
            "animation_path": "outputs/review_assets/000123/animation.gif",
            "min_velocity_acceleration_mps2": -7.5,
        }
    ]

    answer = LLMReporter().generate(digest)

    assert "000123" in answer
    assert "hard_braking" in answer
    assert "-7.5" in answer
    assert "刹停加速度最大的事件" in answer
    assert "outputs/review_assets/000123/animation.gif" in answer


def test_llm_reporter_prefers_export_paths_for_copy_tasks() -> None:
    digest = ExecutionDigest(
        task="复制急刹事件证据",
        status="completed",
        events_found=1,
        event_types=["hard_braking"],
        files_copied=3,
        output_paths=[
            "outputs\\review_assets\\000123\\animation.gif",
            "outputs\\exports\\hard_braking_top1_copy",
            "outputs\\exports\\hard_braking_top1_copy\\manifest.json",
        ],
    )

    answer = LLMReporter().generate(digest)

    assert "outputs\\exports\\hard_braking_top1_copy" in answer
    assert "outputs\\exports\\hard_braking_top1_copy\\manifest.json" in answer
    assert "outputs\\review_assets\\000123\\animation.gif" not in answer


def test_llm_reporter_fallback_includes_closest_following_metric() -> None:
    digest = ExecutionDigest(
        task="给我跟车距离最近的事件",
        status="completed",
        events_found=1,
        event_types=["close_following"],
        output_paths=["outputs/exports/close_following_top1"],
    )
    digest.events = [
        {
            "review_id": "000002",
            "event_type": "close_following",
            "scenario_id": "scene-1",
            "min_front_distance_m": 4.2,
            "min_ttc_s": 0.5,
            "min_velocity_acceleration_mps2": float("nan"),
        }
    ]

    answer = LLMReporter().generate(digest)

    assert "跟车距离最近的事件" in answer
    assert "刹停加速度最大的事件" not in answer
    assert "000002" in answer
    assert "min_front_distance_m=4.2" in answer
    assert "min_ttc_s=0.5" in answer
    assert "nan" not in answer.lower()
