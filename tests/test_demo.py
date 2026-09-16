from __future__ import annotations

import pytest

from scripts import run_demo


def test_offline_demo_remains_the_default() -> None:
    result = run_demo.run_demo("找 2 个有效急刹案例")

    assert result["runtime_mode"] == "offline · deterministic planner"
    assert result["evaluation"]["status"] == "completed"
    assert result["plan"]


def test_online_demo_requires_an_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        "model:\n  model: test-model\n  base_url: https://example.test/v1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_demo.build_demo_agent(online=True, model_config=config_path)
