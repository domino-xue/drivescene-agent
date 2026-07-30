import subprocess
import sys
import textwrap


def test_agent_app_import_does_not_require_cli_build_reporter_symbol() -> None:
    code = textwrap.dedent(
        """
        import sys
        import types

        fake_cli = types.ModuleType("scripts.run_agent_cli")
        fake_cli.build_planner = lambda *args, **kwargs: None
        fake_cli.build_registry_from_paths = lambda *args, **kwargs: None
        sys.modules["scripts.run_agent_cli"] = fake_cli

        import scripts.agent_app as agent_app

        assert hasattr(agent_app, "_build_reporter_for_app")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_agent_app_factory_ignores_stale_top_level_agent_class(monkeypatch) -> None:
    import scripts.agent_app as agent_app

    class StaleAgent:
        def __init__(self, registry, planner=None, memory_writer=None) -> None:
            self.registry = registry
            self.planner = planner
            self.memory_writer = memory_writer

    monkeypatch.setattr(agent_app, "PlanAndExecuteAgent", StaleAgent)

    agent = agent_app._create_plan_execute_agent_for_app(
        registry="registry",
        planner="planner",
        reporter="reporter",
        memory_writer="memory_writer",
    )

    assert not isinstance(agent, StaleAgent)
    assert agent.registry == "registry"
    assert agent.planner == "planner"
    assert agent.reporter == "reporter"
    assert agent.memory_writer == "memory_writer"
