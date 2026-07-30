from pathlib import Path

from drivescene.agent.model_config import load_model_config


def test_load_model_config_supports_assignment_style_file(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        'model="test-model", api_key="test-key", base_url="https://example.test/v1",',
        encoding="utf-8",
    )

    config = load_model_config(config_path)

    assert config.model == "test-model"
    assert config.api_key == "test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.temperature == 0


def test_load_model_config_supports_yaml_model_section(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        """
model:
  model: test-model
  api_key: test-key
  base_url: https://example.test/v1
  temperature: 0.2
""",
        encoding="utf-8",
    )

    config = load_model_config(config_path)

    assert config.model == "test-model"
    assert config.api_key == "test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.temperature == 0.2


def test_load_model_config_uses_environment_api_key_when_file_omits_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "model.yml"
    config_path.write_text(
        """
model:
  model: test-model
  base_url: https://example.test/v1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")

    config = load_model_config(config_path)

    assert config.model == "test-model"
    assert config.api_key == "env-test-key"
    assert config.base_url == "https://example.test/v1"
