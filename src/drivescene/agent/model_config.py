from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0
    reasoning_effort: str | None = None
    store: bool | None = None
    use_responses_api: bool = False

    def to_chat_openai_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.store is not None:
            kwargs["store"] = self.store
        if self.use_responses_api:
            kwargs["use_responses_api"] = True
        return kwargs


def load_model_config(config_path: Path | str = "config/model.yml") -> ModelConfig:
    path = Path(config_path)
    if not path.exists():
        return ModelConfig(
            model=os.environ.get("DRIVESCENE_AGENT_MODEL", "deepseek-v3.2"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            temperature=0,
        )

    text = path.read_text(encoding="utf-8").strip()
    parsed = yaml.safe_load(text)
    if isinstance(parsed, dict):
        raw = parsed.get("model", parsed)
        if isinstance(raw, dict):
            return _config_from_mapping(raw)

    return _config_from_mapping(_parse_assignment_style(text))


def _config_from_mapping(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        model=str(raw.get("model") or os.environ.get("DRIVESCENE_AGENT_MODEL", "deepseek-v3.2")),
        api_key=raw.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        base_url=raw.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
        temperature=float(raw.get("temperature", 0)),
        reasoning_effort=raw.get("reasoning_effort") or None,
        store=_optional_bool(raw.get("store")),
        use_responses_api=_optional_bool(raw.get("use_responses_api")) is True,
    )


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _parse_assignment_style(text: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value")
        for match in re.finditer(
            r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(?P<value>[^"]*)"',
            text,
        )
    }
