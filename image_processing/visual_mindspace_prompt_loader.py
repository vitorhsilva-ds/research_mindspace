#!/usr/bin/env python3
"""
Name: visual_mindspace_prompt_loader
Input: visual MINDSPACE prompt configuration YAML
Output: prompt templates and framework metadata
Usage: imported by visual extraction and validation scripts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os


DEFAULT_ENCODING = "utf-8"
DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_VISUAL_PROMPT_CONFIG_PATH = Path(
    DEFAULT_WORKSPACE_ROOT / "configs" / "visual_mindspace_prompt_config.yaml"
)


class VisualPromptConfigurationError(Exception):
    """Raised when the visual prompt configuration cannot be loaded."""


@dataclass(frozen=True)
class VisualPromptConfiguration:
    raw_config: dict[str, Any]

    @property
    def component_order(self) -> tuple[str, ...]:
        return tuple(self.raw_config["framework"]["component_order"])

    @property
    def component_names(self) -> dict[str, str]:
        return dict(self.raw_config["framework"]["component_names"])

    @property
    def dominance_order(self) -> tuple[str, ...]:
        return tuple(self.raw_config["framework"]["dominance_order"])

    @property
    def system_prompt(self) -> str:
        return self.raw_config["reference"]["system_prompt"].strip()

    @property
    def component_reference(self) -> str:
        return self.raw_config["reference"]["component_reference"].strip()

    @property
    def visual_calibration(self) -> str:
        return self.raw_config["calibration"]["visual_rules"].strip()

    @property
    def extraction_schema(self) -> str:
        return self.raw_config["schemas"]["extraction_json_schema"].strip()

    def build_extraction_user_prompt(
        self,
        record_id: str,
        page_type: str,
        year: int | str,
    ) -> str:
        return self.raw_config["templates"]["extraction_user_prompt"].format(
            record_id=record_id,
            page_type=page_type,
            year=year,
            component_reference=self.component_reference,
            visual_calibration=self.visual_calibration,
            json_schema=self.extraction_schema,
        )

    def build_validation_user_prompt(self) -> str:
        return self.raw_config["templates"]["validation_user_prompt"].format(
            component_reference=self.component_reference,
            visual_calibration=self.visual_calibration,
            json_schema=self.extraction_schema,
        )


class VisualPromptConfigurationRepository:
    """Loads visual prompt configuration from YAML."""

    def load(self, config_file_path: Path) -> VisualPromptConfiguration:
        try:
            import yaml
        except ImportError as error:
            raise VisualPromptConfigurationError(
                "PyYAML is required to load the visual prompt configuration."
            ) from error

        if not config_file_path.exists():
            raise VisualPromptConfigurationError(
                f"Visual prompt configuration file not found: {config_file_path}"
            )

        with config_file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            raw_config = yaml.safe_load(input_file)

        if not isinstance(raw_config, dict):
            raise VisualPromptConfigurationError(
                "Visual prompt configuration root must be a mapping."
            )

        return VisualPromptConfiguration(raw_config=raw_config)
