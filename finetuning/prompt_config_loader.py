"""
Name: prompt_config_loader
Input: prompt configuration YAML
Output: prompt template service
Usage: import from pipeline scripts
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_ENCODING = "utf-8"


class PromptConfigLoadError(Exception):
    """Raised when the prompt configuration cannot be loaded."""


class PromptConfigRepository:
    """Loads prompt configuration from a YAML file."""

    def load(self, config_file_path: Path) -> dict:
        try:
            import yaml
        except ImportError as error:
            raise PromptConfigLoadError(
                "PyYAML is required to load the prompt configuration file."
            ) from error

        if not config_file_path.exists():
            raise PromptConfigLoadError(
                f"Prompt configuration file not found: {config_file_path}"
            )

        with config_file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            return yaml.safe_load(input_file)


class PromptTemplateService:
    """Builds prompt text from configured templates."""

    def __init__(self, prompt_config: dict) -> None:
        self._prompt_config = prompt_config

    @property
    def component_order(self) -> tuple[str, ...]:
        return tuple(self._prompt_config["framework"]["component_order"])

    @property
    def component_names(self) -> dict[str, str]:
        return dict(self._prompt_config["framework"]["component_names"])

    @property
    def natural_combinations(self) -> dict[str, list[str]]:
        return dict(self._prompt_config["framework"].get("natural_combinations", {}))

    @property
    def dominance_order(self) -> tuple[str, ...]:
        return tuple(
            self._prompt_config["framework"].get(
                "dominance_order",
                self.component_order,
            )
        )

    @property
    def component_full_names(self) -> tuple[str, ...]:
        return tuple(self.component_names.values())

    def build_classification_system_prompt(self) -> str:
        return self._prompt_config["classification"]["system_template"].format(
            mindspace_reference=self._mindspace_reference,
        )

    def build_classification_user_prompt(self, domain: str, text: str) -> str:
        return self._prompt_config["classification"]["user_template"].format(
            domain=domain,
            text=text,
        )

    def build_validation_system_prompt(self) -> str:
        return self._prompt_config["validation"]["system_template"].format(
            mindspace_reference=self._mindspace_reference,
        )

    def build_generation_system_prompt(self) -> str:
        return self._prompt_config["generation"]["system_template"].format(
            mindspace_reference=self._mindspace_reference,
        )

    def build_generation_user_prompt(
        self,
        component: str,
        domain: str,
        mode: str,
        secondaries: list[str],
    ) -> str:
        generation_config = self._prompt_config["generation"]

        if mode == "isolated":
            mode_instruction = generation_config["isolated_mode_instruction"].format(
                component=component,
            )
        else:
            mode_instruction = generation_config["combined_mode_instruction"].format(
                component=component,
                secondaries=" and ".join(secondaries),
            )

        template_key = (
            "interface_user_template"
            if domain == "interface"
            else "complaint_user_template"
        )

        return generation_config[template_key].format(
            component=component,
            mode_instruction=mode_instruction,
        )

    @property
    def _mindspace_reference(self) -> str:
        return self._prompt_config["reference"]["mindspace"]
