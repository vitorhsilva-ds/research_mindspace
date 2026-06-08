#!/usr/bin/env python3
"""
Name: visual_mindspace_pipeline_core
Input: image files, JSONL records, model API responses
Output: structured visual extraction records and validation metrics
Usage: imported by visual MINDSPACE pipeline scripts
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_ENCODING = "utf-8"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MIME_TYPES_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class VisualMindspacePipelineError(Exception):
    """Base exception for visual MINDSPACE pipeline failures."""


@dataclass(frozen=True)
class ImagePayload:
    encoded_image: str
    mime_type: str


class ImageEncodingService:
    """Encodes image files into base64 payloads."""

    def encode(self, image_file_path: Path) -> ImagePayload:
        extension = image_file_path.suffix.lower()
        mime_type = MIME_TYPES_BY_EXTENSION.get(extension, "image/png")

        with image_file_path.open("rb") as input_file:
            encoded_image = base64.b64encode(input_file.read()).decode(DEFAULT_ENCODING)

        return ImagePayload(encoded_image=encoded_image, mime_type=mime_type)


class JsonLinesRepository:
    """Loads and saves JSON Lines records."""

    def load_records(self, file_path: Path) -> list[dict]:
        records: list[dict] = []

        if not file_path.exists():
            return records

        with file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            for line in input_file:
                stripped_line = line.strip()
                if stripped_line:
                    records.append(json.loads(stripped_line))

        return records

    def save_records(self, file_path: Path, records: Iterable[dict]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding=DEFAULT_ENCODING) as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


class JsonRepository:
    """Loads and saves JSON payloads."""

    def load(self, file_path: Path) -> dict:
        return json.loads(file_path.read_text(encoding=DEFAULT_ENCODING))

    def save(self, file_path: Path, payload) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=DEFAULT_ENCODING,
        )


class JsonResponseExtractionService:
    """Extracts structured JSON from model text responses."""

    def try_parse_json(self, text: str) -> dict | None:
        if not text:
            return None

        cleaned_text = self._remove_markdown_fences(text)

        try:
            return json.loads(cleaned_text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None

        return None

    def parse_or_repair(self, text: str, finish_reason: str | None = None) -> tuple[dict | None, bool]:
        parsed = self.try_parse_json(text)

        if parsed is not None:
            return parsed, False

        if finish_reason == "length":
            repaired_text = self.repair_truncated_json(text)
            repaired_parsed = self.try_parse_json(repaired_text)

            if repaired_parsed is not None:
                return repaired_parsed, True

        return None, False

    def repair_truncated_json(self, text: str) -> str:
        text = text.rstrip()
        stack: list[str] = []
        in_string = False
        escape = False

        for character in text:
            if escape:
                escape = False
                continue
            if character == "\\" and in_string:
                escape = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if not in_string:
                if character in "{[":
                    stack.append("}" if character == "{" else "]")
                elif character in "}]":
                    if stack and stack[-1] == character:
                        stack.pop()

        while stack:
            text += stack.pop()

        return text

    def _remove_markdown_fences(self, text: str) -> str:
        cleaned_text = text.strip()
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text, flags=re.MULTILINE)
        return cleaned_text.strip()


class ComponentVectorService:
    """Converts MINDSPACE component structures into binary vectors."""

    def extract_vector(self, mindspace_payload: dict, component_order: tuple[str, ...]) -> dict[str, int]:
        vector: dict[str, int] = {}

        for component_code in component_order:
            component_payload = mindspace_payload.get(component_code, {})
            if isinstance(component_payload, dict):
                vector[component_code] = 1 if component_payload.get("presente") is True else 0
            else:
                vector[component_code] = 0

        return vector


class PrimaryComponentResolutionService:
    """Normalizes and resolves primary MINDSPACE components."""

    def __init__(self, component_names: dict[str, str], dominance_order: tuple[str, ...]) -> None:
        self._component_names = component_names
        self._dominance_order = dominance_order
        self._normalization_map = self._build_normalization_map()

    def normalize_primary(self, raw_value: str) -> str | None:
        if not raw_value:
            return None

        key = raw_value.strip().lower()
        if key in self._normalization_map:
            return self._normalization_map[key]

        for variant, canonical in self._normalization_map.items():
            if variant in key:
                return canonical

        return None

    def resolve_primary(self, raw_value: str, vector: dict[str, int]) -> str:
        canonical = self.normalize_primary(raw_value)
        if canonical and canonical in set(self._component_names.values()):
            return canonical

        for component_code in self._dominance_order:
            if vector.get(component_code) == 1:
                return self._component_names[component_code]

        return "Unknown"

    def _build_normalization_map(self) -> dict[str, str | None]:
        mapping: dict[str, str | None] = {}
        for component_name in self._component_names.values():
            mapping[component_name.lower()] = component_name

        mapping.update({
            "incentive": "Incentives",
            "norm": "Norms",
            "default": "Defaults",
            "commitment": "Commitments",
            "mensageiro": "Messenger",
            "incentivos": "Incentives",
            "normas": "Norms",
            "padrões": "Defaults",
            "padroes": "Defaults",
            "saliência": "Salience",
            "saliencia": "Salience",
            "ativação": "Priming",
            "ativacao": "Priming",
            "afeto": "Affect",
            "compromissos": "Commitments",
            "identidade": "Ego",
            "sem evidência suficiente": None,
            "sem evidencia suficiente": None,
            "indeterminado": None,
            "nenhum": None,
            "none": None,
            "unknown": None,
        })
        return mapping


class MetricComputationService:
    """Computes validation metrics."""

    def cohen_kappa(self, y_true: list[int], y_pred: list[int]) -> float:
        sample_count = len(y_true)
        if sample_count == 0:
            return 0.0

        observed_agreement = sum(
            1 for actual, predicted in zip(y_true, y_pred) if actual == predicted
        ) / sample_count
        true_positive_rate = sum(y_true) / sample_count
        predicted_positive_rate = sum(y_pred) / sample_count
        expected_agreement = (
            true_positive_rate * predicted_positive_rate
            + (1 - true_positive_rate) * (1 - predicted_positive_rate)
        )
        if expected_agreement == 1.0:
            return 1.0
        return (observed_agreement - expected_agreement) / (1 - expected_agreement)

    def interpret_kappa(self, value: float) -> str:
        if value < 0:
            return "below_chance"
        if value < 0.20:
            return "slight"
        if value < 0.40:
            return "fair"
        if value < 0.60:
            return "moderate"
        if value < 0.80:
            return "substantial"
        return "near_perfect"


def get_timestamped_file_name(prefix: str, suffix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{suffix.lstrip('.')}"
