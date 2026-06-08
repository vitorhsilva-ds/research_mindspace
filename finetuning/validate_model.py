#!/usr/bin/env python3
"""
Name: model_validation_pipeline
Input: validation text JSONL files, annotation JSONL file, prompt configuration YAML
Output: validation_results.json
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import httpx

from prompt_config_loader import PromptConfigRepository, PromptTemplateService


DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_PROMPT_CONFIG_PATH = DEFAULT_WORKSPACE_ROOT / "configs" / "mindspace_prompt_config.yaml"
DEFAULT_MODEL_API_URL = "http://VLLM-SERVER:8000/v1/chat/completions"
DEFAULT_MODEL_NAME = "model_artifacts/model_merged_mxfp4"
DEFAULT_INTERFACE_VALIDATION_FILE = Path("./validation_files/interface_validation_records.jsonl")
DEFAULT_COMPLAINT_VALIDATION_FILE = Path("./validation_files/complaint_validation_records.jsonl")
DEFAULT_ANNOTATIONS_FILE = Path("./validation_files/annotation_mindspace.jsonl")
DEFAULT_OUTPUT_FILE = Path("./validation_files/validation_results.json")
DEFAULT_CONCURRENCY = 4
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 20000
DEFAULT_RETRY_LIMIT = 3
DEFAULT_RETRY_DELAY = 4.0
DEFAULT_HTTP_TIMEOUT = 300.0
DEFAULT_ENCODING = "utf-8"


@dataclass(frozen=True)
class ValidationInputContract:
    interface_file_path: Path
    complaint_file_path: Path
    annotation_file_path: Path


@dataclass(frozen=True)
class ValidationOutputContract:
    output_file_path: Path


@dataclass(frozen=True)
class ValidationModelApiContract:
    url: str
    model_name: str
    temperature: float
    max_tokens: int
    retry_limit: int
    retry_delay: float
    http_timeout: float


@dataclass(frozen=True)
class ValidationExecutionContext:
    prompt_config_path: Path
    input_contract: ValidationInputContract
    output_contract: ValidationOutputContract
    model_api_contract: ValidationModelApiContract


class JsonLinesRepository:
    """Loads JSON Lines records."""

    def load_records(self, file_path: Path) -> list[dict]:
        if not file_path.exists():
            print(f"Input file not found: {file_path}")
            return []

        with file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            return [json.loads(line) for line in input_file if line.strip()]


class JsonRepository:
    """Saves JSON payloads."""

    def save(self, file_path: Path, payload: dict) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=DEFAULT_ENCODING,
        )


class PrimaryNormalizationService:
    """Normalizes predicted primary component values."""

    def __init__(self, prompt_service: PromptTemplateService) -> None:
        self._prompt_service = prompt_service
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

    def resolve_primary(self, raw_value: str, vector: dict) -> str:
        canonical = self.normalize_primary(raw_value)

        if canonical in set(self._prompt_service.component_names.values()):
            return canonical

        for component_code in self._prompt_service.dominance_order:
            if vector.get(component_code) == 1:
                return self._prompt_service.component_names[component_code]

        return "Unknown"

    def _build_normalization_map(self) -> dict[str, str | None]:
        mapping: dict[str, str | None] = {}

        for component_name in self._prompt_service.component_names.values():
            mapping[component_name.lower()] = component_name

        mapping.update(
            {
                "incentive": "Incentives",
                "norm": "Norms",
                "default": "Defaults",
                "commitment": "Commitments",
                "none": None,
                "nenhum": None,
                "sem evidência suficiente": None,
                "sem evidencia suficiente": None,
                "indeterminado": None,
                "insufficient evidence": None,
                "no evidence": None,
            }
        )
        return mapping


class JsonResponseExtractionService:
    """Extracts classification JSON from model responses."""

    def extract_result(self, message: dict) -> dict:
        raw_content = (message.get("content") or "").strip()

        if raw_content:
            return self._parse_json(raw_content)

        reasoning = (message.get("reasoning") or message.get("reasoning_content") or "").strip()
        candidates = self._extract_json_candidates(reasoning)

        for candidate in reversed(candidates):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid classification JSON found.")

    def _parse_json(self, text: str) -> dict:
        cleaned_text = text.strip()

        if cleaned_text.startswith("```"):
            parts = cleaned_text.split("```")
            cleaned_text = parts[1] if len(parts) > 1 else cleaned_text
            if cleaned_text.startswith("json"):
                cleaned_text = cleaned_text[4:]
            cleaned_text = cleaned_text.strip()

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].strip()

        return json.loads(cleaned_text)

    def _extract_json_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        index = 0

        while index < len(text):
            if text[index] == "{":
                depth = 0
                start = index

                for cursor in range(index, len(text)):
                    if text[cursor] == "{":
                        depth += 1
                    elif text[cursor] == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[start : cursor + 1]
                            if '"vector"' in candidate:
                                candidates.append(candidate)
                            index = cursor
                            break

            index += 1

        return candidates


class ValidationClassificationGateway:
    """Requests validation classifications from the configured model endpoint."""

    def __init__(
        self,
        prompt_service: PromptTemplateService,
        response_extractor: JsonResponseExtractionService,
        primary_normalizer: PrimaryNormalizationService,
    ) -> None:
        self._prompt_service = prompt_service
        self._response_extractor = response_extractor
        self._primary_normalizer = primary_normalizer

    async def classify_record(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        record: dict,
        model_api_contract: ValidationModelApiContract,
    ) -> tuple[str, dict | None]:
        record_id = record["id"]
        domain = record.get("source", record.get("domain", "interface"))
        text = record.get("text", "")

        payload = {
            "model": model_api_contract.model_name,
            "messages": [
                {"role": "system", "content": self._prompt_service.build_validation_system_prompt()},
                {
                    "role": "user",
                    "content": self._prompt_service.build_classification_user_prompt(
                        domain=domain,
                        text=text,
                    ),
                },
            ],
            "temperature": model_api_contract.temperature,
            "max_tokens": model_api_contract.max_tokens,
        }

        async with semaphore:
            for attempt in range(model_api_contract.retry_limit):
                try:
                    response = await asyncio.wait_for(
                        client.post(
                            model_api_contract.url,
                            json=payload,
                            headers={"Authorization": "Bearer EMPTY"},
                        ),
                        timeout=model_api_contract.http_timeout,
                    )
                    response.raise_for_status()
                    message = response.json()["choices"][0]["message"]
                    result = self._response_extractor.extract_result(message)
                    self._normalize_result(result)
                    return record_id, result

                except Exception:
                    if attempt < model_api_contract.retry_limit - 1:
                        await asyncio.sleep(model_api_contract.retry_delay + random.uniform(0, 2))
                    else:
                        return record_id, None

        return record_id, None

    def _normalize_result(self, result: dict) -> None:
        vector = result.get("vector", {})

        for component_code in self._prompt_service.component_order:
            if component_code not in vector:
                raise ValueError(f"Missing vector component: {component_code}")

            if vector[component_code] not in (0, 1):
                raise ValueError(f"Invalid vector value: {component_code}")

        if result.get("nudge_or_sludge", "").lower().strip() == "indeterminado":
            result["vector"] = {
                component_code: 0
                for component_code in self._prompt_service.component_order
            }
            result["primary"] = "Sem evidência suficiente"
            return

        result["primary"] = self._primary_normalizer.resolve_primary(
            raw_value=result.get("primary", ""),
            vector=result.get("vector", {}),
        )


class ValidationMetricService:
    """Computes validation metrics."""

    def cohen_kappa(self, y_true: list[int], y_pred: list[int]) -> float:
        sample_count = len(y_true)

        if sample_count == 0:
            return 0.0

        observed_agreement = (
            sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)
            / sample_count
        )
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


class ValidationReportService:
    """Builds validation result payloads."""

    def __init__(
        self,
        prompt_service: PromptTemplateService,
        metric_service: ValidationMetricService,
    ) -> None:
        self._prompt_service = prompt_service
        self._metric_service = metric_service

    def build_report(
        self,
        common_identifiers: list[str],
        annotations: dict[str, dict],
        model_results: dict[str, dict],
        failed_count: int,
    ) -> dict:
        indeterminate_identifiers = [
            record_id
            for record_id in common_identifiers
            if annotations[record_id].get("nudge_or_sludge") == "indeterminado"
            or annotations[record_id].get("primary") == "Sem evidência suficiente"
        ]
        valid_identifiers = [
            record_id
            for record_id in common_identifiers
            if record_id in model_results and record_id not in indeterminate_identifiers
        ]
        indeterminate_classified = [
            record_id
            for record_id in indeterminate_identifiers
            if record_id in model_results
        ]

        kappas: dict[str, float] = {}
        accuracies: dict[str, float] = {}

        for component_code in self._prompt_service.component_order:
            y_true = [
                annotations[record_id]["vector"][component_code]
                for record_id in valid_identifiers
            ]
            y_pred = [
                model_results[record_id]["vector"][component_code]
                for record_id in valid_identifiers
            ]
            kappas[component_code] = self._metric_service.cohen_kappa(y_true, y_pred)
            accuracies[component_code] = (
                sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)
                / len(y_true)
                if y_true
                else 0.0
            )

        kappa_mean = sum(kappas.values()) / len(kappas) if kappas else 0.0
        primary_match_count = sum(
            1
            for record_id in valid_identifiers
            if annotations[record_id]["primary"] == model_results[record_id].get("primary", "")
        )
        binary_identifiers = valid_identifiers + indeterminate_classified
        binary_correct_count = sum(
            1
            for record_id in binary_identifiers
            if (annotations[record_id]["nudge_or_sludge"] in ("nudge", "sludge"))
            == (model_results[record_id].get("nudge_or_sludge", "") in ("nudge", "sludge"))
        )

        return {
            "summary": {
                "n_with_mechanism": len(valid_identifiers),
                "n_indeterminate": len(indeterminate_identifiers),
                "n_failed": failed_count,
                "kappa_mean": kappa_mean,
                "kappa_interpretation": self._metric_service.interpret_kappa(kappa_mean),
                "primary_accuracy": (
                    primary_match_count / len(valid_identifiers)
                    if valid_identifiers
                    else 0
                ),
                "binary_detection": (
                    binary_correct_count / len(binary_identifiers)
                    if binary_identifiers
                    else 0
                ),
            },
            "kappa_by_component": {
                self._prompt_service.component_names[component_code]: {
                    "kappa": kappas[component_code],
                    "accuracy": accuracies[component_code],
                    "interpretation": self._metric_service.interpret_kappa(
                        kappas[component_code]
                    ),
                }
                for component_code in self._prompt_service.component_order
            },
            "per_record": {
                record_id: {
                    "human": annotations[record_id],
                    "model": model_results.get(record_id),
                    "primary_match": (
                        annotations[record_id]["primary"]
                        == model_results.get(record_id, {}).get("primary", "")
                    ),
                }
                for record_id in valid_identifiers + indeterminate_classified
            },
        }


class ModelValidationOrchestrator:
    """Coordinates the model validation lifecycle."""

    def __init__(
        self,
        jsonl_repository: JsonLinesRepository,
        json_repository: JsonRepository,
        response_extractor: JsonResponseExtractionService,
        metric_service: ValidationMetricService,
    ) -> None:
        self._jsonl_repository = jsonl_repository
        self._json_repository = json_repository
        self._response_extractor = response_extractor
        self._metric_service = metric_service

    async def execute(self, execution_context: ValidationExecutionContext) -> None:
        prompt_config = PromptConfigRepository().load(execution_context.prompt_config_path)
        prompt_service = PromptTemplateService(prompt_config)
        primary_normalizer = PrimaryNormalizationService(prompt_service)
        classification_gateway = ValidationClassificationGateway(
            prompt_service=prompt_service,
            response_extractor=self._response_extractor,
            primary_normalizer=primary_normalizer,
        )
        report_service = ValidationReportService(prompt_service, self._metric_service)

        texts = self._load_texts(execution_context.input_contract)
        annotations = self._load_annotations(execution_context.input_contract.annotation_file_path)
        common_identifiers = sorted(set(texts) & set(annotations))

        print("Validation input loaded.")
        print(f"Text records: {len(texts)}")
        print(f"Annotation records: {len(annotations)}")
        print(f"Records to classify: {len(common_identifiers)}")

        if not common_identifiers:
            print("No overlapping validation records found.")
            return

        semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)
        model_results: dict[str, dict] = {}
        failed_count = 0
        limits = httpx.Limits(
            max_connections=DEFAULT_CONCURRENCY + 2,
            max_keepalive_connections=DEFAULT_CONCURRENCY,
        )

        async with httpx.AsyncClient(
            timeout=execution_context.model_api_contract.http_timeout,
            limits=limits,
        ) as client:
            tasks = [
                classification_gateway.classify_record(
                    client=client,
                    semaphore=semaphore,
                    record=texts[record_id],
                    model_api_contract=execution_context.model_api_contract,
                )
                for record_id in common_identifiers
            ]

            for index, coroutine in enumerate(asyncio.as_completed(tasks), 1):
                record_id, result = await coroutine
                if result is not None:
                    model_results[record_id] = result
                    active_count = sum(1 for value in result["vector"].values() if value == 1)
                    print(
                        f"  [{index:02d}/{len(common_identifiers)}] "
                        f"{record_id} primary={result['primary'][:12]:<12} "
                        f"active={active_count} confidence={result['confidence']}"
                    )
                else:
                    failed_count += 1
                    print(f"  [{index:02d}/{len(common_identifiers)}] {record_id} failed")

        report = report_service.build_report(
            common_identifiers=common_identifiers,
            annotations=annotations,
            model_results=model_results,
            failed_count=failed_count,
        )
        self._json_repository.save(execution_context.output_contract.output_file_path, report)

        print("Execution completed.")
        print(f"Output file: {execution_context.output_contract.output_file_path}")

    def _load_texts(self, input_contract: ValidationInputContract) -> dict[str, dict]:
        records: dict[str, dict] = {}

        for file_path in (input_contract.interface_file_path, input_contract.complaint_file_path):
            for record in self._jsonl_repository.load_records(file_path):
                records[record["id"]] = record

        return records

    def _load_annotations(self, annotation_file_path: Path) -> dict[str, dict]:
        annotations: dict[str, dict] = {}

        for record in self._jsonl_repository.load_records(annotation_file_path):
            annotations[record["id"]] = record["annotation"]

        return annotations


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a MINDSPACE classification model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROMPT_CONFIG_PATH)
    parser.add_argument("--model-api-url", type=str, default=DEFAULT_MODEL_API_URL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--interface-file", type=Path, default=DEFAULT_INTERFACE_VALIDATION_FILE)
    parser.add_argument("--complaint-file", type=Path, default=DEFAULT_COMPLAINT_VALIDATION_FILE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> ValidationExecutionContext:
    return ValidationExecutionContext(
        prompt_config_path=args.config,
        input_contract=ValidationInputContract(
            interface_file_path=args.interface_file,
            complaint_file_path=args.complaint_file,
            annotation_file_path=args.annotations,
        ),
        output_contract=ValidationOutputContract(output_file_path=args.output),
        model_api_contract=ValidationModelApiContract(
            url=args.model_api_url,
            model_name=args.model_name,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            retry_limit=DEFAULT_RETRY_LIMIT,
            retry_delay=DEFAULT_RETRY_DELAY,
            http_timeout=DEFAULT_HTTP_TIMEOUT,
        ),
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = ModelValidationOrchestrator(
        jsonl_repository=JsonLinesRepository(),
        json_repository=JsonRepository(),
        response_extractor=JsonResponseExtractionService(),
        metric_service=ValidationMetricService(),
    )
    asyncio.run(orchestrator.execute(execution_context))


if __name__ == "__main__":
    main()
