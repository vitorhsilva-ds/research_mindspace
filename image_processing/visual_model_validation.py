#!/usr/bin/env python3
"""
Name: visual_model_validation_pipeline
Input: validation image records, annotations, and visual prompt configuration YAML
Output: validation_results_qwenvl.json
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from dataclasses import dataclass
from pathlib import Path

import httpx

from visual_mindspace_pipeline_core import (
    ComponentVectorService,
    ImageEncodingService,
    JsonLinesRepository,
    JsonRepository,
    JsonResponseExtractionService,
    MetricComputationService,
    PrimaryComponentResolutionService,
)
from visual_mindspace_prompt_loader import (
    DEFAULT_VISUAL_PROMPT_CONFIG_PATH,
    VisualPromptConfiguration,
    VisualPromptConfigurationRepository,
)


DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_MODEL_API_URL = "http://VLLM-SERVER:8001/v1/chat/completions"
DEFAULT_MODEL_NAME = "models/Qwen3-VL-8B-Instruct"
DEFAULT_IMAGE_BASE = DEFAULT_WORKSPACE_ROOT / "image_capture" / "wayback_captures" / "2019"
DEFAULT_VALIDATION_FILE = DEFAULT_WORKSPACE_ROOT / "image_processing" / "validation_files" / "visual_validation_records.jsonl"
DEFAULT_ANNOTATIONS_FILE = DEFAULT_WORKSPACE_ROOT / "image_processing" / "validation_files" / "visual_annotations.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_WORKSPACE_ROOT / "image_processing" / "validation_files" / "validation_results_qwenvl.json"
DEFAULT_CONCURRENCY = 2
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 3000
DEFAULT_RETRY_LIMIT = 3
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_HTTP_TIMEOUT = 180.0


@dataclass(frozen=True)
class ValidationInputContract:
    validation_file_path: Path
    annotations_file_path: Path
    image_base_path: Path


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
    workers: int


class ValidationDataRepository:
    """Loads validation records and annotations."""

    def __init__(self, jsonl_repository: JsonLinesRepository) -> None:
        self._jsonl_repository = jsonl_repository

    def load_images(self, input_contract: ValidationInputContract) -> dict[str, dict]:
        records: dict[str, dict] = {}
        for record in self._jsonl_repository.load_records(input_contract.validation_file_path):
            normalized_record = dict(record)
            file_name = normalized_record.get("filename", "")
            page_type = normalized_record.get("page_type", "")
            if input_contract.image_base_path:
                normalized_record["image_path"] = str(input_contract.image_base_path / page_type / "slices" / file_name)
            records[normalized_record["id"]] = normalized_record
        return records

    def load_annotations(self, annotations_file_path: Path) -> dict[str, dict]:
        annotations: dict[str, dict] = {}
        for record in self._jsonl_repository.load_records(annotations_file_path):
            annotations[record["id"]] = record["annotation"]
        return annotations


class VisualValidationGateway:
    """Requests validation classifications from the configured model endpoint."""

    def __init__(
        self,
        prompt_config: VisualPromptConfiguration,
        image_encoder: ImageEncodingService,
        response_extractor: JsonResponseExtractionService,
        vector_service: ComponentVectorService,
        primary_service: PrimaryComponentResolutionService,
    ) -> None:
        self._prompt_config = prompt_config
        self._image_encoder = image_encoder
        self._response_extractor = response_extractor
        self._vector_service = vector_service
        self._primary_service = primary_service

    async def classify_record(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        record: dict,
        model_api_contract: ValidationModelApiContract,
    ) -> tuple[str, dict | None]:
        record_id = record["id"]
        image_path = Path(record.get("image_path", ""))
        if not image_path.exists():
            return (
                record_id,
                {
                    "error": f"Image file not found: {image_path}",
                    "vector": {component_code: 0 for component_code in self._prompt_config.component_order},
                    "primary": "Unknown",
                    "nudge_or_sludge": "indeterminado",
                    "confidence": "baixo",
                },
            )
        image_payload = self._image_encoder.encode(image_path)
        payload = {
            "model": model_api_contract.model_name,
            "messages": [
                {"role": "system", "content": self._prompt_config.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_payload.mime_type};base64,{image_payload.encoded_image}"},
                        },
                        {"type": "text", "text": self._prompt_config.build_validation_user_prompt()},
                    ],
                },
            ],
            "temperature": model_api_contract.temperature,
            "max_tokens": model_api_contract.max_tokens,
        }
        async with semaphore:
            for attempt in range(model_api_contract.retry_limit):
                try:
                    response = await asyncio.wait_for(
                        client.post(model_api_contract.url, json=payload, headers={"Authorization": "Bearer EMPTY"}),
                        timeout=model_api_contract.http_timeout,
                    )
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    raw_content = (choice["message"].get("content") or "").strip()
                    finish_reason = choice.get("finish_reason", "")
                    parsed_payload, _ = self._response_extractor.parse_or_repair(raw_content, finish_reason=finish_reason)
                    if parsed_payload is None:
                        raise ValueError("Unable to parse model response JSON.")
                    return record_id, self._normalize_result(parsed_payload)
                except Exception as error:
                    if attempt < model_api_contract.retry_limit - 1:
                        await asyncio.sleep(model_api_contract.retry_delay + random.uniform(0, 2))
                    else:
                        print(f"  [{record_id}] failed: {type(error).__name__}: {str(error)[:120]}")
                        return record_id, None
        return record_id, None

    def _normalize_result(self, parsed_payload: dict) -> dict:
        mindspace_payload = parsed_payload.get("mindspace", {})
        vector = self._vector_service.extract_vector(mindspace_payload, self._prompt_config.component_order)
        raw_primary = parsed_payload.get("primary", "")
        nudge_or_sludge = str(parsed_payload.get("nudge_or_sludge", "")).lower().strip()
        if nudge_or_sludge not in ("nudge", "sludge", "indeterminado"):
            if "nudge" in nudge_or_sludge:
                nudge_or_sludge = "nudge"
            elif "sludge" in nudge_or_sludge:
                nudge_or_sludge = "sludge"
            else:
                nudge_or_sludge = "indeterminado"
        if nudge_or_sludge == "indeterminado":
            vector = {component_code: 0 for component_code in self._prompt_config.component_order}
            raw_primary = "Sem evidência suficiente"
        primary = self._primary_service.resolve_primary(raw_primary, vector)
        return {
            "vector": vector,
            "primary": primary,
            "nudge_or_sludge": nudge_or_sludge,
            "confidence": "alto",
            "mindspace_raw": mindspace_payload,
            "image_metadata": parsed_payload.get("image_metadata", {}),
        }


class ValidationReportService:
    """Builds validation report payloads."""

    def __init__(self, prompt_config: VisualPromptConfiguration, metric_service: MetricComputationService) -> None:
        self._prompt_config = prompt_config
        self._metric_service = metric_service

    def build_report(self, annotations: dict[str, dict], model_results: dict[str, dict], common_identifiers: list[str], failed_count: int) -> dict:
        nil_identifiers = [
            record_id
            for record_id in common_identifiers
            if annotations[record_id].get("nudge_or_sludge") == "indeterminado"
            or sum(annotations[record_id].get("vector", {}).values()) == 0
        ]
        valid_identifiers_all = [record_id for record_id in common_identifiers if record_id not in nil_identifiers]
        nil_classified = [record_id for record_id in nil_identifiers if record_id in model_results]
        valid_identifiers = [record_id for record_id in valid_identifiers_all if record_id in model_results]
        kappas: dict[str, float] = {}
        accuracies: dict[str, float] = {}
        for component_code in self._prompt_config.component_order:
            y_true = [annotations[record_id]["vector"][component_code] for record_id in valid_identifiers]
            y_pred = [model_results[record_id]["vector"][component_code] for record_id in valid_identifiers]
            kappas[component_code] = self._metric_service.cohen_kappa(y_true, y_pred)
            accuracies[component_code] = (
                sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted) / len(y_true)
                if y_true
                else 0.0
            )
        kappa_mean = sum(kappas.values()) / len(kappas) if kappas else 0.0
        primary_match = sum(
            1
            for record_id in valid_identifiers
            if annotations[record_id]["primary"] == model_results[record_id].get("primary", "")
        )
        binary_identifiers = valid_identifiers + nil_classified
        binary_correct = sum(
            1
            for record_id in binary_identifiers
            if (annotations[record_id]["nudge_or_sludge"] in ("nudge", "sludge"))
            == (model_results[record_id].get("nudge_or_sludge", "") in ("nudge", "sludge"))
        )
        nil_correct = (
            sum(
                1
                for record_id in nil_classified
                if model_results[record_id].get("nudge_or_sludge") == "indeterminado"
                or sum(model_results[record_id].get("vector", {}).values()) == 0
            )
            if nil_classified
            else 0
        )
        return {
            "summary": {
                "n_with_mindspace": len(valid_identifiers),
                "n_nil": len(nil_identifiers),
                "n_failed": failed_count,
                "kappa_mean": kappa_mean,
                "kappa_interpretation": self._metric_service.interpret_kappa(kappa_mean),
                "primary_accuracy": primary_match / len(valid_identifiers) if valid_identifiers else 0,
                "binary_detection": binary_correct / len(binary_identifiers) if binary_identifiers else 0,
                "nil_detection": nil_correct / len(nil_classified) if nil_classified else None,
            },
            "kappa_by_component": {
                self._prompt_config.component_names[component_code]: {
                    "kappa": kappas[component_code],
                    "accuracy": accuracies[component_code],
                    "interpretation": self._metric_service.interpret_kappa(kappas[component_code]),
                }
                for component_code in self._prompt_config.component_order
            },
            "per_evidence": {
                record_id: {
                    "human": annotations[record_id],
                    "model": model_results.get(record_id),
                    "primary_match": annotations[record_id]["primary"] == model_results.get(record_id, {}).get("primary", ""),
                }
                for record_id in valid_identifiers + nil_classified
            },
        }


class VisualValidationOrchestrator:
    """Coordinates visual model validation."""

    def __init__(
        self,
        prompt_repository: VisualPromptConfigurationRepository,
        data_repository: ValidationDataRepository,
        json_repository: JsonRepository,
        image_encoder: ImageEncodingService,
        response_extractor: JsonResponseExtractionService,
        vector_service: ComponentVectorService,
        metric_service: MetricComputationService,
    ) -> None:
        self._prompt_repository = prompt_repository
        self._data_repository = data_repository
        self._json_repository = json_repository
        self._image_encoder = image_encoder
        self._response_extractor = response_extractor
        self._vector_service = vector_service
        self._metric_service = metric_service

    async def execute(self, execution_context: ValidationExecutionContext) -> None:
        prompt_config = self._prompt_repository.load(execution_context.prompt_config_path)
        primary_service = PrimaryComponentResolutionService(prompt_config.component_names, prompt_config.dominance_order)
        validation_gateway = VisualValidationGateway(
            prompt_config,
            self._image_encoder,
            self._response_extractor,
            self._vector_service,
            primary_service,
        )
        report_service = ValidationReportService(prompt_config, self._metric_service)
        images = self._data_repository.load_images(execution_context.input_contract)
        annotations = self._data_repository.load_annotations(execution_context.input_contract.annotations_file_path)
        common_identifiers = sorted(set(images.keys()) & set(annotations.keys()))
        print("Validation input loaded.")
        print(f"Image records: {len(images)}")
        print(f"Annotation records: {len(annotations)}")
        print(f"Records to classify: {len(common_identifiers)}")
        if not common_identifiers:
            print("No overlapping records found.")
            return
        semaphore = asyncio.Semaphore(execution_context.workers)
        model_results: dict[str, dict] = {}
        failed_count = 0
        limits = httpx.Limits(max_connections=execution_context.workers + 2, max_keepalive_connections=execution_context.workers)
        async with httpx.AsyncClient(timeout=execution_context.model_api_contract.http_timeout, limits=limits) as client:
            tasks = [
                validation_gateway.classify_record(
                    client,
                    semaphore,
                    images[record_id],
                    execution_context.model_api_contract,
                )
                for record_id in common_identifiers
            ]
            for index, coroutine in enumerate(asyncio.as_completed(tasks), 1):
                record_id, result = await coroutine
                if result is not None:
                    model_results[record_id] = result
                    active_count = sum(1 for value in result["vector"].values() if value == 1)
                    print(f"  [{index:02d}/{len(common_identifiers)}] {record_id} primary={result.get('primary', '?')[:14]:<14} active={active_count} ns={result.get('nudge_or_sludge', '?')}")
                else:
                    failed_count += 1
                    print(f"  [{index:02d}/{len(common_identifiers)}] {record_id} failed")
        report = report_service.build_report(annotations, model_results, common_identifiers, failed_count)
        self._json_repository.save(execution_context.output_contract.output_file_path, report)
        print("Execution completed.")
        print(f"Output file: {execution_context.output_contract.output_file_path}")
        print(f"Kappa mean: {report['summary']['kappa_mean']:.3f}")
        print(f"Primary accuracy: {report['summary']['primary_accuracy']:.3f}")
        print(f"Binary detection: {report['summary']['binary_detection']:.3f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate visual MINDSPACE model outputs against annotation records.")
    parser.add_argument("--config", type=Path, default=DEFAULT_VISUAL_PROMPT_CONFIG_PATH)
    parser.add_argument("--model-api-url", type=str, default=DEFAULT_MODEL_API_URL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image-base", type=Path, default=DEFAULT_IMAGE_BASE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--workers", type=int, default=DEFAULT_CONCURRENCY, choices=[1, 2, 3])
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> ValidationExecutionContext:
    return ValidationExecutionContext(
        prompt_config_path=args.config,
        input_contract=ValidationInputContract(args.validation_file, args.annotations, args.image_base),
        output_contract=ValidationOutputContract(args.output),
        model_api_contract=ValidationModelApiContract(
            url=args.model_api_url,
            model_name=args.model_name,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            retry_limit=DEFAULT_RETRY_LIMIT,
            retry_delay=DEFAULT_RETRY_DELAY,
            http_timeout=DEFAULT_HTTP_TIMEOUT,
        ),
        workers=args.workers,
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = VisualValidationOrchestrator(
        prompt_repository=VisualPromptConfigurationRepository(),
        data_repository=ValidationDataRepository(JsonLinesRepository()),
        json_repository=JsonRepository(),
        image_encoder=ImageEncodingService(),
        response_extractor=JsonResponseExtractionService(),
        vector_service=ComponentVectorService(),
        metric_service=MetricComputationService(),
    )
    asyncio.run(orchestrator.execute(execution_context))


if __name__ == "__main__":
    main()
