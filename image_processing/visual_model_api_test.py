#!/usr/bin/env python3
"""
Name: visual_model_api_test_pipeline
Input: validation image directory and visual prompt configuration YAML
Output: per-image JSON files, batch JSON file, classifier JSONL export
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

from visual_mindspace_pipeline_core import (
    DEFAULT_ENCODING,
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageEncodingService,
    JsonRepository,
    JsonResponseExtractionService,
)
from visual_mindspace_prompt_loader import (
    DEFAULT_VISUAL_PROMPT_CONFIG_PATH,
    VisualPromptConfiguration,
    VisualPromptConfigurationRepository,
)


DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_MODEL_API_BASE_URL = "http://localhost:8001/v1"
DEFAULT_MODEL_NAME = "models/Qwen3-VL-8B-Instruct"
DEFAULT_IMAGE_DIR = DEFAULT_WORKSPACE_ROOT / "image_processing" / "data" / "validation"
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE_ROOT / "image_processing" / "data" / "validation" / "vlm_output"
DEFAULT_CLASSIFIER_EXPORT_FILE_NAME = "corpus_para_classificador.jsonl"
DEFAULT_MAX_TOKENS = 2500
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_WORKERS = 2


@dataclass(frozen=True)
class TestModelApiContract:
    base_url: str
    model_name: str
    max_tokens: int
    request_timeout: int

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass(frozen=True)
class ImageTestInputContract:
    image_dir: Path
    single_image: str | None


@dataclass(frozen=True)
class ImageTestOutputContract:
    output_dir: Path
    classifier_export_file_name: str


@dataclass(frozen=True)
class ImageTestExecutionContext:
    prompt_config_path: Path
    model_api_contract: TestModelApiContract
    input_contract: ImageTestInputContract
    output_contract: ImageTestOutputContract
    workers: int


class ImageDiscoveryService:
    """Discovers image files for visual model testing."""

    def discover(self, input_contract: ImageTestInputContract) -> list[Path]:
        if input_contract.single_image:
            single_path = Path(input_contract.single_image)
            if not single_path.is_absolute():
                single_path = input_contract.image_dir / input_contract.single_image
            if not single_path.exists():
                raise FileNotFoundError(f"Image file not found: {single_path}")
            return [single_path]

        image_paths = sorted(
            image_file_path
            for image_file_path in input_contract.image_dir.iterdir()
            if image_file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {input_contract.image_dir}")
        return image_paths


class VisualApiTestGateway:
    """Calls the configured visual model endpoint."""

    def __init__(
        self,
        prompt_config: VisualPromptConfiguration,
        image_encoder: ImageEncodingService,
        response_extractor: JsonResponseExtractionService,
    ) -> None:
        self._prompt_config = prompt_config
        self._image_encoder = image_encoder
        self._response_extractor = response_extractor

    def classify_image(self, image_file_path: Path, model_api_contract: TestModelApiContract) -> dict:
        image_payload = self._image_encoder.encode(image_file_path)
        start_time = time.time()
        payload = json.dumps(
            {
                "model": model_api_contract.model_name,
                "messages": [
                    {"role": "system", "content": self._prompt_config.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_payload.mime_type};base64,{image_payload.encoded_image}"
                                },
                            },
                            {"type": "text", "text": self._prompt_config.build_validation_user_prompt()},
                        ],
                    },
                ],
                "max_tokens": model_api_contract.max_tokens,
                "temperature": 0.1,
            }
        ).encode(DEFAULT_ENCODING)
        request = urllib.request.Request(
            model_api_contract.chat_completions_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=model_api_contract.request_timeout) as response:
            response_payload = json.loads(response.read())
        choice = response_payload["choices"][0]
        raw_content = choice["message"].get("content", "") or ""
        finish_reason = choice.get("finish_reason", "unknown")
        parsed_payload, repaired = self._response_extractor.parse_or_repair(
            raw_content,
            finish_reason=finish_reason,
        )
        return {
            "image": str(image_file_path),
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - start_time, 2),
            "finish_reason": finish_reason,
            "tokens_used": response_payload.get("usage", {}).get("total_tokens"),
            "parse_ok": parsed_payload is not None,
            "result": parsed_payload if parsed_payload is not None else {"raw_response": raw_content[:500]},
            "_repaired": repaired,
        }


class VisualTestRecordProcessor:
    """Processes one image file and persists the result."""

    def __init__(self, gateway: VisualApiTestGateway, json_repository: JsonRepository, print_lock: Lock) -> None:
        self._gateway = gateway
        self._json_repository = json_repository
        self._print_lock = print_lock

    def process(
        self,
        image_file_path: Path,
        index: int,
        total: int,
        worker_id: int,
        output_dir: Path,
        model_api_contract: TestModelApiContract,
    ) -> dict:
        with self._print_lock:
            print(f"  -> [W{worker_id}] starting: {image_file_path.name}", flush=True)
        try:
            result = self._gateway.classify_image(image_file_path, model_api_contract)
        except Exception as error:
            result = {
                "image": str(image_file_path),
                "timestamp": datetime.now().isoformat(),
                "elapsed_s": 0,
                "tokens_used": None,
                "parse_ok": False,
                "error": str(error),
                "result": None,
            }
        self._print_summary(result, index, total, worker_id)
        output_file_path = output_dir / f"{image_file_path.stem}_vlm.json"
        self._json_repository.save(output_file_path, result)
        return result

    def _print_summary(self, result: dict, index: int, total: int, worker_id: int) -> None:
        image_name = Path(result["image"]).name
        status = "OK" if result["parse_ok"] else "ERR"
        elapsed_seconds = result["elapsed_s"]
        tokens_used = result.get("tokens_used", "?")
        with self._print_lock:
            print("-" * 60)
            print(f"[{index}/{total}][W{worker_id}] {status} {image_name} elapsed={elapsed_seconds}s tokens={tokens_used}")
            if result.get("error"):
                print(f"   Error: {result['error']}")
                return
            if not result["parse_ok"]:
                print("   Invalid JSON response.")
                return
            payload = result["result"]
            metadata = payload.get("image_metadata", {})
            mindspace = payload.get("mindspace", {})
            present_components = [
                component_code
                for component_code, component_payload in mindspace.items()
                if isinstance(component_payload, dict) and component_payload.get("presente")
            ]
            print(f"   Page type: {metadata.get('tipo_pagina', '?')}")
            print(f"   Palette: {metadata.get('paleta_dominante', '?')}")
            print(f"   Present components: {', '.join(present_components) if present_components else 'none'}")


class ClassifierExportService:
    """Exports visual extraction results for a downstream text classifier."""

    def export(self, results: list[dict], output_dir: Path, file_name: str) -> Path:
        output_file_path = output_dir / file_name
        exported_count = 0
        with output_file_path.open("w", encoding=DEFAULT_ENCODING) as output_file:
            for result in results:
                if not result or not result.get("parse_ok") or not result.get("result"):
                    continue
                payload = result["result"]
                classifier_text = payload.get("texto_para_classificador", "")
                if not classifier_text:
                    continue
                image_id = Path(result["image"]).stem
                record = {
                    "id": image_id,
                    "source": "vlm_extraction",
                    "text": classifier_text,
                    "mindspace_hints": {
                        component_code: component_payload.get("presente", False)
                        for component_code, component_payload in payload.get("mindspace", {}).items()
                        if isinstance(component_payload, dict)
                    },
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported_count += 1
        print(f"Classifier export: {output_file_path} records={exported_count}")
        return output_file_path


class VisualApiTestOrchestrator:
    """Coordinates visual model API test execution."""

    def __init__(
        self,
        prompt_repository: VisualPromptConfigurationRepository,
        image_discovery_service: ImageDiscoveryService,
        image_encoder: ImageEncodingService,
        response_extractor: JsonResponseExtractionService,
        json_repository: JsonRepository,
        export_service: ClassifierExportService,
    ) -> None:
        self._prompt_repository = prompt_repository
        self._image_discovery_service = image_discovery_service
        self._image_encoder = image_encoder
        self._response_extractor = response_extractor
        self._json_repository = json_repository
        self._export_service = export_service

    def execute(self, execution_context: ImageTestExecutionContext) -> None:
        prompt_config = self._prompt_repository.load(execution_context.prompt_config_path)
        image_paths = self._image_discovery_service.discover(execution_context.input_contract)
        output_dir = execution_context.output_contract.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        workers = min(execution_context.workers, len(image_paths))
        print(f"Processing images: total={len(image_paths)} workers={workers}")
        print(f"Model API: {execution_context.model_api_contract.base_url}")
        print(f"Model: {execution_context.model_api_contract.model_name}")
        print(f"Output directory: {output_dir}")
        print_lock = Lock()
        processor = VisualTestRecordProcessor(
            gateway=VisualApiTestGateway(prompt_config, self._image_encoder, self._response_extractor),
            json_repository=self._json_repository,
            print_lock=print_lock,
        )
        start_time = time.time()
        results: list[dict] = [None] * len(image_paths)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    processor.process,
                    image_file_path,
                    index + 1,
                    len(image_paths),
                    (index % workers) + 1,
                    output_dir,
                    execution_context.model_api_contract,
                ): index
                for index, image_file_path in enumerate(image_paths)
            }
            for future in as_completed(future_to_index):
                result_index = future_to_index[future]
                try:
                    results[result_index] = future.result()
                except Exception as error:
                    image_file_path = image_paths[result_index]
                    results[result_index] = {
                        "image": str(image_file_path),
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_s": 0,
                        "tokens_used": None,
                        "parse_ok": False,
                        "error": f"Worker exception: {error}",
                        "result": None,
                    }
        batch_file_path = output_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._json_repository.save(batch_file_path, results)
        total_elapsed = time.time() - start_time
        valid_count = sum(1 for result in results if result and result["parse_ok"])
        error_count = len(results) - valid_count
        average_seconds = sum(result["elapsed_s"] for result in results if result) / len(results) if results else 0
        throughput = len(results) / total_elapsed * 60 if total_elapsed > 0 else 0
        print("Execution completed.")
        print(f"Valid JSON: {valid_count}/{len(results)}")
        print(f"Errors: {error_count}")
        print(f"Total elapsed: {total_elapsed:.1f}s")
        print(f"Average elapsed: {average_seconds:.1f}s/image")
        print(f"Throughput: {throughput:.1f} images/min")
        print(f"Batch file: {batch_file_path}")
        self._export_service.export(results, output_dir, execution_context.output_contract.classifier_export_file_name)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test visual model API extraction using configured visual MINDSPACE prompts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_VISUAL_PROMPT_CONFIG_PATH)
    parser.add_argument("--model-api-base-url", type=str, default=DEFAULT_MODEL_API_BASE_URL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--single", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, choices=[1, 2, 3])
    parser.add_argument("--classifier-export-file-name", type=str, default=DEFAULT_CLASSIFIER_EXPORT_FILE_NAME)
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> ImageTestExecutionContext:
    return ImageTestExecutionContext(
        prompt_config_path=args.config,
        model_api_contract=TestModelApiContract(
            base_url=args.model_api_base_url,
            model_name=args.model_name,
            max_tokens=DEFAULT_MAX_TOKENS,
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
        ),
        input_contract=ImageTestInputContract(image_dir=args.image_dir, single_image=args.single),
        output_contract=ImageTestOutputContract(output_dir=args.output_dir, classifier_export_file_name=args.classifier_export_file_name),
        workers=args.workers,
    )


def main() -> None:
    args = parse_arguments()
    try:
        execution_context = build_execution_context(args)
        orchestrator = VisualApiTestOrchestrator(
            prompt_repository=VisualPromptConfigurationRepository(),
            image_discovery_service=ImageDiscoveryService(),
            image_encoder=ImageEncodingService(),
            response_extractor=JsonResponseExtractionService(),
            json_repository=JsonRepository(),
            export_service=ClassifierExportService(),
        )
        orchestrator.execute(execution_context)
    except FileNotFoundError as error:
        print(str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
