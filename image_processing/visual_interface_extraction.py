#!/usr/bin/env python3
"""
Name: visual_interface_extraction_pipeline
Input: interface corpus JSONL file and visual prompt configuration YAML
Output: per-record JSON files and consolidated JSONL batch
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from visual_mindspace_pipeline_core import (
    ImageEncodingService,
    JsonLinesRepository,
    JsonRepository,
    JsonResponseExtractionService,
)
from visual_mindspace_prompt_loader import (
    DEFAULT_VISUAL_PROMPT_CONFIG_PATH,
    VisualPromptConfiguration,
    VisualPromptConfigurationRepository,
)


DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_MODEL_API_URL = "http://VLLM-SERVER:8001/v1/chat/completions"
DEFAULT_MODEL_NAME = "models/Qwen3-VL-8B-Instruct"
DEFAULT_MAX_TOKENS = 2500
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_CORPUS_FILE = DEFAULT_WORKSPACE_ROOT / "corpus" / "corpus_interface.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus" / "vlm_output"
DEFAULT_WORKERS = 2
DEFAULT_ENCODING = "utf-8"


@dataclass(frozen=True)
class ModelApiContract:
    url: str
    model_name: str
    max_tokens: int
    request_timeout: int


@dataclass(frozen=True)
class ExtractionInputContract:
    corpus_file_path: Path
    year_filter: int | None
    test_identifier: str | None


@dataclass(frozen=True)
class ExtractionOutputContract:
    output_dir: Path


@dataclass(frozen=True)
class ExtractionExecutionContext:
    prompt_config_path: Path
    model_api_contract: ModelApiContract
    input_contract: ExtractionInputContract
    output_contract: ExtractionOutputContract
    workers: int
    skip_existing: bool


class VisualModelGateway:
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

    def classify_record(self, record: dict, model_api_contract: ModelApiContract) -> dict:
        record_id = record["id"]
        file_path = record["filepath"]
        page_type = record["tipo_pagina"]
        year = record["ano"]
        image_payload = self._image_encoder.encode(Path(file_path))
        user_prompt = self._prompt_config.build_extraction_user_prompt(
            record_id=record_id,
            page_type=page_type,
            year=year,
        )
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
                                    "url": "data:image/png;base64," + image_payload.encoded_image
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                "max_tokens": model_api_contract.max_tokens,
                "temperature": 0.0,
            }
        ).encode(DEFAULT_ENCODING)
        request = urllib.request.Request(
            model_api_contract.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=model_api_contract.request_timeout) as response:
            response_payload = json.loads(response.read())
        choice = response_payload["choices"][0]
        finish_reason = choice.get("finish_reason", "unknown")
        raw_content = choice["message"].get("content", "") or ""
        result = {
            "id": record_id,
            "ano": year,
            "tipo_pagina": page_type,
            "janela": record.get("janela", ""),
            "filepath": file_path,
            "finish_reason": finish_reason,
            "_repaired": False,
        }
        parsed_payload, repaired = self._response_extractor.parse_or_repair(
            raw_content,
            finish_reason=finish_reason,
        )
        if parsed_payload:
            result.update(parsed_payload)
            result["_repaired"] = repaired
        else:
            result["_parse_error"] = True
            result["_raw"] = raw_content[:500]
        return result


class InterfaceCorpusRepository:
    """Loads interface corpus records."""

    def __init__(self, jsonl_repository: JsonLinesRepository) -> None:
        self._jsonl_repository = jsonl_repository

    def load_records(self, input_contract: ExtractionInputContract) -> list[dict]:
        records = self._jsonl_repository.load_records(input_contract.corpus_file_path)
        if input_contract.test_identifier:
            return [record for record in records if record["id"] == input_contract.test_identifier]
        if input_contract.year_filter is not None:
            return [record for record in records if record.get("ano") == input_contract.year_filter]
        return records


class ExtractionRecordProcessor:
    """Processes and persists one visual extraction record."""

    def __init__(self, model_gateway: VisualModelGateway, json_repository: JsonRepository) -> None:
        self._model_gateway = model_gateway
        self._json_repository = json_repository

    def process(
        self,
        record: dict,
        output_dir: Path,
        skip_existing: bool,
        model_api_contract: ModelApiContract,
    ) -> dict:
        output_file_path = output_dir / f"{record['id']}.json"
        if skip_existing and output_file_path.exists():
            try:
                return self._json_repository.load(output_file_path)
            except Exception:
                pass
        start_time = time.time()
        try:
            result = self._model_gateway.classify_record(record, model_api_contract)
        except Exception as error:
            result = {
                "id": record["id"],
                "ano": record.get("ano"),
                "tipo_pagina": record.get("tipo_pagina"),
                "janela": record.get("janela"),
                "filepath": record.get("filepath"),
                "_error": str(error),
            }
        result["_elapsed_s"] = round(time.time() - start_time, 1)
        self._json_repository.save(output_file_path, result)
        return result


class VisualExtractionOrchestrator:
    """Coordinates visual extraction execution."""

    def __init__(
        self,
        prompt_repository: VisualPromptConfigurationRepository,
        corpus_repository: InterfaceCorpusRepository,
        json_repository: JsonRepository,
        jsonl_repository: JsonLinesRepository,
        image_encoder: ImageEncodingService,
        response_extractor: JsonResponseExtractionService,
    ) -> None:
        self._prompt_repository = prompt_repository
        self._corpus_repository = corpus_repository
        self._json_repository = json_repository
        self._jsonl_repository = jsonl_repository
        self._image_encoder = image_encoder
        self._response_extractor = response_extractor

    def execute(self, execution_context: ExtractionExecutionContext) -> None:
        prompt_config = self._prompt_repository.load(execution_context.prompt_config_path)
        output_dir = execution_context.output_contract.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        records = self._corpus_repository.load_records(execution_context.input_contract)
        print(f"Records loaded: {len(records)}")
        if execution_context.input_contract.test_identifier and not records:
            print("Test identifier was not found in the corpus.")
            return
        processor = ExtractionRecordProcessor(
            model_gateway=VisualModelGateway(
                prompt_config=prompt_config,
                image_encoder=self._image_encoder,
                response_extractor=self._response_extractor,
            ),
            json_repository=self._json_repository,
        )
        total = len(records)
        errors = 0
        results: list[dict] = []
        print(
            "Starting visual extraction: "
            f"workers={execution_context.workers}, "
            f"skip_existing={execution_context.skip_existing}"
        )
        if execution_context.workers == 1 or execution_context.input_contract.test_identifier:
            for index, record in enumerate(records, 1):
                result = processor.process(
                    record,
                    output_dir,
                    execution_context.skip_existing,
                    execution_context.model_api_contract,
                )
                results.append(result)
                if "_error" in result or "_parse_error" in result:
                    errors += 1
                status = "ERR" if ("_error" in result or "_parse_error" in result) else "OK"
                print(f"  [{status}] {record['id']} ({index}/{total}) elapsed={result.get('_elapsed_s', '?')}s")
        else:
            with ThreadPoolExecutor(max_workers=execution_context.workers) as executor:
                futures = {
                    executor.submit(
                        processor.process,
                        record,
                        output_dir,
                        execution_context.skip_existing,
                        execution_context.model_api_contract,
                    ): record
                    for record in records
                }
                for index, future in enumerate(as_completed(futures), 1):
                    record = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {"id": record["id"], "_error": str(error)}
                    results.append(result)
                    if "_error" in result or "_parse_error" in result:
                        errors += 1
                    if index % 20 == 0 or index == total:
                        print(f"  Progress: {index}/{total} errors={errors}")
        batch_file_path = output_dir / f"vlm_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._jsonl_repository.save_records(batch_file_path, sorted(results, key=lambda item: item.get("id", "")))
        print("Execution completed.")
        print(f"Total processed: {total}")
        print(f"Errors: {errors}")
        print(f"Batch file: {batch_file_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured visual MINDSPACE records from image corpus records."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_VISUAL_PROMPT_CONFIG_PATH)
    parser.add_argument("--model-api-url", type=str, default=DEFAULT_MODEL_API_URL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--test", type=str, default=None)
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> ExtractionExecutionContext:
    return ExtractionExecutionContext(
        prompt_config_path=args.config,
        model_api_contract=ModelApiContract(
            url=args.model_api_url,
            model_name=args.model_name,
            max_tokens=DEFAULT_MAX_TOKENS,
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
        ),
        input_contract=ExtractionInputContract(
            corpus_file_path=args.corpus,
            year_filter=args.year,
            test_identifier=args.test,
        ),
        output_contract=ExtractionOutputContract(output_dir=args.out_dir),
        workers=args.workers,
        skip_existing=args.skip_existing,
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = VisualExtractionOrchestrator(
        prompt_repository=VisualPromptConfigurationRepository(),
        corpus_repository=InterfaceCorpusRepository(JsonLinesRepository()),
        json_repository=JsonRepository(),
        jsonl_repository=JsonLinesRepository(),
        image_encoder=ImageEncodingService(),
        response_extractor=JsonResponseExtractionService(),
    )
    orchestrator.execute(execution_context)


if __name__ == "__main__":
    main()
