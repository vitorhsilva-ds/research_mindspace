#!/usr/bin/env python3
"""
Name: complaint_corpus_classification_pipeline
Input: complaint corpus JSONL file and prompt configuration YAML
Output: per-record classification JSON files and consolidated JSONL batch
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from prompt_config_loader import PromptConfigRepository, PromptTemplateService

DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_PROMPT_CONFIG_PATH = DEFAULT_WORKSPACE_ROOT / "configs" / "mindspace_prompt_config.yaml"
DEFAULT_MODEL_API_URL = "http://VLLM-SERVER:8000/v1/chat/completions"
DEFAULT_MODEL_NAME = "model_artifacts/model_merged_mxfp4"
DEFAULT_CORPUS_FILE = DEFAULT_WORKSPACE_ROOT / "corpus" / "corpus_complaints.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus" / "complaint_output"
DEFAULT_MAX_TOKENS = 20000
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_ENCODING = "utf-8"


@dataclass(frozen=True)
class ComplaintCorpusInputContract:
    corpus_file_path: Path


@dataclass(frozen=True)
class ComplaintCorpusOutputContract:
    output_dir: Path


@dataclass(frozen=True)
class ComplaintModelApiContract:
    url: str
    model_name: str
    max_tokens: int
    request_timeout: int


@dataclass(frozen=True)
class ComplaintCorpusExecutionContext:
    prompt_config_path: Path
    input_contract: ComplaintCorpusInputContract
    output_contract: ComplaintCorpusOutputContract
    model_api_contract: ComplaintModelApiContract
    skip_existing: bool
    year_filter: int | None
    test_identifier: str | None


class JsonLinesRepository:
    """Loads and saves JSON Lines records."""

    def load_records(self, file_path: Path) -> list[dict]:
        records: list[dict] = []

        with file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            for line in input_file:
                if line.strip():
                    records.append(json.loads(line))

        return records

    def save_records(self, file_path: Path, records: list[dict]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding=DEFAULT_ENCODING) as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


class JsonRepository:
    """Loads and saves JSON files."""

    def load(self, file_path: Path) -> dict:
        return json.loads(file_path.read_text(encoding=DEFAULT_ENCODING))

    def save(self, file_path: Path, payload: dict) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=DEFAULT_ENCODING,
        )


class ComplaintPromptBuilder:
    """Builds user prompts for complaint corpus classification."""

    def build_user_prompt(self, record: dict) -> str:
        title = record.get("title", "")
        text = record.get("text", "")
        year = record.get("ano", "")

        return (
            "Classify the following consumer narrative using the MINDSPACE framework.\n\n"
            f"YEAR: {year}\n\n"
            f"TITLE: {title}\n\n"
            f"TEXT: {text}\n\n"
            "Return only valid JSON using the configured schema."
        )


class ComplaintModelGateway:
    """Calls the configured model endpoint for complaint classification."""

    def __init__(
        self,
        prompt_service: PromptTemplateService,
        prompt_builder: ComplaintPromptBuilder,
    ) -> None:
        self._prompt_service = prompt_service
        self._prompt_builder = prompt_builder

    def classify(self, record: dict, model_api_contract: ComplaintModelApiContract) -> dict:
        payload = json.dumps(
            {
                "model": model_api_contract.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": self._prompt_service.build_validation_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._prompt_builder.build_user_prompt(record),
                    },
                ],
                "max_tokens": model_api_contract.max_tokens,
                "temperature": 0.0,
            }
        ).encode()

        request = urllib.request.Request(
            model_api_contract.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=model_api_contract.request_timeout,
        ) as response:
            return json.loads(response.read())


class ClassificationResponseExtractionService:
    """Extracts classification JSON from model API responses."""

    def extract_json_from_response(self, response_data: dict) -> dict | None:
        choice = response_data["choices"][0]
        message = choice.get("message", {})
        reasoning = (
            message.get("reasoning_content", "")
            or message.get("reasoning", "")
            or ""
        )
        content = message.get("content", "") or ""

        for source in (content, reasoning):
            parsed = self._try_parse_json(source)
            if parsed and "vector" in parsed:
                return parsed

        return self._try_parse_json_aggressive(f"{reasoning}\n{content}")

    def _try_parse_json(self, text: str) -> dict | None:
        if not text:
            return None

        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)

        try:
            return json.loads(text)
        except Exception:
            pass

        start_index = text.find("{")
        if start_index >= 0:
            try:
                return json.loads(text[start_index:])
            except Exception:
                pass

        return None

    def _try_parse_json_aggressive(self, text: str) -> dict | None:
        candidates = list(re.finditer(r"\{", text))

        for match in reversed(candidates):
            parsed = self._try_parse_json(text[match.start() :])
            if parsed and "vector" in parsed:
                return parsed

        return None


class ComplaintClassificationNormalizationService:
    """Normalizes classification result structure and value types."""

    def __init__(self, prompt_service: PromptTemplateService) -> None:
        self._prompt_service = prompt_service

    def normalize(self, parsed_result: dict) -> dict:
        vector = parsed_result.get("vector", {})
        normalized_vector: dict[str, bool] = {}

        for component_code in self._prompt_service.component_order:
            value = vector.get(component_code, False)

            if isinstance(value, str):
                value = value.lower() in ("true", "1", "sim", "yes")

            normalized_vector[component_code] = bool(value)

        primary = str(parsed_result.get("primary", "indeterminado")).strip()
        valid_primaries = set(self._prompt_service.component_order) | {"indeterminado"}

        if primary not in valid_primaries:
            for component_code in self._prompt_service.component_order:
                if component_code in primary.upper():
                    primary = component_code
                    break
            else:
                active_components = [
                    component_code
                    for component_code in self._prompt_service.component_order
                    if normalized_vector[component_code]
                ]
                primary = active_components[0] if active_components else "indeterminado"

        return {
            "primary": primary,
            "nudge_or_sludge": str(parsed_result.get("nudge_or_sludge", "indeterminado")),
            "confidence": str(parsed_result.get("confidence", "baixa")),
            "vector": normalized_vector,
            "justificativa": str(
                parsed_result.get(
                    "justificativa",
                    parsed_result.get("justification", ""),
                )
            ),
        }


class ReviewFlagPolicy:
    """Computes review flags for complaint classifications."""

    def get_review_flags(self, result: dict) -> list[str]:
        flags: list[str] = []
        vector = result.get("vector", {})

        if vector.get("C") and not vector.get("I"):
            flags.append("C=1 without I=1: review required")

        if vector.get("P") and not vector.get("I") and not vector.get("S"):
            flags.append("P=1 isolated: review required")

        if vector.get("M"):
            flags.append("M=1: verify source-authority effect")

        return flags


class ComplaintRecordClassificationService:
    """Classifies a single complaint corpus record."""

    def __init__(
        self,
        prompt_service: PromptTemplateService,
        model_gateway: ComplaintModelGateway,
        response_extractor: ClassificationResponseExtractionService,
        normalization_service: ComplaintClassificationNormalizationService,
        review_flag_policy: ReviewFlagPolicy,
        json_repository: JsonRepository,
    ) -> None:
        self._prompt_service = prompt_service
        self._model_gateway = model_gateway
        self._response_extractor = response_extractor
        self._normalization_service = normalization_service
        self._review_flag_policy = review_flag_policy
        self._json_repository = json_repository

    def process_record(
        self,
        record: dict,
        output_dir: Path,
        skip_existing: bool,
        model_api_contract: ComplaintModelApiContract,
    ) -> dict:
        record_id = record["id"]
        output_file_path = output_dir / f"{record_id}.json"

        if skip_existing and output_file_path.exists():
            try:
                return self._json_repository.load(output_file_path)
            except Exception:
                pass

        if not record.get("text"):
            result = self._build_no_text_result(record)
            self._json_repository.save(output_file_path, result)
            return result

        start_time = time.time()

        try:
            api_response = self._model_gateway.classify(record, model_api_contract)
            parsed_result = self._response_extractor.extract_json_from_response(api_response)
            finish_reason = api_response["choices"][0].get("finish_reason", "unknown")

            if parsed_result:
                normalized_result = self._normalization_service.normalize(parsed_result)
                flags = self._review_flag_policy.get_review_flags(normalized_result)
                result = {
                    "id": record_id,
                    "ano": record.get("ano"),
                    "janela": record.get("janela"),
                    "complaint_date": record.get("complaint_date"),
                    "filename": record.get("filename"),
                    "finish_reason": finish_reason,
                    **normalized_result,
                    "needs_review": 1 if flags else 0,
                    "review_reasons": " | ".join(flags),
                    "_elapsed_s": round(time.time() - start_time, 1),
                }
            else:
                result = self._build_parse_error_result(record, finish_reason, start_time)

        except Exception as error:
            result = self._build_api_error_result(record, error, start_time)

        self._json_repository.save(output_file_path, result)
        return result

    def _build_no_text_result(self, record: dict) -> dict:
        return {
            "id": record["id"],
            "ano": record.get("ano"),
            "janela": record.get("janela"),
            "complaint_date": record.get("complaint_date"),
            "filename": record.get("filename"),
            "_no_text": True,
            "primary": "indeterminado",
            "nudge_or_sludge": "indeterminado",
            "confidence": "baixa",
            "vector": {
                component_code: False
                for component_code in self._prompt_service.component_order
            },
            "justificativa": "empty input text",
            "needs_review": 0,
            "review_reasons": "",
        }

    def _build_parse_error_result(
        self,
        record: dict,
        finish_reason: str,
        start_time: float,
    ) -> dict:
        return {
            "id": record["id"],
            "ano": record.get("ano"),
            "janela": record.get("janela"),
            "complaint_date": record.get("complaint_date"),
            "filename": record.get("filename"),
            "finish_reason": finish_reason,
            "_parse_error": True,
            "primary": "indeterminado",
            "nudge_or_sludge": "indeterminado",
            "confidence": "baixa",
            "vector": {
                component_code: False
                for component_code in self._prompt_service.component_order
            },
            "justificativa": "",
            "needs_review": 1,
            "review_reasons": "parse_error: manual review required",
            "_elapsed_s": round(time.time() - start_time, 1),
        }

    def _build_api_error_result(
        self,
        record: dict,
        error: Exception,
        start_time: float,
    ) -> dict:
        return {
            "id": record["id"],
            "ano": record.get("ano"),
            "janela": record.get("janela"),
            "_error": str(error),
            "primary": "indeterminado",
            "nudge_or_sludge": "indeterminado",
            "confidence": "baixa",
            "vector": {
                component_code: False
                for component_code in self._prompt_service.component_order
            },
            "justificativa": "",
            "needs_review": 1,
            "review_reasons": f"api_error: {str(error)[:100]}",
            "_elapsed_s": round(time.time() - start_time, 1),
        }


class ComplaintCorpusClassificationOrchestrator:
    """Coordinates complaint corpus classification."""

    def __init__(
        self,
        jsonl_repository: JsonLinesRepository,
        json_repository: JsonRepository,
    ) -> None:
        self._jsonl_repository = jsonl_repository
        self._json_repository = json_repository

    def execute(self, execution_context: ComplaintCorpusExecutionContext) -> None:
        prompt_config = PromptConfigRepository().load(execution_context.prompt_config_path)
        prompt_service = PromptTemplateService(prompt_config)
        classification_service = ComplaintRecordClassificationService(
            prompt_service=prompt_service,
            model_gateway=ComplaintModelGateway(
                prompt_service=prompt_service,
                prompt_builder=ComplaintPromptBuilder(),
            ),
            response_extractor=ClassificationResponseExtractionService(),
            normalization_service=ComplaintClassificationNormalizationService(prompt_service),
            review_flag_policy=ReviewFlagPolicy(),
            json_repository=self._json_repository,
        )

        output_dir = execution_context.output_contract.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        records = self._jsonl_repository.load_records(
            execution_context.input_contract.corpus_file_path,
        )
        records = self._apply_filters(
            records=records,
            year_filter=execution_context.year_filter,
            test_identifier=execution_context.test_identifier,
        )

        total = len(records)
        errors = 0
        print("Starting complaint corpus classification.")
        print(f"Records to process: {total}")

        for index, record in enumerate(records, 1):
            result = classification_service.process_record(
                record=record,
                output_dir=output_dir,
                skip_existing=execution_context.skip_existing,
                model_api_contract=execution_context.model_api_contract,
            )
            has_error = result.get("_parse_error") or result.get("_error")
            if has_error:
                errors += 1

            print(
                f"  [{index:04d}/{total:04d}] {record['id']} "
                f"primary={result.get('primary', '?')} "
                f"elapsed={result.get('_elapsed_s', '?')}s"
            )

        batch_file_path = self._write_batch_file(output_dir)
        print("Execution completed.")
        print(f"Total records: {total}")
        print(f"Errors: {errors}")
        print(f"Batch file: {batch_file_path}")

    def _apply_filters(
        self,
        records: list[dict],
        year_filter: int | None,
        test_identifier: str | None,
    ) -> list[dict]:
        if test_identifier:
            return [record for record in records if record["id"] == test_identifier]

        if year_filter:
            return [record for record in records if record.get("ano") == year_filter]

        return records

    def _write_batch_file(self, output_dir: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file_path = output_dir / f"complaint_classified_{timestamp}.jsonl"
        all_results: list[dict] = []

        for json_file_path in sorted(output_dir.glob("*.json")):
            try:
                all_results.append(self._json_repository.load(json_file_path))
            except Exception:
                pass

        self._jsonl_repository.save_records(
            batch_file_path,
            sorted(all_results, key=lambda item: item.get("id", "")),
        )
        return batch_file_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify complaint corpus records.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROMPT_CONFIG_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-api-url", type=str, default=DEFAULT_MODEL_API_URL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--test", type=str, default=None)
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> ComplaintCorpusExecutionContext:
    return ComplaintCorpusExecutionContext(
        prompt_config_path=args.config,
        input_contract=ComplaintCorpusInputContract(corpus_file_path=args.corpus),
        output_contract=ComplaintCorpusOutputContract(output_dir=args.out_dir),
        model_api_contract=ComplaintModelApiContract(
            url=args.model_api_url,
            model_name=args.model_name,
            max_tokens=DEFAULT_MAX_TOKENS,
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
        ),
        skip_existing=args.skip_existing,
        year_filter=args.year,
        test_identifier=args.test,
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = ComplaintCorpusClassificationOrchestrator(
        jsonl_repository=JsonLinesRepository(),
        json_repository=JsonRepository(),
    )
    orchestrator.execute(execution_context)


if __name__ == "__main__":
    main()
