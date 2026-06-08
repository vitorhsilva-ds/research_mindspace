#!/usr/bin/env python3
"""
Name: synthetic_text_classification_pipeline
Input: synthetic text JSONL files and prompt configuration YAML
Output: classified JSONL files and consolidated training dataset JSONL
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
DEFAULT_MODEL_API_KEY = "EMPTY"
DEFAULT_MODEL_NAME = "models/gpt-oss-20b"
DEFAULT_INPUT_DIR = Path("./nudges_raw")
DEFAULT_OUTPUT_DIR = Path("./nudges_classified")
DEFAULT_FINAL_OUTPUT_FILE = Path("./dataset_finetune_full.jsonl")
DEFAULT_TEMPERATURE = 0.10
DEFAULT_MAX_TOKENS = 3000
DEFAULT_CONCURRENCY = 16
DEFAULT_RETRY_LIMIT = 3
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_HTTP_TIMEOUT = 120.0
DEFAULT_TASK_TIMEOUT = 110.0
DEFAULT_ENCODING = "utf-8"
DEFAULT_DOMAINS = ("interface", "complaint")


@dataclass(frozen=True)
class ClassificationModelApiContract:
    url: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    retry_limit: int
    retry_delay: float
    task_timeout: float
    http_timeout: float


@dataclass(frozen=True)
class ClassificationInputContract:
    input_dir: Path
    components: tuple[str, ...]
    domains: tuple[str, ...]


@dataclass(frozen=True)
class ClassificationOutputContract:
    output_dir: Path
    final_output_file: Path


@dataclass(frozen=True)
class ClassificationExecutionContext:
    prompt_config_path: Path
    model_api_contract: ClassificationModelApiContract
    input_contract: ClassificationInputContract
    output_contract: ClassificationOutputContract
    concurrency: int
    resume: bool


class SyntheticRecordRepository:
    """Loads raw synthetic records and persists classified records."""

    def load_records(self, input_contract: ClassificationInputContract) -> list[dict]:
        records: list[dict] = []

        for component in input_contract.components:
            for domain in input_contract.domains:
                input_file_path = input_contract.input_dir / f"{component.lower()}_{domain}.jsonl"

                if not input_file_path.exists():
                    print(f"Input file not found: {input_file_path}")
                    continue

                with input_file_path.open(encoding=DEFAULT_ENCODING) as input_file:
                    for line in input_file:
                        if line.strip():
                            records.append(json.loads(line))

        return records

    def remove_existing_outputs(self, output_dir: Path) -> None:
        if output_dir.exists():
            for output_file_path in output_dir.glob("*.jsonl"):
                output_file_path.unlink()

    def load_classified_identifiers(self, output_dir: Path) -> set[str]:
        classified_identifiers: set[str] = set()

        if not output_dir.exists():
            return classified_identifiers

        for output_file_path in output_dir.glob("*.jsonl"):
            with output_file_path.open(encoding=DEFAULT_ENCODING) as input_file:
                for line in input_file:
                    if line.strip():
                        classified_identifiers.add(json.loads(line)["id"])

        return classified_identifiers

    def append_classification(self, output_dir: Path, result: dict) -> None:
        output_file_path = output_dir / f"{result['component_gold'].lower()}_{result['domain']}.jsonl"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        with output_file_path.open("a", encoding=DEFAULT_ENCODING) as output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    def consolidate_outputs(self, output_dir: Path, final_output_file: Path) -> list[dict]:
        all_records: list[dict] = []

        for output_file_path in sorted(output_dir.glob("*.jsonl")):
            with output_file_path.open(encoding=DEFAULT_ENCODING) as input_file:
                for line in input_file:
                    if line.strip():
                        all_records.append(json.loads(line))

        random.shuffle(all_records)
        final_output_file.parent.mkdir(parents=True, exist_ok=True)

        with final_output_file.open("w", encoding=DEFAULT_ENCODING) as output_file:
            for record in all_records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return all_records


class JsonResponseExtractionService:
    """Extracts JSON payloads from model responses."""

    def extract_json(self, text: str) -> dict:
        cleaned_text = (text or "").strip()

        if cleaned_text.startswith("```"):
            parts = cleaned_text.split("```")
            cleaned_text = parts[1] if len(parts) > 1 else cleaned_text
            if cleaned_text.startswith("json"):
                cleaned_text = cleaned_text[4:]
            cleaned_text = cleaned_text.strip()

        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].strip()

        return json.loads(cleaned_text)


class ClassificationValidationService:
    """Validates classification payload structure."""

    def validate(self, classification: dict, component_order: tuple[str, ...]) -> None:
        vector = classification.get("vector", {})

        for component_code in component_order:
            if component_code not in vector:
                raise ValueError(f"Missing component in vector: {component_code}")

            if vector[component_code] not in (0, 1):
                raise ValueError(
                    f"Invalid vector value for {component_code}: {vector[component_code]}"
                )

        for field_name in ("primary", "nudge_or_sludge", "confidence", "justification"):
            if field_name not in classification:
                raise ValueError(f"Missing required field: {field_name}")


class ModelClassificationGateway:
    """Requests classifications from the configured model endpoint."""

    def __init__(
        self,
        prompt_service: PromptTemplateService,
        response_extractor: JsonResponseExtractionService,
        validation_service: ClassificationValidationService,
    ) -> None:
        self._prompt_service = prompt_service
        self._response_extractor = response_extractor
        self._validation_service = validation_service

    async def classify_record(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        model_api_contract: ClassificationModelApiContract,
        record: dict,
    ) -> dict | None:
        payload = {
            "model": model_api_contract.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self._prompt_service.build_classification_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._prompt_service.build_classification_user_prompt(
                        domain=record["domain"],
                        text=record["text"],
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
                            headers={"Authorization": f"Bearer {model_api_contract.api_key}"},
                        ),
                        timeout=model_api_contract.task_timeout,
                    )
                    response.raise_for_status()
                    message = response.json()["choices"][0]["message"]
                    final_content = (message.get("content") or "").strip()
                    chain_of_thought = (
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    ).strip()

                    if not final_content:
                        raise ValueError("Model response content is empty.")

                    classification = self._response_extractor.extract_json(final_content)
                    self._validation_service.validate(
                        classification=classification,
                        component_order=self._prompt_service.component_order,
                    )

                    return self._build_finetune_record(
                        source_record=record,
                        classification=classification,
                        chain_of_thought=chain_of_thought,
                    )

                except (json.JSONDecodeError, ValueError):
                    if attempt < model_api_contract.retry_limit - 1:
                        await asyncio.sleep(model_api_contract.retry_delay + random.uniform(0, 2))
                    else:
                        return None
                except asyncio.TimeoutError:
                    if attempt < model_api_contract.retry_limit - 1:
                        await asyncio.sleep(model_api_contract.retry_delay)
                    else:
                        return None
                except Exception:
                    if attempt < model_api_contract.retry_limit - 1:
                        await asyncio.sleep(model_api_contract.retry_delay + random.uniform(0, 2))
                    else:
                        return None

        return None

    def _build_finetune_record(
        self,
        source_record: dict,
        classification: dict,
        chain_of_thought: str,
    ) -> dict:
        active_components = [
            self._prompt_service.component_names[component_code]
            for component_code in self._prompt_service.component_order
            if classification["vector"][component_code] == 1
        ]
        gold_in_vector = source_record["component"] in active_components

        return {
            "id": source_record["id"],
            "component_gold": source_record["component"],
            "domain": source_record["domain"],
            "instruction": self._prompt_service.build_classification_system_prompt(),
            "input": source_record["text"],
            "chain_of_thought": chain_of_thought,
            "output": classification,
            "metadata": {
                "active_components": active_components,
                "n_active": len(active_components),
                "gold_in_vector": gold_in_vector,
                "primary": classification["primary"],
                "nudge_or_sludge": classification["nudge_or_sludge"],
                "confidence": classification["confidence"],
                "cot_length": len(chain_of_thought),
            },
        }


class ClassificationProgressReporter:
    """Reports operational classification progress."""

    async def report(self, counters: dict[str, int], total: int, interval: float = 3.0) -> None:
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            percentage = 100 * counters["done"] / total if total > 0 else 100
            rate = counters["done"] / elapsed_time if elapsed_time > 0 else 0
            estimated_seconds = (total - counters["done"]) / rate if rate > 0 else 0
            hit_rate = (
                100 * counters["gold_hits"] / counters["success"]
                if counters["success"] > 0
                else 0
            )

            print(
                f"\r  [{percentage:5.1f}%] {counters['done']:>4}/{total}  "
                f"success={counters['success']} failed={counters['failed']} "
                f"gold_in_vector={hit_rate:.0f}% "
                f"{rate:.1f} req/s eta={estimated_seconds:.0f}s   ",
                end="",
                flush=True,
            )

            if counters["done"] >= total:
                break

            await asyncio.sleep(interval)

        print()


class SyntheticClassificationOrchestrator:
    """Coordinates the synthetic text classification lifecycle."""

    def __init__(
        self,
        record_repository: SyntheticRecordRepository,
        response_extractor: JsonResponseExtractionService,
        validation_service: ClassificationValidationService,
        progress_reporter: ClassificationProgressReporter,
    ) -> None:
        self._record_repository = record_repository
        self._response_extractor = response_extractor
        self._validation_service = validation_service
        self._progress_reporter = progress_reporter

    async def execute(self, execution_context: ClassificationExecutionContext) -> None:
        prompt_config = PromptConfigRepository().load(execution_context.prompt_config_path)
        prompt_service = PromptTemplateService(prompt_config)
        classification_gateway = ModelClassificationGateway(
            prompt_service=prompt_service,
            response_extractor=self._response_extractor,
            validation_service=self._validation_service,
        )

        output_contract = execution_context.output_contract
        output_contract.output_dir.mkdir(parents=True, exist_ok=True)

        if not execution_context.resume:
            self._record_repository.remove_existing_outputs(output_contract.output_dir)
            print("Existing classification files removed.")

        source_records = self._record_repository.load_records(execution_context.input_contract)
        print(f"Source records loaded: {len(source_records)}")

        classified_identifiers = (
            self._record_repository.load_classified_identifiers(output_contract.output_dir)
            if execution_context.resume
            else set()
        )
        pending_records = [
            record for record in source_records if record["id"] not in classified_identifiers
        ]

        if not pending_records:
            self._consolidate(output_contract)
            return

        await self._process_pending_records(
            pending_records=pending_records,
            classification_gateway=classification_gateway,
            execution_context=execution_context,
        )
        self._consolidate(output_contract)

    async def _process_pending_records(
        self,
        pending_records: list[dict],
        classification_gateway: ModelClassificationGateway,
        execution_context: ClassificationExecutionContext,
    ) -> None:
        total = len(pending_records)
        semaphore = asyncio.Semaphore(execution_context.concurrency)
        file_lock = asyncio.Lock()
        counters = {"done": 0, "success": 0, "failed": 0, "gold_hits": 0}
        batch_size = execution_context.concurrency * 2
        batches = [
            pending_records[index : index + batch_size]
            for index in range(0, len(pending_records), batch_size)
        ]
        limits = httpx.Limits(
            max_connections=execution_context.concurrency + 4,
            max_keepalive_connections=execution_context.concurrency,
        )

        async with httpx.AsyncClient(
            timeout=execution_context.model_api_contract.http_timeout,
            limits=limits,
        ) as client:
            batch_tasks = [
                self._classify_batch(
                    client=client,
                    semaphore=semaphore,
                    file_lock=file_lock,
                    counters=counters,
                    classification_gateway=classification_gateway,
                    model_api_contract=execution_context.model_api_contract,
                    records=batch,
                    output_dir=execution_context.output_contract.output_dir,
                )
                for batch in batches
            ]
            await asyncio.gather(self._progress_reporter.report(counters, total), *batch_tasks)

        print("Classification completed.")
        print(
            "Records: "
            f"success={counters['success']}, "
            f"failed={counters['failed']}, "
            f"gold_hits={counters['gold_hits']}"
        )

    async def _classify_batch(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        file_lock: asyncio.Lock,
        counters: dict[str, int],
        classification_gateway: ModelClassificationGateway,
        model_api_contract: ClassificationModelApiContract,
        records: list[dict],
        output_dir: Path,
    ) -> None:
        results = await asyncio.gather(
            *[
                classification_gateway.classify_record(
                    client=client,
                    semaphore=semaphore,
                    model_api_contract=model_api_contract,
                    record=record,
                )
                for record in records
            ]
        )

        async with file_lock:
            for result in results:
                if result is not None:
                    self._record_repository.append_classification(output_dir, result)
                    counters["success"] += 1
                    if result["metadata"]["gold_in_vector"]:
                        counters["gold_hits"] += 1
                else:
                    counters["failed"] += 1

                counters["done"] += 1

    def _consolidate(self, output_contract: ClassificationOutputContract) -> None:
        records = self._record_repository.consolidate_outputs(
            output_dir=output_contract.output_dir,
            final_output_file=output_contract.final_output_file,
        )
        print(f"Consolidated dataset: {output_contract.final_output_file}")
        print(f"Total records: {len(records)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify synthetic texts with the MINDSPACE framework.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROMPT_CONFIG_PATH)
    parser.add_argument("--model-api-url", type=str, default=DEFAULT_MODEL_API_URL)
    parser.add_argument("--model-api-key", type=str, default=DEFAULT_MODEL_API_KEY)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL_OUTPUT_FILE)
    parser.add_argument("--components", nargs="+", default=None)
    parser.add_argument("--domains", nargs="+", choices=list(DEFAULT_DOMAINS), default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> ClassificationExecutionContext:
    prompt_config = PromptConfigRepository().load(args.config)
    prompt_service = PromptTemplateService(prompt_config)

    return ClassificationExecutionContext(
        prompt_config_path=args.config,
        model_api_contract=ClassificationModelApiContract(
            url=args.model_api_url,
            api_key=args.model_api_key,
            model_name=args.model_name,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            retry_limit=DEFAULT_RETRY_LIMIT,
            retry_delay=DEFAULT_RETRY_DELAY,
            task_timeout=DEFAULT_TASK_TIMEOUT,
            http_timeout=DEFAULT_HTTP_TIMEOUT,
        ),
        input_contract=ClassificationInputContract(
            input_dir=args.input_dir,
            components=tuple(args.components or prompt_service.component_full_names),
            domains=tuple(args.domains or DEFAULT_DOMAINS),
        ),
        output_contract=ClassificationOutputContract(
            output_dir=args.output_dir,
            final_output_file=args.final_output,
        ),
        concurrency=args.concurrency,
        resume=not args.no_resume,
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = SyntheticClassificationOrchestrator(
        record_repository=SyntheticRecordRepository(),
        response_extractor=JsonResponseExtractionService(),
        validation_service=ClassificationValidationService(),
        progress_reporter=ClassificationProgressReporter(),
    )

    if args.consolidate_only:
        orchestrator._consolidate(execution_context.output_contract)
        return

    asyncio.run(orchestrator.execute(execution_context))


if __name__ == "__main__":
    main()
