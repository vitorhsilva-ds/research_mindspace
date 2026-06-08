#!/usr/bin/env python3
"""
Name: synthetic_text_generation_pipeline
Input: prompt configuration YAML
Output: synthetic text JSONL files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from prompt_config_loader import PromptConfigRepository, PromptTemplateService

DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_PROMPT_CONFIG_PATH = DEFAULT_WORKSPACE_ROOT / "configs" / "mindspace_prompt_config.yaml"
DEFAULT_MODEL_API_URL = "http://VLLM-SERVER:8000/v1/chat/completions"
DEFAULT_MODEL_API_KEY = "EMPTY"
DEFAULT_MODEL_NAME = "models/gpt-oss-20b"
DEFAULT_OUTPUT_DIR = Path("./nudges_raw")
DEFAULT_EXAMPLES_PER_COMPONENT = 100
DEFAULT_TEMPERATURE = 0.90
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CONCURRENCY = 16
DEFAULT_RETRY_LIMIT = 3
DEFAULT_RETRY_DELAY = 4.0
DEFAULT_HTTP_TIMEOUT = 90.0
DEFAULT_TASK_TIMEOUT = 85.0
DEFAULT_ENCODING = "utf-8"
DEFAULT_DOMAINS = ("interface", "complaint")


@dataclass(frozen=True)
class ModelApiContract:
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
class GenerationContract:
    output_dir: Path
    components: tuple[str, ...]
    domains: tuple[str, ...]
    examples_per_component: int
    concurrency: int
    resume: bool


@dataclass(frozen=True)
class GenerationExecutionContext:
    prompt_config_path: Path
    model_api_contract: ModelApiContract
    generation_contract: GenerationContract


@dataclass(frozen=True)
class GeneratedTextResult:
    index: int
    text: str | None


class TextOutputRepository:
    """Persists generated synthetic records."""

    def remove_existing_outputs(self, output_dir: Path) -> None:
        if output_dir.exists():
            for output_file_path in output_dir.glob("*.jsonl"):
                output_file_path.unlink()

    def load_completed_indices(self, output_file_path: Path) -> set[int]:
        completed_indices: set[int] = set()

        if not output_file_path.exists():
            return completed_indices

        with output_file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            for line in input_file:
                if line.strip():
                    completed_indices.add(json.loads(line).get("idx", 0))

        return completed_indices

    def append_records(self, output_file_path: Path, records: list[dict]) -> None:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        with output_file_path.open("a", encoding=DEFAULT_ENCODING) as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


class GeneratedTextSanitizationService:
    """Applies deterministic cleanup to generated text."""

    def sanitize(self, text: str) -> str:
        text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"`{1,3}", "", text)
        return text.strip()


class ModelGenerationGateway:
    """Requests generated text from the configured model endpoint."""

    def __init__(
        self,
        prompt_service: PromptTemplateService,
        sanitization_service: GeneratedTextSanitizationService,
    ) -> None:
        self._prompt_service = prompt_service
        self._sanitization_service = sanitization_service

    async def generate_text(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        model_api_contract: ModelApiContract,
        component: str,
        domain: str,
        index: int,
        mode: str,
        secondaries: list[str],
    ) -> GeneratedTextResult:
        payload = {
            "model": model_api_contract.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self._prompt_service.build_generation_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._prompt_service.build_generation_user_prompt(
                        component=component,
                        domain=domain,
                        mode=mode,
                        secondaries=secondaries,
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
                    text = response.json()["choices"][0]["message"]["content"].strip()
                    sanitized_text = self._sanitization_service.sanitize(text)

                    if len(sanitized_text) < 30:
                        raise ValueError("Generated text is shorter than the minimum length.")

                    if sanitized_text.lstrip().startswith("{"):
                        raise ValueError("Generated content is JSON instead of plain text.")

                    return GeneratedTextResult(index=index, text=sanitized_text)

                except asyncio.TimeoutError:
                    if attempt < model_api_contract.retry_limit - 1:
                        await asyncio.sleep(model_api_contract.retry_delay)
                    else:
                        return GeneratedTextResult(index=index, text=None)
                except Exception:
                    if attempt < model_api_contract.retry_limit - 1:
                        await asyncio.sleep(model_api_contract.retry_delay + random.uniform(0, 2))
                    else:
                        return GeneratedTextResult(index=index, text=None)

        return GeneratedTextResult(index=index, text=None)


class GenerationProgressReporter:
    """Reports operational generation progress."""

    async def report(self, counters: dict[str, int], total: int, interval: float = 3.0) -> None:
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            percentage = 100 * counters["done"] / total if total > 0 else 100
            rate = counters["done"] / elapsed_time if elapsed_time > 0 else 0
            estimated_seconds = (total - counters["done"]) / rate if rate > 0 else 0

            print(
                f"\r  [{percentage:5.1f}%] {counters['done']:>4}/{total}  "
                f"success={counters['success']} "
                f"isolated={counters['isolated']} combined={counters['combined']} "
                f"failed={counters['failed']} "
                f"{rate:.1f} req/s eta={estimated_seconds:.0f}s   ",
                end="",
                flush=True,
            )

            if counters["done"] >= total:
                break

            await asyncio.sleep(interval)

        print()


class SyntheticTextGenerationOrchestrator:
    """Coordinates the synthetic text generation lifecycle."""

    def __init__(
        self,
        output_repository: TextOutputRepository,
        sanitization_service: GeneratedTextSanitizationService,
        progress_reporter: GenerationProgressReporter,
    ) -> None:
        self._output_repository = output_repository
        self._sanitization_service = sanitization_service
        self._progress_reporter = progress_reporter

    async def execute(self, execution_context: GenerationExecutionContext) -> None:
        prompt_config = PromptConfigRepository().load(execution_context.prompt_config_path)
        prompt_service = PromptTemplateService(prompt_config)
        generation_gateway = ModelGenerationGateway(
            prompt_service=prompt_service,
            sanitization_service=self._sanitization_service,
        )

        generation_contract = execution_context.generation_contract
        generation_contract.output_dir.mkdir(parents=True, exist_ok=True)

        if not generation_contract.resume:
            self._output_repository.remove_existing_outputs(generation_contract.output_dir)
            print("Existing output files removed.")

        total = (
            len(generation_contract.components)
            * len(generation_contract.domains)
            * generation_contract.examples_per_component
        )
        counters = {"done": 0, "success": 0, "failed": 0, "isolated": 0, "combined": 0}
        semaphore = asyncio.Semaphore(generation_contract.concurrency)
        file_lock = asyncio.Lock()

        limits = httpx.Limits(
            max_connections=generation_contract.concurrency + 4,
            max_keepalive_connections=generation_contract.concurrency,
        )

        print("Starting synthetic text generation.")
        print(f"Output directory: {generation_contract.output_dir.resolve()}")
        print(f"Total requested records: {total}")

        async with httpx.AsyncClient(
            timeout=execution_context.model_api_contract.http_timeout,
            limits=limits,
        ) as client:
            tasks = [
                self._generate_file(
                    client=client,
                    semaphore=semaphore,
                    file_lock=file_lock,
                    counters=counters,
                    generation_gateway=generation_gateway,
                    prompt_service=prompt_service,
                    model_api_contract=execution_context.model_api_contract,
                    generation_contract=generation_contract,
                    component=component,
                    domain=domain,
                )
                for component in generation_contract.components
                for domain in generation_contract.domains
            ]

            await asyncio.gather(self._progress_reporter.report(counters, total), *tasks)

        print("Execution completed.")
        print(
            "Records: "
            f"success={counters['success']}, "
            f"isolated={counters['isolated']}, "
            f"combined={counters['combined']}, "
            f"failed={counters['failed']}"
        )

    async def _generate_file(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        file_lock: asyncio.Lock,
        counters: dict[str, int],
        generation_gateway: ModelGenerationGateway,
        prompt_service: PromptTemplateService,
        model_api_contract: ModelApiContract,
        generation_contract: GenerationContract,
        component: str,
        domain: str,
    ) -> None:
        output_file_path = generation_contract.output_dir / f"{component.lower()}_{domain}.jsonl"
        completed_indices = self._output_repository.load_completed_indices(output_file_path)
        pending_indices = [
            index
            for index in range(1, generation_contract.examples_per_component + 1)
            if index not in completed_indices
        ]

        if not pending_indices:
            async with file_lock:
                counters["done"] += generation_contract.examples_per_component
            return

        half_point = generation_contract.examples_per_component // 2
        secondaries = prompt_service.natural_combinations.get(component, [])
        results = await asyncio.gather(
            *[
                generation_gateway.generate_text(
                    client=client,
                    semaphore=semaphore,
                    model_api_contract=model_api_contract,
                    component=component,
                    domain=domain,
                    index=index,
                    mode="isolated" if index <= half_point else "combined",
                    secondaries=secondaries,
                )
                for index in pending_indices
            ]
        )

        async with file_lock:
            records_to_append: list[dict] = []

            for result in results:
                mode = "isolated" if result.index <= half_point else "combined"

                if result.text is not None:
                    records_to_append.append(
                        {
                            "id": f"{component.lower()}_{domain}_{result.index:04d}",
                            "idx": result.index,
                            "component": component,
                            "domain": domain,
                            "mode": mode,
                            "secondaries": secondaries if mode == "combined" else [],
                            "text": result.text,
                        }
                    )
                    counters["success"] += 1
                    counters[mode] += 1
                else:
                    counters["failed"] += 1

                counters["done"] += 1

            self._output_repository.append_records(output_file_path, records_to_append)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic MINDSPACE training texts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROMPT_CONFIG_PATH)
    parser.add_argument("--model-api-url", type=str, default=DEFAULT_MODEL_API_URL)
    parser.add_argument("--model-api-key", type=str, default=DEFAULT_MODEL_API_KEY)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--components", nargs="+", default=None)
    parser.add_argument("--domains", nargs="+", choices=list(DEFAULT_DOMAINS), default=None)
    parser.add_argument("--n", type=int, default=DEFAULT_EXAMPLES_PER_COMPONENT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> GenerationExecutionContext:
    prompt_config = PromptConfigRepository().load(args.config)
    prompt_service = PromptTemplateService(prompt_config)
    examples_per_component = args.n if args.n % 2 == 0 else args.n + 1

    return GenerationExecutionContext(
        prompt_config_path=args.config,
        model_api_contract=ModelApiContract(
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
        generation_contract=GenerationContract(
            output_dir=args.output_dir,
            components=tuple(args.components or prompt_service.component_full_names),
            domains=tuple(args.domains or DEFAULT_DOMAINS),
            examples_per_component=examples_per_component,
            concurrency=args.concurrency,
            resume=not args.no_resume,
        ),
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = SyntheticTextGenerationOrchestrator(
        output_repository=TextOutputRepository(),
        sanitization_service=GeneratedTextSanitizationService(),
        progress_reporter=GenerationProgressReporter(),
    )
    asyncio.run(orchestrator.execute(execution_context))


if __name__ == "__main__":
    main()
