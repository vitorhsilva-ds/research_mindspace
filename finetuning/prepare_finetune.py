#!/usr/bin/env python3
"""
Name: training_dataset_preparation_pipeline
Input: consolidated classification JSONL dataset
Output: dataset_train.jsonl, dataset_val.jsonl, dataset_stats.json
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_FILE = Path("./dataset_finetune_full.jsonl")
DEFAULT_OUTPUT_DIR = Path("./finetune_data")
DEFAULT_TRAIN_FILE_NAME = "dataset_train.jsonl"
DEFAULT_VALIDATION_FILE_NAME = "dataset_val.jsonl"
DEFAULT_STATS_FILE_NAME = "dataset_stats.json"
DEFAULT_VALIDATION_RATIO = 0.20
DEFAULT_RANDOM_SEED = 42
DEFAULT_ENCODING = "utf-8"

MINDSPACE_COMPONENTS = (
    "Messenger",
    "Incentives",
    "Norms",
    "Defaults",
    "Salience",
    "Priming",
    "Affect",
    "Commitments",
    "Ego",
)
VALID_COMPONENTS = set(MINDSPACE_COMPONENTS)
PRIMARY_SEPARATORS = (",", " e ", " and ", "/")
MIN_PRIMARY_COUNT = 9999


@dataclass(frozen=True)
class TrainingDatasetInputContract:
    input_file_path: Path


@dataclass(frozen=True)
class TrainingDatasetOutputContract:
    output_dir: Path
    train_file_name: str = DEFAULT_TRAIN_FILE_NAME
    validation_file_name: str = DEFAULT_VALIDATION_FILE_NAME
    stats_file_name: str = DEFAULT_STATS_FILE_NAME


@dataclass(frozen=True)
class TrainingDatasetSplitConfig:
    validation_ratio: float
    random_seed: int
    drop_low_confidence: bool
    drop_empty_chain_of_thought: bool = True


@dataclass(frozen=True)
class TrainingDatasetExecutionContext:
    input_contract: TrainingDatasetInputContract
    output_contract: TrainingDatasetOutputContract
    split_config: TrainingDatasetSplitConfig
    stats_only: bool


@dataclass(frozen=True)
class FilteredRecords:
    records: list[dict]
    stats: dict[str, int]


@dataclass(frozen=True)
class DatasetSplitResult:
    train_records: list[dict]
    validation_records: list[dict]


class JsonLinesRepository:
    """Loads and saves JSON Lines records."""

    def load_records(self, file_path: Path) -> list[dict]:
        records: list[dict] = []

        with file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            for line in input_file:
                if line.strip():
                    records.append(json.loads(line))

        return records

    def save_records(self, file_path: Path, records: Iterable[dict]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding=DEFAULT_ENCODING) as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


class JsonRepository:
    """Saves JSON payloads."""

    def save(self, file_path: Path, payload: dict) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=DEFAULT_ENCODING,
        )


class PrimaryComponentNormalizationService:
    """Normalizes primary component values."""

    def normalize_primary(self, raw_value: str) -> str:
        stripped_value = raw_value.strip()

        if stripped_value in VALID_COMPONENTS:
            return stripped_value

        for separator in PRIMARY_SEPARATORS:
            if separator in stripped_value:
                parts = [part.strip() for part in stripped_value.split(separator)]
                for part in parts:
                    if part in VALID_COMPONENTS:
                        return part

        for component in MINDSPACE_COMPONENTS:
            if component in stripped_value:
                return component

        return stripped_value


class RecordQualityFilterService:
    """Filters records according to training dataset quality rules."""

    def filter_records(
        self,
        records: list[dict],
        split_config: TrainingDatasetSplitConfig,
    ) -> FilteredRecords:
        stats = {
            "total_input": len(records),
            "dropped_empty_cot": 0,
            "dropped_low_conf": 0,
            "dropped_no_primary": 0,
            "kept": 0,
        }
        kept_records: list[dict] = []

        for record in records:
            if (
                split_config.drop_empty_chain_of_thought
                and record["metadata"]["cot_length"] == 0
            ):
                stats["dropped_empty_cot"] += 1
                continue

            primary = record["output"].get("primary", "").strip()
            if not primary:
                stats["dropped_no_primary"] += 1
                continue

            if (
                split_config.drop_low_confidence
                and record["metadata"]["confidence"] == "baixo"
            ):
                stats["dropped_low_conf"] += 1
                continue

            kept_records.append(record)

        stats["kept"] = len(kept_records)
        return FilteredRecords(records=kept_records, stats=stats)


class PrimaryResolutionService:
    """Resolves the effective training primary component."""

    def __init__(self, normalization_service: PrimaryComponentNormalizationService) -> None:
        self._normalization_service = normalization_service

    def compute_primary_counts(self, records: list[dict]) -> dict[str, int]:
        counts = {component: 0 for component in MINDSPACE_COMPONENTS}

        for record in records:
            raw_primary = record["output"].get("primary", "")
            normalized_primary = self._normalization_service.normalize_primary(raw_primary)

            if normalized_primary in counts:
                counts[normalized_primary] += 1

        return counts

    def resolve_primary(
        self,
        record: dict,
        low_count_components: set[str],
    ) -> tuple[str, str]:
        gold_component = record["component_gold"]

        if gold_component in low_count_components:
            return gold_component, "gold_fallback"

        raw_primary = record["output"].get("primary", "")
        normalized_primary = self._normalization_service.normalize_primary(raw_primary)
        return normalized_primary, "classifier"


class ChatMlConversionService:
    """Converts classified records into ChatML-compatible training records."""

    def __init__(
        self,
        primary_resolution_service: PrimaryResolutionService,
    ) -> None:
        self._primary_resolution_service = primary_resolution_service

    def convert_record(self, record: dict, low_count_components: set[str]) -> dict:
        raw_primary = record["output"].get("primary", "")
        primary, primary_source = self._primary_resolution_service.resolve_primary(
            record,
            low_count_components,
        )

        return {
            "conversations": [
                {
                    "role": "system",
                    "content": record["instruction"],
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze the text below and produce the MINDSPACE multilabel "
                        "classification.\n\n"
                        f"DOMAIN: {record['domain']}\n\n"
                        f"TEXT:\n{record['input']}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": self._build_assistant_response(record),
                },
            ],
            "metadata": {
                "id": record["id"],
                "component_gold": record["component_gold"],
                "component_primary": primary,
                "primary_source": primary_source,
                "primary_raw": raw_primary,
                "domain": record["domain"],
                "mode": record.get("mode", "unknown"),
                "gold_in_vector": record["metadata"]["gold_in_vector"],
                "n_active": record["metadata"]["n_active"],
                "confidence": record["metadata"]["confidence"],
                "nudge_or_sludge": record["metadata"]["nudge_or_sludge"],
                "cot_length": record["metadata"]["cot_length"],
            },
        }

    def _build_assistant_response(self, record: dict) -> str:
        chain_of_thought = record["chain_of_thought"].strip()
        output_json = json.dumps(record["output"], ensure_ascii=False, indent=2)

        if chain_of_thought:
            return f"{chain_of_thought}\n\n{output_json}"

        return output_json


class StratifiedSplitService:
    """Builds a stratified split by effective primary component and domain."""

    def __init__(
        self,
        primary_resolution_service: PrimaryResolutionService,
    ) -> None:
        self._primary_resolution_service = primary_resolution_service

    def split(
        self,
        records: list[dict],
        validation_ratio: float,
        random_seed: int,
        low_count_components: set[str],
    ) -> DatasetSplitResult:
        random_generator = random.Random(random_seed)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for record in records:
            primary, _ = self._primary_resolution_service.resolve_primary(
                record,
                low_count_components,
            )
            groups[(primary, record["domain"])].append(record)

        train_records: list[dict] = []
        validation_records: list[dict] = []

        for group in groups.values():
            random_generator.shuffle(group)
            validation_count = max(1, round(len(group) * validation_ratio))
            validation_records.extend(group[:validation_count])
            train_records.extend(group[validation_count:])

        random_generator.shuffle(train_records)
        random_generator.shuffle(validation_records)

        return DatasetSplitResult(
            train_records=train_records,
            validation_records=validation_records,
        )


class TrainingDatasetStatsService:
    """Builds operational statistics for the generated split."""

    def build_stats(
        self,
        train_records: list[dict],
        validation_records: list[dict],
        filter_stats: dict,
    ) -> dict:
        all_records = train_records + validation_records
        records_by_component: dict[str, list[dict]] = defaultdict(list)

        for record in all_records:
            records_by_component[record["metadata"]["component_primary"]].append(record)

        return {
            "filter": filter_stats,
            "train_size": len(train_records),
            "val_size": len(validation_records),
            "total": len(all_records),
            "by_component": {
                component: {
                    "train": sum(
                        1
                        for record in train_records
                        if record["metadata"]["component_primary"] == component
                    ),
                    "val": sum(
                        1
                        for record in validation_records
                        if record["metadata"]["component_primary"] == component
                    ),
                }
                for component in records_by_component
            },
        }


class TrainingDatasetPreparationOrchestrator:
    """Coordinates training dataset preparation."""

    def __init__(
        self,
        jsonl_repository: JsonLinesRepository,
        json_repository: JsonRepository,
        filter_service: RecordQualityFilterService,
        primary_resolution_service: PrimaryResolutionService,
        split_service: StratifiedSplitService,
        conversion_service: ChatMlConversionService,
        stats_service: TrainingDatasetStatsService,
    ) -> None:
        self._jsonl_repository = jsonl_repository
        self._json_repository = json_repository
        self._filter_service = filter_service
        self._primary_resolution_service = primary_resolution_service
        self._split_service = split_service
        self._conversion_service = conversion_service
        self._stats_service = stats_service

    def execute(self, execution_context: TrainingDatasetExecutionContext) -> None:
        print("Loading consolidated dataset...")
        raw_records = self._jsonl_repository.load_records(
            execution_context.input_contract.input_file_path,
        )
        print(f"Input records loaded: {len(raw_records)}")

        filtered_records = self._filter_service.filter_records(
            records=raw_records,
            split_config=execution_context.split_config,
        )
        primary_counts = self._primary_resolution_service.compute_primary_counts(
            filtered_records.records,
        )
        low_count_components = {
            component
            for component, count in primary_counts.items()
            if count < MIN_PRIMARY_COUNT
        }

        split_result = self._split_service.split(
            records=filtered_records.records,
            validation_ratio=execution_context.split_config.validation_ratio,
            random_seed=execution_context.split_config.random_seed,
            low_count_components=low_count_components,
        )

        train_records = [
            self._conversion_service.convert_record(record, low_count_components)
            for record in split_result.train_records
        ]
        validation_records = [
            self._conversion_service.convert_record(record, low_count_components)
            for record in split_result.validation_records
        ]
        stats = self._stats_service.build_stats(
            train_records=train_records,
            validation_records=validation_records,
            filter_stats=filtered_records.stats,
        )

        print("Split completed.")
        print(
            "Records: "
            f"train={len(train_records)}, "
            f"validation={len(validation_records)}, "
            f"total={len(train_records) + len(validation_records)}"
        )

        if execution_context.stats_only:
            print("Stats-only mode enabled. Output files were not saved.")
            return

        output_contract = execution_context.output_contract
        train_file_path = output_contract.output_dir / output_contract.train_file_name
        validation_file_path = output_contract.output_dir / output_contract.validation_file_name
        stats_file_path = output_contract.output_dir / output_contract.stats_file_name

        self._jsonl_repository.save_records(train_file_path, train_records)
        self._jsonl_repository.save_records(validation_file_path, validation_records)
        self._json_repository.save(stats_file_path, stats)

        print("Execution completed.")
        print(f"Train file: {train_file_path}")
        print(f"Validation file: {validation_file_path}")
        print(f"Stats file: {stats_file_path}")


class TrainingDatasetDependencyFactory:
    """Builds dependencies for training dataset preparation."""

    def build_orchestrator(self) -> TrainingDatasetPreparationOrchestrator:
        normalization_service = PrimaryComponentNormalizationService()
        primary_resolution_service = PrimaryResolutionService(normalization_service)

        return TrainingDatasetPreparationOrchestrator(
            jsonl_repository=JsonLinesRepository(),
            json_repository=JsonRepository(),
            filter_service=RecordQualityFilterService(),
            primary_resolution_service=primary_resolution_service,
            split_service=StratifiedSplitService(primary_resolution_service),
            conversion_service=ChatMlConversionService(primary_resolution_service),
            stats_service=TrainingDatasetStatsService(),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a ChatML training dataset from classified records.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--drop-low-confidence", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--stats-only", action="store_true")
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> TrainingDatasetExecutionContext:
    return TrainingDatasetExecutionContext(
        input_contract=TrainingDatasetInputContract(input_file_path=args.input),
        output_contract=TrainingDatasetOutputContract(output_dir=args.out_dir),
        split_config=TrainingDatasetSplitConfig(
            validation_ratio=args.val_ratio,
            random_seed=args.seed,
            drop_low_confidence=args.drop_low_confidence,
        ),
        stats_only=args.stats_only,
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = TrainingDatasetDependencyFactory().build_orchestrator()
    orchestrator.execute(execution_context)


if __name__ == "__main__":
    main()
