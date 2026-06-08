#!/usr/bin/env python3
"""
Name: complaint_table_builder
Input: JSON classification result files
Output: tr_completa.csv, tr_revisao.csv, tr_summary.json
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import csv
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_RESULT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus" / "ra_output"
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus"
DEFAULT_ENCODING = "utf-8"
CLASSIFICATION_RESULT_PATTERN = "RA_*.json"

COMPONENTS = ("M", "I", "N", "D", "S", "P", "A", "C", "E")

COMPLAINT_TABLE_FIELDS = (
    ["id", "ano", "janela", "complaint_date", "filename"]
    + list(COMPONENTS)
    + ["primary", "nudge_or_sludge", "confidence", "justificativa"]
    + ["needs_review", "review_reasons"]
    + ["_no_text", "_parse_error"]
)


class ComplaintTableBuildError(Exception):
    """Base exception for complaint table build failures."""


class JsonClassificationLoadError(ComplaintTableBuildError):
    """Raised when a JSON classification file cannot be loaded."""


@dataclass(frozen=True)
class ComplaintInputContract:
    result_dir: Path
    result_pattern: str = CLASSIFICATION_RESULT_PATTERN


@dataclass(frozen=True)
class ComplaintOutputContract:
    output_dir: Path
    full_table_file_name: str = "tr_completa.csv"
    review_table_file_name: str = "tr_revisao.csv"
    summary_file_name: str = "tr_summary.json"
    encoding: str = DEFAULT_ENCODING


@dataclass(frozen=True)
class ComplaintTableExecutionContext:
    input_contract: ComplaintInputContract
    output_contract: ComplaintOutputContract


@dataclass(frozen=True)
class ComplaintTableBuildResult:
    full_table_path: Path
    review_table_path: Path
    summary_path: Path
    total_rows: int
    review_rows: int
    eligible_rows: int
    no_text_rows: int
    parse_error_rows: int


class ClassificationResultRepositoryProtocol(Protocol):
    def load_classification_results(
        self,
        input_contract: ComplaintInputContract,
    ) -> list[dict]:
        """Load JSON classification records."""


class TablePersistenceProtocol(Protocol):
    def save_csv(
        self,
        output_file_path: Path,
        fieldnames: Iterable[str],
        rows: Iterable[dict],
    ) -> None:
        """Persist rows to a CSV output file."""

    def save_json(self, output_file_path: Path, payload: dict) -> None:
        """Persist a JSON output file."""


class JsonClassificationResultRepository:
    """Loads classification result records from a directory."""

    def load_classification_results(
        self,
        input_contract: ComplaintInputContract,
    ) -> list[dict]:
        records: list[dict] = []

        for json_file_path in sorted(input_contract.result_dir.glob(input_contract.result_pattern)):
            try:
                records.append(
                    json.loads(json_file_path.read_text(encoding=DEFAULT_ENCODING))
                )
            except Exception as error:
                raise JsonClassificationLoadError(
                    f"Unable to load JSON classification record: {json_file_path.name}"
                ) from error

        return records


class FileTablePersistenceService:
    """Persists CSV and JSON output contracts."""

    def save_csv(
        self,
        output_file_path: Path,
        fieldnames: Iterable[str],
        rows: Iterable[dict],
    ) -> None:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        with output_file_path.open("w", newline="", encoding=DEFAULT_ENCODING) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    def save_json(self, output_file_path: Path, payload: dict) -> None:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=DEFAULT_ENCODING,
        )


class ComplaintRowFactory:
    """Builds complaint table rows from classification records."""

    def build_row(self, classification_record: dict) -> dict:
        component_vector = classification_record.get("vector", {})

        row: dict[str, object] = {
            "id": classification_record.get("id", ""),
            "ano": classification_record.get("ano"),
            "janela": classification_record.get("janela", ""),
            "complaint_date": classification_record.get("complaint_date", ""),
            "filename": classification_record.get("filename", ""),
            "primary": classification_record.get("primary", "indeterminado"),
            "nudge_or_sludge": classification_record.get(
                "nudge_or_sludge",
                "indeterminado",
            ),
            "confidence": classification_record.get("confidence", "baixa"),
            "justificativa": classification_record.get("justificativa", ""),
            "needs_review": classification_record.get("needs_review", 0),
            "review_reasons": classification_record.get("review_reasons", ""),
            "_no_text": 1 if classification_record.get("_no_text") else 0,
            "_parse_error": 1 if classification_record.get("_parse_error") else 0,
        }

        for component in COMPONENTS:
            component_value = component_vector.get(component, False)
            row[component] = 1 if (component_value is True or component_value == 1) else 0

        return row


class ComplaintSummaryComputationService:
    """Computes the summary payload required by downstream consumers."""

    def compute_summary(self, rows: list[dict]) -> dict:
        eligible_rows = [
            row
            for row in rows
            if not row.get("_no_text")
            and not row.get("_parse_error")
            and row.get("ano") is not None
        ]

        rows_by_year: dict[int, list[dict]] = {}
        for row in eligible_rows:
            year = int(row["ano"])
            rows_by_year.setdefault(year, []).append(row)

        summary: dict[int, dict] = {}

        for year, year_rows in sorted(rows_by_year.items()):
            total_complaint_rows = len(year_rows)
            summary[year] = {
                "N_R": total_complaint_rows,
                "components": {},
            }

            for component in COMPONENTS:
                present_count = sum(
                    1 for row in year_rows if int(row.get(component, 0)) == 1
                )
                proportion = (
                    round(present_count / total_complaint_rows, 4)
                    if total_complaint_rows > 0
                    else 0.0
                )
                summary[year]["components"][component] = {
                    "n_R": present_count,
                    "p_R": proportion,
                }

            primary_distribution: dict[str, int] = {}
            for row in year_rows:
                primary_value = row.get("primary", "indeterminado")
                primary_distribution[primary_value] = (
                    primary_distribution.get(primary_value, 0) + 1
                )
            summary[year]["primary_distribution"] = primary_distribution

            window_distribution: dict[str, int] = {}
            for row in year_rows:
                window_value = row.get("janela", "?")
                window_distribution[window_value] = (
                    window_distribution.get(window_value, 0) + 1
                )
            summary[year]["janela_distribution"] = window_distribution

        return summary


class ComplaintTableMetricsService:
    """Computes operational row counts."""

    def compute_metrics(self, rows: list[dict]) -> dict[str, int]:
        return {
            "total_rows": len(rows),
            "eligible_rows": sum(
                1 for row in rows if not row["_no_text"] and not row["_parse_error"]
            ),
            "no_text_rows": sum(1 for row in rows if row["_no_text"]),
            "parse_error_rows": sum(1 for row in rows if row["_parse_error"]),
        }


class ComplaintTableOrchestrator:
    """Coordinates the complaint table build lifecycle."""

    def __init__(
        self,
        result_repository: ClassificationResultRepositoryProtocol,
        persistence_service: TablePersistenceProtocol,
        row_factory: ComplaintRowFactory,
        summary_service: ComplaintSummaryComputationService,
        metrics_service: ComplaintTableMetricsService,
    ) -> None:
        self._result_repository = result_repository
        self._persistence_service = persistence_service
        self._row_factory = row_factory
        self._summary_service = summary_service
        self._metrics_service = metrics_service

    def execute(
        self,
        execution_context: ComplaintTableExecutionContext,
    ) -> ComplaintTableBuildResult | None:
        output_contract = execution_context.output_contract
        output_contract.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading JSON classification records...")
        classification_records = self._result_repository.load_classification_results(
            execution_context.input_contract,
        )
        print(f"JSON classification records loaded: {len(classification_records)}")

        if not classification_records:
            print("No JSON classification records found.")
            return None

        print("Building complaint rows...")
        complaint_rows = [
            self._row_factory.build_row(classification_record)
            for classification_record in classification_records
        ]
        review_rows = [row for row in complaint_rows if row["needs_review"]]
        metrics = self._metrics_service.compute_metrics(complaint_rows)

        full_table_path = output_contract.output_dir / output_contract.full_table_file_name
        review_table_path = output_contract.output_dir / output_contract.review_table_file_name
        summary_path = output_contract.output_dir / output_contract.summary_file_name

        print("Saving complaint table outputs...")
        self._persistence_service.save_csv(
            full_table_path,
            COMPLAINT_TABLE_FIELDS,
            complaint_rows,
        )
        self._persistence_service.save_csv(
            review_table_path,
            COMPLAINT_TABLE_FIELDS,
            review_rows,
        )
        self._persistence_service.save_json(
            summary_path,
            self._summary_service.compute_summary(complaint_rows),
        )

        return ComplaintTableBuildResult(
            full_table_path=full_table_path,
            review_table_path=review_table_path,
            summary_path=summary_path,
            total_rows=metrics["total_rows"],
            review_rows=len(review_rows),
            eligible_rows=metrics["eligible_rows"],
            no_text_rows=metrics["no_text_rows"],
            parse_error_rows=metrics["parse_error_rows"],
        )


class ComplaintTableDependencyFactory:
    """Builds dependencies for the complaint table pipeline."""

    def build_orchestrator(self) -> ComplaintTableOrchestrator:
        return ComplaintTableOrchestrator(
            result_repository=JsonClassificationResultRepository(),
            persistence_service=FileTablePersistenceService(),
            row_factory=ComplaintRowFactory(),
            summary_service=ComplaintSummaryComputationService(),
            metrics_service=ComplaintTableMetricsService(),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the complaint table output files.",
    )
    parser.add_argument("--ra-out-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> ComplaintTableExecutionContext:
    return ComplaintTableExecutionContext(
        input_contract=ComplaintInputContract(
            result_dir=args.ra_out_dir,
        ),
        output_contract=ComplaintOutputContract(
            output_dir=args.out_dir,
        ),
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = ComplaintTableDependencyFactory().build_orchestrator()
    result = orchestrator.execute(execution_context)

    if result is None:
        return

    print("Execution completed.")
    print(f"Full table: {result.full_table_path}")
    print(f"Review table: {result.review_table_path}")
    print(f"Summary: {result.summary_path}")
    print(
        "Rows: "
        f"total={result.total_rows}, "
        f"eligible={result.eligible_rows}, "
        f"review={result.review_rows}, "
        f"no_text={result.no_text_rows}, "
        f"parse_errors={result.parse_error_rows}"
    )


if __name__ == "__main__":
    main()
