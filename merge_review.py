#!/usr/bin/env python3
"""
Name: reviewed_table_merge
Input: complete CSV files and review CSV files
Output: revised complete CSV files
Usage: run as a Python script
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Protocol


DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_CORPUS_DIR = DEFAULT_WORKSPACE_ROOT / "corpus"
DEFAULT_ENCODING = "utf-8"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")


class TableMergeError(Exception):
    """Base exception for table merge failures."""


class DuplicateOutputIdentifierError(TableMergeError):
    """Raised when duplicate identifiers remain after merge execution."""


@dataclass(frozen=True)
class ReviewMergeContract:
    complete_file_path: Path
    review_file_path: Path
    output_file_path: Path
    label: str


@dataclass(frozen=True)
class CsvTable:
    fieldnames: list[str]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class ReviewMergeResult:
    label: str
    output_file_path: Path
    complete_raw_rows: int
    complete_unique_rows: int
    review_rows: int
    substituted_rows: int
    ignored_review_rows: int
    skipped_duplicate_rows: int
    output_rows: int


class CsvTableRepositoryProtocol(Protocol):
    def load_table(self, file_path: Path) -> CsvTable | None:
        """Load a CSV table if it exists."""

    def save_table(
        self,
        file_path: Path,
        fieldnames: Iterable[str],
        rows: Iterable[dict[str, str]],
    ) -> None:
        """Save a CSV table."""


class CsvTableRepository:
    """Loads and saves CSV tables."""

    def load_table(self, file_path: Path) -> CsvTable | None:
        if not file_path.exists():
            return None

        with file_path.open(newline="", encoding=DEFAULT_ENCODING) as input_file:
            reader = csv.DictReader(input_file)
            return CsvTable(
                fieldnames=list(reader.fieldnames or []),
                rows=list(reader),
            )

    def save_table(
        self,
        file_path: Path,
        fieldnames: Iterable[str],
        rows: Iterable[dict[str, str]],
    ) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", newline="", encoding=DEFAULT_ENCODING) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field_name: self._sanitize_value(value)
                        for field_name, value in row.items()
                    }
                )

    def _sanitize_value(self, value: object) -> object:
        if isinstance(value, str):
            return (
                value.replace("\r\n", " ")
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )

        return value


class ReviewMergePolicy:
    """Applies review rows over complete rows while preserving complete table scope."""

    def merge_tables(
        self,
        complete_table: CsvTable,
        review_table: CsvTable,
    ) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
        extra_fieldnames = [
            field_name
            for field_name in review_table.fieldnames
            if field_name not in complete_table.fieldnames
        ]
        merged_fieldnames = complete_table.fieldnames + extra_fieldnames

        review_row_by_identifier = {
            row["id"]: row for row in review_table.rows if "id" in row
        }
        complete_identifiers = {
            row["id"] for row in complete_table.rows if "id" in row
        }
        ignored_review_identifiers = set(review_row_by_identifier) - complete_identifiers

        merged_rows: list[dict[str, str]] = []
        seen_identifiers: set[str] = set()
        substituted_rows = 0
        skipped_duplicate_rows = 0

        for row in complete_table.rows:
            row_identifier = row["id"]

            if row_identifier in seen_identifiers:
                skipped_duplicate_rows += 1
                continue

            seen_identifiers.add(row_identifier)

            if row_identifier in review_row_by_identifier:
                merged_row = dict(row)
                merged_row.update(review_row_by_identifier[row_identifier])
                merged_rows.append(merged_row)
                substituted_rows += 1
            else:
                merged_rows.append(dict(row))

        self._validate_unique_output_identifiers(merged_rows)

        metrics = {
            "substituted_rows": substituted_rows,
            "ignored_review_rows": len(ignored_review_identifiers),
            "skipped_duplicate_rows": skipped_duplicate_rows,
            "complete_unique_rows": len(complete_table.rows) - skipped_duplicate_rows,
        }

        return merged_fieldnames, merged_rows, metrics

    def _validate_unique_output_identifiers(self, rows: list[dict[str, str]]) -> None:
        output_identifiers = [row["id"] for row in rows]

        if len(output_identifiers) != len(set(output_identifiers)):
            raise DuplicateOutputIdentifierError(
                "Duplicate identifiers remain after merge execution."
            )


class ReviewMergeOrchestrator:
    """Coordinates review merge execution for configured table contracts."""

    def __init__(
        self,
        table_repository: CsvTableRepositoryProtocol,
        merge_policy: ReviewMergePolicy,
    ) -> None:
        self._table_repository = table_repository
        self._merge_policy = merge_policy

    def execute(self, merge_contracts: Iterable[ReviewMergeContract]) -> list[ReviewMergeResult]:
        results: list[ReviewMergeResult] = []

        for merge_contract in merge_contracts:
            result = self._execute_single_contract(merge_contract)

            if result is not None:
                results.append(result)

        return results

    def _execute_single_contract(
        self,
        merge_contract: ReviewMergeContract,
    ) -> ReviewMergeResult | None:
        print(f"Processing merge contract: {merge_contract.label}")

        complete_table = self._table_repository.load_table(
            merge_contract.complete_file_path,
        )
        if complete_table is None:
            print(f"Complete table not found: {merge_contract.complete_file_path}")
            return None

        review_table = self._table_repository.load_table(
            merge_contract.review_file_path,
        )
        if review_table is None:
            print(f"Review table not found: {merge_contract.review_file_path}")
            return None

        fieldnames, merged_rows, metrics = self._merge_policy.merge_tables(
            complete_table=complete_table,
            review_table=review_table,
        )

        self._table_repository.save_table(
            file_path=merge_contract.output_file_path,
            fieldnames=fieldnames,
            rows=merged_rows,
        )

        result = ReviewMergeResult(
            label=merge_contract.label,
            output_file_path=merge_contract.output_file_path,
            complete_raw_rows=len(complete_table.rows),
            complete_unique_rows=metrics["complete_unique_rows"],
            review_rows=len(review_table.rows),
            substituted_rows=metrics["substituted_rows"],
            ignored_review_rows=metrics["ignored_review_rows"],
            skipped_duplicate_rows=metrics["skipped_duplicate_rows"],
            output_rows=len(merged_rows),
        )

        print(
            "Merge completed: "
            f"label={result.label}, "
            f"output_rows={result.output_rows}, "
            f"substituted_rows={result.substituted_rows}, "
            f"ignored_review_rows={result.ignored_review_rows}, "
            f"skipped_duplicate_rows={result.skipped_duplicate_rows}"
        )

        return result


class ReviewMergeContractFactory:
    """Builds default review merge contracts."""

    def build_default_contracts(self) -> tuple[ReviewMergeContract, ...]:
        return (
            ReviewMergeContract(
                complete_file_path=DEFAULT_CORPUS_DIR / "tr_completa.csv",
                review_file_path=DEFAULT_CORPUS_DIR / "tr_revisao_2026-05-06.csv",
                output_file_path=DEFAULT_CORPUS_DIR / f"tr_completa_revisado_{CURRENT_DATE}.csv",
                label="TR",
            ),
            ReviewMergeContract(
                complete_file_path=DEFAULT_CORPUS_DIR / "ti_completa.csv",
                review_file_path=DEFAULT_CORPUS_DIR / "ti_revisao_2026-05-10.csv",
                output_file_path=DEFAULT_CORPUS_DIR / f"ti_completa_revisado_{CURRENT_DATE}.csv",
                label="TI",
            ),
        )


def build_orchestrator() -> ReviewMergeOrchestrator:
    return ReviewMergeOrchestrator(
        table_repository=CsvTableRepository(),
        merge_policy=ReviewMergePolicy(),
    )


def main() -> None:
    merge_contracts = ReviewMergeContractFactory().build_default_contracts()
    orchestrator = build_orchestrator()
    results = orchestrator.execute(merge_contracts)

    print("Execution completed.")
    print(f"Merge contracts completed: {len(results)}")


if __name__ == "__main__":
    main()
