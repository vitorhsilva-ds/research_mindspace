#!/usr/bin/env python3
"""
Name: analytical_base_table_builder
Input: reviewed interface and complaint CSV files
Output: abt.csv, abt_summary.txt
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus"
DEFAULT_ENCODING = "utf-8"

COMPONENTS = ("M", "I", "N", "D", "S", "P", "A", "C", "E")
YEARS = (2020, 2021, 2022, 2023, 2024, 2025)

COMPONENT_NAMES = {
    "M": "Messenger",
    "I": "Incentives",
    "N": "Norms",
    "D": "Defaults",
    "S": "Salience",
    "P": "Priming",
    "A": "Affect",
    "C": "Commitments",
    "E": "Ego",
}

ABT_FIELDS = [
    "comp",
    "ano",
    "N_I",
    "n_I",
    "p_I",
    "density_media",
    "N_R",
    "n_R",
    "p_R",
    "gap_IR",
]


class AnalyticalBaseTableBuildError(Exception):
    """Base exception for analytical base table build failures."""


class RequiredColumnMissingError(AnalyticalBaseTableBuildError):
    """Raised when a required input column is missing."""


@dataclass(frozen=True)
class ReviewedInputContract:
    interface_table_path: Path
    complaint_table_path: Path


@dataclass(frozen=True)
class AnalyticalBaseTableOutputContract:
    output_dir: Path
    table_file_name: str = "abt.csv"
    summary_file_name: str = "abt_summary.txt"
    encoding: str = DEFAULT_ENCODING


@dataclass(frozen=True)
class AnalyticalBaseTableExecutionContext:
    input_contract: ReviewedInputContract
    output_contract: AnalyticalBaseTableOutputContract


@dataclass(frozen=True)
class AnalyticalBaseTableBuildResult:
    table_path: Path
    summary_path: Path
    row_count: int


@dataclass(frozen=True)
class ReviewedTableRows:
    source_path: Path
    rows: list[dict[str, str]]


class CsvRepositoryProtocol(Protocol):
    def load_rows(self, file_path: Path) -> list[dict[str, str]]:
        """Load CSV rows from a file."""

    def save_rows(
        self,
        file_path: Path,
        fieldnames: Iterable[str],
        rows: Iterable[dict],
    ) -> None:
        """Save CSV rows to a file."""


class TextRepositoryProtocol(Protocol):
    def save_text(self, file_path: Path, text: str) -> None:
        """Save text to a file."""


class CsvRepository:
    """Loads and saves CSV files."""

    def load_rows(self, file_path: Path) -> list[dict[str, str]]:
        if not file_path.exists():
            sys.exit(
                f"Input file not found: {file_path}\n"
                "Run the required upstream merge process or provide an explicit path."
            )

        with file_path.open(newline="", encoding=DEFAULT_ENCODING) as input_file:
            return list(csv.DictReader(input_file))

    def save_rows(
        self,
        file_path: Path,
        fieldnames: Iterable[str],
        rows: Iterable[dict],
    ) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", newline="", encoding=DEFAULT_ENCODING) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)


class TextRepository:
    """Saves text output files."""

    def save_text(self, file_path: Path, text: str) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding=DEFAULT_ENCODING)


class ReviewedFileDiscoveryService:
    """Discovers the most recent reviewed CSV file for a configured prefix."""

    def find_latest_reviewed_file(self, corpus_dir: Path, prefix: str) -> Path | None:
        candidates = sorted(glob.glob(str(corpus_dir / f"{prefix}_revisado_*.csv")))

        if not candidates:
            return None

        return Path(candidates[-1])


class ComponentValueService:
    """Normalizes component activation values."""

    def is_active(self, value: object) -> bool:
        return str(value).strip() == "1"


class InputContractValidationService:
    """Validates reviewed input table schemas."""

    def validate_component_columns(
        self,
        rows: list[dict[str, str]],
        label: str,
    ) -> None:
        if not rows:
            return

        available_columns = set(rows[0].keys())
        missing_components = [
            component for component in COMPONENTS if component not in available_columns
        ]

        if missing_components:
            raise RequiredColumnMissingError(
                f"{label}: missing component columns: {missing_components}"
            )

    def get_missing_instance_columns(
        self,
        rows: list[dict[str, str]],
    ) -> list[str]:
        if not rows:
            return []

        available_columns = set(rows[0].keys())
        expected_instance_columns = [f"{component}_inst" for component in COMPONENTS]
        return [
            column for column in expected_instance_columns if column not in available_columns
        ]


class InterfaceStatisticsExtractionService:
    """Extracts interface statistics by year and component."""

    def __init__(self, component_value_service: ComponentValueService) -> None:
        self._component_value_service = component_value_service

    def extract_statistics(self, rows: list[dict[str, str]]) -> dict[tuple[int, str], dict]:
        rows_by_year = self._initialize_year_accumulator()
        invalid_year_values: set[object] = set()

        for row in rows:
            year = self._parse_year(row.get("ano", ""), invalid_year_values)

            if year is None or year not in YEARS:
                continue

            rows_by_year[year]["total"] += 1

            for component in COMPONENTS:
                if self._component_value_service.is_active(row.get(component, "0")):
                    rows_by_year[year][component]["n"] += 1
                    rows_by_year[year][component]["inst_sum"] += (
                        self._parse_instance_value(row.get(f"{component}_inst", "0"))
                    )

        if invalid_year_values:
            print(f"Invalid interface year values ignored: {sorted(invalid_year_values)}")

        return self._build_statistics(rows_by_year)

    def _initialize_year_accumulator(self) -> dict[int, dict]:
        return {
            year: {
                "total": 0,
                **{
                    component: {"n": 0, "inst_sum": 0.0}
                    for component in COMPONENTS
                },
            }
            for year in YEARS
        }

    def _parse_year(
        self,
        value: object,
        invalid_year_values: set[object],
    ) -> int | None:
        try:
            return int(str(value).strip())
        except ValueError:
            invalid_year_values.add(value)
            return None

    def _parse_instance_value(self, value: object) -> float:
        try:
            return float(str(value).strip() or "0")
        except ValueError:
            return 0.0

    def _build_statistics(self, rows_by_year: dict[int, dict]) -> dict[tuple[int, str], dict]:
        statistics: dict[tuple[int, str], dict] = {}

        for year in YEARS:
            year_data = rows_by_year[year]
            total_rows = year_data["total"]

            for component in COMPONENTS:
                active_count = year_data[component]["n"]
                density = (
                    year_data[component]["inst_sum"] / active_count
                    if active_count > 0
                    else 0.0
                )
                statistics[(year, component)] = {
                    "N_I": total_rows,
                    "n_I": active_count,
                    "p_I": round(active_count / total_rows, 4) if total_rows > 0 else 0.0,
                    "density_media": round(density, 4),
                }

        return statistics


class ComplaintStatisticsExtractionService:
    """Extracts complaint statistics by year and component."""

    def __init__(self, component_value_service: ComponentValueService) -> None:
        self._component_value_service = component_value_service

    def extract_statistics(self, rows: list[dict[str, str]]) -> dict[tuple[int, str], dict]:
        rows_by_year = self._initialize_year_accumulator()
        invalid_year_values: set[object] = set()

        for row in rows:
            year = self._parse_year(row.get("ano", ""), invalid_year_values)

            if year is None or year not in YEARS:
                continue

            rows_by_year[year]["total"] += 1

            for component in COMPONENTS:
                if self._component_value_service.is_active(row.get(component, "0")):
                    rows_by_year[year][component] += 1

        if invalid_year_values:
            print(f"Invalid complaint year values ignored: {sorted(invalid_year_values)}")

        return self._build_statistics(rows_by_year)

    def _initialize_year_accumulator(self) -> dict[int, dict]:
        return {
            year: {
                "total": 0,
                **{component: 0 for component in COMPONENTS},
            }
            for year in YEARS
        }

    def _parse_year(
        self,
        value: object,
        invalid_year_values: set[object],
    ) -> int | None:
        try:
            return int(str(value).strip())
        except ValueError:
            invalid_year_values.add(value)
            return None

    def _build_statistics(self, rows_by_year: dict[int, dict]) -> dict[tuple[int, str], dict]:
        statistics: dict[tuple[int, str], dict] = {}

        for year in YEARS:
            year_data = rows_by_year[year]
            total_rows = year_data["total"]

            for component in COMPONENTS:
                active_count = year_data[component]
                statistics[(year, component)] = {
                    "N_R": total_rows,
                    "n_R": active_count,
                    "p_R": round(active_count / total_rows, 4) if total_rows > 0 else 0.0,
                }

        return statistics


class AnalyticalBaseTableAssemblyService:
    """Builds analytical base table rows from extracted statistics."""

    def build_rows(
        self,
        interface_statistics: dict[tuple[int, str], dict],
        complaint_statistics: dict[tuple[int, str], dict],
    ) -> list[dict]:
        rows: list[dict] = []

        for component in COMPONENTS:
            for year in YEARS:
                interface_values = interface_statistics.get(
                    (year, component),
                    {"N_I": 0, "n_I": 0, "p_I": 0.0, "density_media": 0.0},
                )
                complaint_values = complaint_statistics.get(
                    (year, component),
                    {"N_R": 0, "n_R": 0, "p_R": 0.0},
                )

                interface_proportion = round(float(interface_values["p_I"]), 4)
                complaint_proportion = round(float(complaint_values["p_R"]), 4)

                rows.append(
                    {
                        "comp": component,
                        "ano": year,
                        "N_I": int(interface_values["N_I"]),
                        "n_I": int(interface_values["n_I"]),
                        "p_I": interface_proportion,
                        "density_media": round(
                            float(interface_values["density_media"]),
                            4,
                        ),
                        "N_R": int(complaint_values["N_R"]),
                        "n_R": int(complaint_values["n_R"]),
                        "p_R": complaint_proportion,
                        "gap_IR": round(
                            interface_proportion - complaint_proportion,
                            4,
                        ),
                    }
                )

        return rows


class TechnicalSummaryReportBuilder:
    """Builds a technical text summary without interpretive commentary."""

    def build_report(
        self,
        rows: list[dict],
        interface_source_name: str,
        complaint_source_name: str,
    ) -> str:
        sections = [
            self._build_header(interface_source_name, complaint_source_name),
            self._build_complete_table(rows),
            self._build_component_profile(rows),
        ]

        return "\n".join(sections)

    def _build_header(
        self,
        interface_source_name: str,
        complaint_source_name: str,
    ) -> str:
        return "\n".join(
            [
                "=" * 76,
                "ABT TECHNICAL SUMMARY",
                f"Interface source: {interface_source_name}",
                f"Complaint source: {complaint_source_name}",
                f"Components: {len(COMPONENTS)}",
                f"Years: {len(YEARS)}",
                "=" * 76,
            ]
        )

    def _build_complete_table(self, rows: list[dict]) -> str:
        header = (
            f"{'Comp':>4} | {'Year':>4} | {'N_I':>5} | {'n_I':>4} | "
            f"{'p_I':>5} | {'dens':>5} | {'N_R':>5} | {'n_R':>4} | "
            f"{'p_R':>5} | {'gap_IR':>7}"
        )
        separator = "-" * len(header)
        lines = [
            "",
            "COMPLETE SERIES",
            header,
            separator,
        ]

        current_component = None
        for row in rows:
            if row["comp"] != current_component:
                if current_component is not None:
                    lines.append(separator)
                current_component = row["comp"]

            lines.append(
                f"{row['comp']:>4} | {row['ano']:>4} | {row['N_I']:>5} | "
                f"{row['n_I']:>4} | {row['p_I']:>5.3f} | "
                f"{row['density_media']:>5.3f} | {row['N_R']:>5} | "
                f"{row['n_R']:>4} | {row['p_R']:>5.3f} | "
                f"{row['gap_IR']:>+7.3f}"
            )

        lines.append(separator)
        return "\n".join(lines)

    def _build_component_profile(self, rows: list[dict]) -> str:
        header = (
            f"{'Comp':>4} | {'Name':^12} | {'p_I_avg':>7} | {'p_I_min':>7} | "
            f"{'p_I_max':>7} | {'p_R_avg':>7} | {'p_R_min':>7} | "
            f"{'p_R_max':>7} | {'gap_avg':>7}"
        )
        lines = [
            "",
            "COMPONENT PROFILE",
            header,
            "-" * len(header),
        ]

        for component in COMPONENTS:
            component_rows = [row for row in rows if row["comp"] == component]
            interface_values = [row["p_I"] for row in component_rows]
            complaint_values = [row["p_R"] for row in component_rows]
            gap_values = [row["gap_IR"] for row in component_rows]

            lines.append(
                f"{component:>4} | {COMPONENT_NAMES[component]:^12} | "
                f"{sum(interface_values) / len(interface_values):>7.3f} | "
                f"{min(interface_values):>7.3f} | "
                f"{max(interface_values):>7.3f} | "
                f"{sum(complaint_values) / len(complaint_values):>7.3f} | "
                f"{min(complaint_values):>7.3f} | "
                f"{max(complaint_values):>7.3f} | "
                f"{sum(gap_values) / len(gap_values):>+7.3f}"
            )

        return "\n".join(lines)


class ReviewedInputResolutionService:
    """Resolves reviewed input paths from CLI arguments and auto-discovery."""

    def __init__(self, discovery_service: ReviewedFileDiscoveryService) -> None:
        self._discovery_service = discovery_service

    def resolve_input_contract(
        self,
        interface_path_argument: str | None,
        complaint_path_argument: str | None,
        output_dir: Path,
    ) -> ReviewedInputContract:
        return ReviewedInputContract(
            interface_table_path=self._resolve_single_path(
                explicit_path=interface_path_argument,
                corpus_dir=output_dir,
                prefix="ti_completa",
                label="interface",
                argument_name="--ti",
            ),
            complaint_table_path=self._resolve_single_path(
                explicit_path=complaint_path_argument,
                corpus_dir=output_dir,
                prefix="tr_completa",
                label="complaint",
                argument_name="--tr",
            ),
        )

    def _resolve_single_path(
        self,
        explicit_path: str | None,
        corpus_dir: Path,
        prefix: str,
        label: str,
        argument_name: str,
    ) -> Path:
        if explicit_path:
            return Path(explicit_path)

        discovered_path = self._discovery_service.find_latest_reviewed_file(
            corpus_dir,
            prefix,
        )

        if discovered_path is None:
            sys.exit(
                f"No reviewed {label} file found in {corpus_dir}. "
                f"Provide {argument_name} explicitly."
            )

        print(f"Auto-detected {label} file: {discovered_path.name}")
        return discovered_path


class AnalyticalBaseTableOrchestrator:
    """Coordinates the analytical base table build lifecycle."""

    def __init__(
        self,
        csv_repository: CsvRepositoryProtocol,
        text_repository: TextRepositoryProtocol,
        validation_service: InputContractValidationService,
        interface_statistics_service: InterfaceStatisticsExtractionService,
        complaint_statistics_service: ComplaintStatisticsExtractionService,
        assembly_service: AnalyticalBaseTableAssemblyService,
        report_builder: TechnicalSummaryReportBuilder,
    ) -> None:
        self._csv_repository = csv_repository
        self._text_repository = text_repository
        self._validation_service = validation_service
        self._interface_statistics_service = interface_statistics_service
        self._complaint_statistics_service = complaint_statistics_service
        self._assembly_service = assembly_service
        self._report_builder = report_builder

    def execute(
        self,
        execution_context: AnalyticalBaseTableExecutionContext,
    ) -> AnalyticalBaseTableBuildResult:
        output_contract = execution_context.output_contract
        output_contract.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading reviewed input tables...")
        interface_rows = self._csv_repository.load_rows(
            execution_context.input_contract.interface_table_path,
        )
        complaint_rows = self._csv_repository.load_rows(
            execution_context.input_contract.complaint_table_path,
        )

        print("Validating reviewed input schemas...")
        self._validation_service.validate_component_columns(interface_rows, "Interface")
        self._validation_service.validate_component_columns(complaint_rows, "Complaint")

        missing_instance_columns = self._validation_service.get_missing_instance_columns(
            interface_rows,
        )
        if missing_instance_columns:
            print(f"Missing instance columns detected: {missing_instance_columns}")

        print("Extracting component statistics...")
        interface_statistics = self._interface_statistics_service.extract_statistics(
            interface_rows,
        )
        complaint_statistics = self._complaint_statistics_service.extract_statistics(
            complaint_rows,
        )

        print("Assembling analytical base table...")
        table_rows = self._assembly_service.build_rows(
            interface_statistics=interface_statistics,
            complaint_statistics=complaint_statistics,
        )

        table_path = output_contract.output_dir / output_contract.table_file_name
        summary_path = output_contract.output_dir / output_contract.summary_file_name

        print("Saving analytical base table outputs...")
        self._csv_repository.save_rows(table_path, ABT_FIELDS, table_rows)
        self._text_repository.save_text(
            summary_path,
            self._report_builder.build_report(
                rows=table_rows,
                interface_source_name=execution_context.input_contract.interface_table_path.name,
                complaint_source_name=execution_context.input_contract.complaint_table_path.name,
            ),
        )

        return AnalyticalBaseTableBuildResult(
            table_path=table_path,
            summary_path=summary_path,
            row_count=len(table_rows),
        )


class AnalyticalBaseTableDependencyFactory:
    """Builds dependencies for the analytical base table pipeline."""

    def build_orchestrator(self) -> AnalyticalBaseTableOrchestrator:
        component_value_service = ComponentValueService()

        return AnalyticalBaseTableOrchestrator(
            csv_repository=CsvRepository(),
            text_repository=TextRepository(),
            validation_service=InputContractValidationService(),
            interface_statistics_service=InterfaceStatisticsExtractionService(
                component_value_service,
            ),
            complaint_statistics_service=ComplaintStatisticsExtractionService(
                component_value_service,
            ),
            assembly_service=AnalyticalBaseTableAssemblyService(),
            report_builder=TechnicalSummaryReportBuilder(),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the analytical base table from reviewed CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_abt.py\n"
            "  python build_abt.py --ti corpus/ti_completa_revisado_2026-05-11.csv "
            "--tr corpus/tr_completa_revisado_2026-05-11.csv\n"
            "  python build_abt.py --out corpus/\n"
        ),
    )
    parser.add_argument(
        "--ti",
        default=None,
        metavar="PATH",
        help="Reviewed interface CSV file. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--tr",
        default=None,
        metavar="PATH",
        help="Reviewed complaint CSV file. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help="Output directory.",
    )
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> AnalyticalBaseTableExecutionContext:
    output_dir = Path(args.out)
    input_resolution_service = ReviewedInputResolutionService(
        ReviewedFileDiscoveryService(),
    )

    return AnalyticalBaseTableExecutionContext(
        input_contract=input_resolution_service.resolve_input_contract(
            interface_path_argument=args.ti,
            complaint_path_argument=args.tr,
            output_dir=output_dir,
        ),
        output_contract=AnalyticalBaseTableOutputContract(
            output_dir=output_dir,
        ),
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = AnalyticalBaseTableDependencyFactory().build_orchestrator()
    result = orchestrator.execute(execution_context)

    print("Execution completed.")
    print(f"CSV output: {result.table_path}")
    print(f"Text output: {result.summary_path}")
    print(f"Rows: {result.row_count}")


if __name__ == "__main__":
    main()
