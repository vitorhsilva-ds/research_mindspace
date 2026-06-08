#!/usr/bin/env python3
"""
Name: interface_table_builder
Input: JSON result files, optional corpus JSONL file
Output: ti_completa.csv, ti_revisao.csv, ti_summary.json
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
DEFAULT_RESULT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus" / "vlm_output"
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus"
DEFAULT_ENCODING = "utf-8"
VISUAL_RESULT_PATTERN = "KBW_*.json"

COMPONENTS = ("M", "I", "N", "D", "S", "P", "A", "C", "E")

LOCUS_BY_PAGE_TYPE = {
    "campanha": "topo_campanha",
    "principal": "pagina_principal",
    "produto": "card_produto",
}

INTERFACE_TABLE_FIELDS = (
    ["id", "ano", "tipo_pagina", "janela", "session_ts", "locus", "filepath"]
    + list(COMPONENTS)
    + [f"{component}_inst" for component in COMPONENTS]
    + [f"{component}_ev" for component in COMPONENTS]
    + [f"{component}_loc" for component in COMPONENTS]
    + ["needs_review", "review_reasons", "parse_ok"]
)


class InterfaceTableBuildError(Exception):
    """Base exception for interface table build failures."""


class JsonRecordLoadError(InterfaceTableBuildError):
    """Raised when a JSON input record cannot be loaded."""


@dataclass(frozen=True)
class InterfaceInputContract:
    result_dir: Path
    corpus_file_path: Path
    result_pattern: str = VISUAL_RESULT_PATTERN


@dataclass(frozen=True)
class InterfaceOutputContract:
    output_dir: Path
    full_table_file_name: str = "ti_completa.csv"
    review_table_file_name: str = "ti_revisao.csv"
    summary_file_name: str = "ti_summary.json"
    encoding: str = DEFAULT_ENCODING


@dataclass(frozen=True)
class InterfaceTableExecutionContext:
    input_contract: InterfaceInputContract
    output_contract: InterfaceOutputContract


@dataclass(frozen=True)
class InterfaceTableBuildResult:
    full_table_path: Path
    review_table_path: Path
    summary_path: Path
    total_rows: int
    review_rows: int
    parse_error_rows: int


class JsonResultRepositoryProtocol(Protocol):
    def load_result_records(self, input_contract: InterfaceInputContract) -> list[dict]:
        """Load JSON result records from the configured input contract."""


class CorpusTimestampRepositoryProtocol(Protocol):
    def load_timestamp_mapping(self, corpus_file_path: Path) -> dict[str, str]:
        """Load a record identifier to timestamp mapping."""


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


class JsonResultRepository:
    """Loads JSON result records from a directory."""

    def load_result_records(self, input_contract: InterfaceInputContract) -> list[dict]:
        records: list[dict] = []

        for json_file_path in sorted(input_contract.result_dir.glob(input_contract.result_pattern)):
            try:
                records.append(
                    json.loads(json_file_path.read_text(encoding=DEFAULT_ENCODING))
                )
            except Exception as error:
                raise JsonRecordLoadError(
                    f"Unable to load JSON record: {json_file_path.name}"
                ) from error

        return records


class CorpusTimestampRepository:
    """Loads timestamp values from the optional corpus file."""

    def load_timestamp_mapping(self, corpus_file_path: Path) -> dict[str, str]:
        timestamp_by_record_id: dict[str, str] = {}

        if not corpus_file_path.exists():
            return timestamp_by_record_id

        with corpus_file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            for line in input_file:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                record = json.loads(stripped_line)
                timestamp_by_record_id[record["id"]] = record.get("session_ts", "")

        return timestamp_by_record_id


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


class ComponentFieldExtractionService:
    """Extracts component fields from a visual model result record."""

    def extract_component_fields(self, result_record: dict) -> dict:
        extracted_fields: dict[str, object] = {}
        component_payload = result_record.get("mindspace", {})

        for component in COMPONENTS:
            component_data = component_payload.get(component, {})

            if isinstance(component_data, dict):
                is_present = bool(component_data.get("presente", False))
                instance_count = int(
                    component_data.get("instancias", 1 if is_present else 0)
                )
                evidence_items = component_data.get("evidencias", [])
                screen_location = component_data.get("localizacao", "")
            else:
                is_present = bool(component_data)
                instance_count = 1 if is_present else 0
                evidence_items = []
                screen_location = ""

            if is_present and instance_count == 0:
                instance_count = 1

            if not is_present:
                instance_count = 0

            extracted_fields[component] = 1 if is_present else 0
            extracted_fields[f"{component}_inst"] = instance_count
            extracted_fields[f"{component}_ev"] = (
                " | ".join(str(evidence) for evidence in evidence_items)
                if evidence_items
                else ""
            )
            extracted_fields[f"{component}_loc"] = str(screen_location)

        return extracted_fields


class ReviewFlagPolicy:
    """Builds review flags for generated interface rows."""

    def get_review_flags(self, row: dict) -> list[str]:
        review_flags: list[str] = []

        if row.get("S"):
            review_flags.append("S=1: confirmar sinal fechado (5 tipos definidos)")

        if row.get("C") and row.get("tipo_pagina") in ("principal", "campanha"):
            review_flags.append("C=1 em pag principal/campanha: verificar nao e nav")

        if not row.get("D") and row.get("tipo_pagina") == "principal":
            review_flags.append("D=0 em pag principal: verificar formulario opt-in via MHTML")

        return review_flags


class InterfaceRowFactory:
    """Builds interface table rows from input result records."""

    def __init__(
        self,
        component_extractor: ComponentFieldExtractionService,
        review_flag_policy: ReviewFlagPolicy,
    ) -> None:
        self._component_extractor = component_extractor
        self._review_flag_policy = review_flag_policy

    def build_row(self, result_record: dict, timestamp_by_record_id: dict[str, str]) -> dict:
        record_id = result_record.get("id", "")
        page_type = result_record.get("tipo_pagina", "")
        parse_ok = "_parse_error" not in result_record and "_error" not in result_record

        row: dict[str, object] = {
            "id": record_id,
            "ano": result_record.get("ano"),
            "tipo_pagina": page_type,
            "janela": result_record.get("janela", ""),
            "session_ts": timestamp_by_record_id.get(record_id, ""),
            "locus": LOCUS_BY_PAGE_TYPE.get(page_type, page_type),
            "filepath": result_record.get("filepath", ""),
            "parse_ok": 1 if parse_ok else 0,
        }

        if parse_ok:
            row.update(self._component_extractor.extract_component_fields(result_record))
        else:
            row.update(self._build_empty_component_fields())

        review_flags = self._review_flag_policy.get_review_flags(row)
        row["needs_review"] = 1 if review_flags else 0
        row["review_reasons"] = " | ".join(review_flags)

        return row

    def _build_empty_component_fields(self) -> dict:
        empty_fields: dict[str, object] = {}

        for component in COMPONENTS:
            empty_fields[component] = 0
            empty_fields[f"{component}_inst"] = 0
            empty_fields[f"{component}_ev"] = ""
            empty_fields[f"{component}_loc"] = ""

        return empty_fields


class InterfaceSummaryComputationService:
    """Computes the summary payload required by downstream consumers."""

    def compute_summary(self, rows: list[dict]) -> dict:
        rows_by_year: dict[int, list[dict]] = {}

        for row in rows:
            if row.get("parse_ok") and row.get("ano"):
                year = int(row["ano"])
                rows_by_year.setdefault(year, []).append(row)

        summary: dict[int, dict] = {}

        for year, year_rows in sorted(rows_by_year.items()):
            total_interface_rows = len(year_rows)
            summary[year] = {
                "N_I": total_interface_rows,
                "components": {},
            }

            for component in COMPONENTS:
                present_rows = [
                    row for row in year_rows if int(row.get(component, 0)) == 1
                ]
                present_count = len(present_rows)
                proportion = (
                    round(present_count / total_interface_rows, 4)
                    if total_interface_rows > 0
                    else 0.0
                )
                instance_values = [
                    int(row.get(f"{component}_inst", 1)) for row in present_rows
                ]
                average_density = (
                    round(sum(instance_values) / len(instance_values), 2)
                    if instance_values
                    else 0.0
                )

                summary[year]["components"][component] = {
                    "n_I": present_count,
                    "p_I": proportion,
                    "density_media": average_density,
                }

        return summary


class InterfaceTableOrchestrator:
    """Coordinates the interface table build lifecycle."""

    def __init__(
        self,
        result_repository: JsonResultRepositoryProtocol,
        timestamp_repository: CorpusTimestampRepositoryProtocol,
        persistence_service: TablePersistenceProtocol,
        row_factory: InterfaceRowFactory,
        summary_service: InterfaceSummaryComputationService,
    ) -> None:
        self._result_repository = result_repository
        self._timestamp_repository = timestamp_repository
        self._persistence_service = persistence_service
        self._row_factory = row_factory
        self._summary_service = summary_service

    def execute(
        self,
        execution_context: InterfaceTableExecutionContext,
    ) -> InterfaceTableBuildResult | None:
        output_contract = execution_context.output_contract
        output_contract.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading JSON result records...")
        result_records = self._result_repository.load_result_records(
            execution_context.input_contract,
        )
        print(f"JSON result records loaded: {len(result_records)}")

        if not result_records:
            print("No JSON result records found.")
            return None

        timestamp_by_record_id = self._timestamp_repository.load_timestamp_mapping(
            execution_context.input_contract.corpus_file_path,
        )

        print("Building interface rows...")
        interface_rows = [
            self._row_factory.build_row(result_record, timestamp_by_record_id)
            for result_record in result_records
        ]

        review_rows = [row for row in interface_rows if row["needs_review"]]
        parse_error_rows = sum(1 for row in interface_rows if not row["parse_ok"])

        full_table_path = output_contract.output_dir / output_contract.full_table_file_name
        review_table_path = output_contract.output_dir / output_contract.review_table_file_name
        summary_path = output_contract.output_dir / output_contract.summary_file_name

        print("Saving interface table outputs...")
        self._persistence_service.save_csv(
            full_table_path,
            INTERFACE_TABLE_FIELDS,
            interface_rows,
        )
        self._persistence_service.save_csv(
            review_table_path,
            INTERFACE_TABLE_FIELDS,
            review_rows,
        )
        self._persistence_service.save_json(
            summary_path,
            self._summary_service.compute_summary(interface_rows),
        )

        return InterfaceTableBuildResult(
            full_table_path=full_table_path,
            review_table_path=review_table_path,
            summary_path=summary_path,
            total_rows=len(interface_rows),
            review_rows=len(review_rows),
            parse_error_rows=parse_error_rows,
        )


class InterfaceTableDependencyFactory:
    """Builds dependencies for the interface table pipeline."""

    def build_orchestrator(self) -> InterfaceTableOrchestrator:
        return InterfaceTableOrchestrator(
            result_repository=JsonResultRepository(),
            timestamp_repository=CorpusTimestampRepository(),
            persistence_service=FileTablePersistenceService(),
            row_factory=InterfaceRowFactory(
                component_extractor=ComponentFieldExtractionService(),
                review_flag_policy=ReviewFlagPolicy(),
            ),
            summary_service=InterfaceSummaryComputationService(),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the interface table output files.",
    )
    parser.add_argument("--vlm-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "corpus_kabum.jsonl",
        help="Optional corpus JSONL file used to enrich session_ts.",
    )
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> InterfaceTableExecutionContext:
    return InterfaceTableExecutionContext(
        input_contract=InterfaceInputContract(
            result_dir=args.vlm_dir,
            corpus_file_path=args.corpus,
        ),
        output_contract=InterfaceOutputContract(
            output_dir=args.out_dir,
        ),
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = InterfaceTableDependencyFactory().build_orchestrator()
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
        f"review={result.review_rows}, "
        f"parse_errors={result.parse_error_rows}"
    )


if __name__ == "__main__":
    main()
