#!/usr/bin/env python3
"""
Name: corpus_normalization_pipeline
Input: image slice directories and MHTML directories
Output: corpus_kabum.jsonl, corpus_ra.jsonl, ra_sem_data.jsonl
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import quopri
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol


DEFAULT_WORKSPACE_ROOT = Path(os.path.expanduser("~/tese"))
DEFAULT_INTERFACE_CAPTURE_DIR = DEFAULT_WORKSPACE_ROOT / "image_capture" / "wayback_captures"
DEFAULT_COMPLAINT_CAPTURE_DIR = DEFAULT_WORKSPACE_ROOT / "ra_capture"
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE_ROOT / "corpus"
DEFAULT_ENCODING = "utf-8"

PAGE_TYPES = ("campanha", "principal", "produto")
INTERFACE_YEARS = set(range(2020, 2026))
COMPLAINT_YEARS = set(range(2020, 2026))
WINDOW_BOUNDARY_DAYS = 3

REFERENCE_DATES = {
    2020: datetime(2020, 11, 27),
    2021: datetime(2021, 11, 26),
    2022: datetime(2022, 11, 25),
    2023: datetime(2023, 11, 24),
    2024: datetime(2024, 11, 29),
    2025: datetime(2025, 11, 28),
}

CREATED_DATE_PATTERN = re.compile(
    r'"created"\s*:\s*\[\s*0\s*,\s*"(\d{4}-\d{2}-\d{2})',
    re.DOTALL,
)
DATE_FALLBACK_PATTERNS = (
    re.compile(r'"createdAt"\s*:\s*"(\d{4}-\d{2}-\d{2})', re.DOTALL),
    re.compile(r'"openedAt"\s*:\s*"(\d{4}-\d{2}-\d{2})', re.DOTALL),
    re.compile(r'"date"\s*:\s*"(\d{4}-\d{2}-\d{2})', re.DOTALL),
    re.compile(r'datetime="(\d{4}-\d{2}-\d{2})', re.DOTALL),
    re.compile(r'published_time["\s]+content="(\d{4}-\d{2}-\d{2})', re.DOTALL),
)
DESCRIPTION_PATTERN = re.compile(
    r'"description"\s*:\s*\[\s*0\s*,\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
TITLE_FIELD_PATTERN = re.compile(
    r'"title"\s*:\s*\[\s*0\s*,\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
TITLE_TAG_PATTERN = re.compile(
    r"<title[^>]*>(.*?)</title>",
    re.IGNORECASE | re.DOTALL,
)
TIMESTAMP_PATTERN = re.compile(r"(\d{14})")


class CorpusNormalizationError(Exception):
    """Base exception for corpus normalization failures."""


@dataclass(frozen=True)
class CorpusInputContract:
    interface_capture_dir: Path
    complaint_capture_dir: Path


@dataclass(frozen=True)
class CorpusOutputContract:
    output_dir: Path
    interface_corpus_file_name: str = "corpus_kabum.jsonl"
    complaint_corpus_file_name: str = "corpus_ra.jsonl"
    undated_complaint_file_name: str = "ra_sem_data.jsonl"
    encoding: str = DEFAULT_ENCODING


@dataclass(frozen=True)
class CorpusNormalizationExecutionContext:
    input_contract: CorpusInputContract
    output_contract: CorpusOutputContract


@dataclass(frozen=True)
class CorpusNormalizationResult:
    interface_corpus_path: Path
    complaint_corpus_path: Path
    undated_complaint_path: Path | None
    interface_record_count: int
    complaint_record_count: int
    undated_complaint_record_count: int


@dataclass(frozen=True)
class TextExtractionResult:
    title: str
    text: str


class JsonLinesRepositoryProtocol(Protocol):
    def save_records(self, output_file_path: Path, records: Iterable[dict]) -> None:
        """Save records using the JSON Lines format."""


class JsonLinesRepository:
    """Persists records using the JSON Lines format."""

    def save_records(self, output_file_path: Path, records: Iterable[dict]) -> None:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        with output_file_path.open("w", encoding=DEFAULT_ENCODING) as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


class TimeWindowClassificationService:
    """Classifies a datetime into the configured event window."""

    def classify_window(self, target_date: datetime, reference_year: int) -> str:
        reference_date = REFERENCE_DATES.get(reference_year)

        if reference_date is None:
            return "desconhecida"

        day_delta = (target_date - reference_date).days

        if day_delta < -WINDOW_BOUNDARY_DAYS:
            return "pre"

        if day_delta > WINDOW_BOUNDARY_DAYS:
            return "pos"

        return "bf"


class FileHashService:
    """Computes file hashes used for input deduplication."""

    def calculate_md5(self, file_path: Path) -> str:
        hash_builder = hashlib.md5()

        with file_path.open("rb") as input_file:
            hash_builder.update(input_file.read(65536))

        return hash_builder.hexdigest()


class MhtmlDecoder:
    """Decodes MHTML content into text."""

    def decode(self, file_path: Path) -> str:
        raw_content = file_path.read_bytes()

        try:
            decoded_content = quopri.decodestring(raw_content).decode(
                DEFAULT_ENCODING,
                errors="replace",
            )
        except Exception:
            decoded_content = raw_content.decode(DEFAULT_ENCODING, errors="replace")

        return html.unescape(decoded_content)


class ComplaintDateExtractionService:
    """Extracts complaint dates from decoded MHTML content."""

    def extract_date(self, decoded_content: str) -> datetime | None:
        primary_match = CREATED_DATE_PATTERN.search(decoded_content)

        if primary_match is not None:
            parsed_date = self._parse_date_string(primary_match.group(1))
            if parsed_date is not None:
                return parsed_date

        for fallback_pattern in DATE_FALLBACK_PATTERNS:
            fallback_match = fallback_pattern.search(decoded_content)

            if fallback_match is None:
                continue

            parsed_date = self._parse_date_string(fallback_match.group(1))
            if parsed_date is not None:
                return parsed_date

        return None

    def _parse_date_string(self, date_value: str) -> datetime | None:
        try:
            return datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError:
            return None


class ComplaintTextExtractionService:
    """Extracts title and text from decoded MHTML content."""

    def extract_text(self, decoded_content: str) -> TextExtractionResult:
        return TextExtractionResult(
            title=self._extract_title(decoded_content),
            text=self._extract_description(decoded_content),
        )

    def _extract_title(self, decoded_content: str) -> str:
        field_match = TITLE_FIELD_PATTERN.search(decoded_content)

        if field_match is not None:
            try:
                return json.loads(f'"{field_match.group(1)}"').strip()
            except Exception:
                return field_match.group(1).strip()

        tag_match = TITLE_TAG_PATTERN.search(decoded_content)
        if tag_match is not None:
            return tag_match.group(1).strip()

        return ""

    def _extract_description(self, decoded_content: str) -> str:
        description_match = DESCRIPTION_PATTERN.search(decoded_content)

        if description_match is None:
            return ""

        try:
            return json.loads(f'"{description_match.group(1)}"').strip()
        except Exception:
            return description_match.group(1).strip()


class InterfaceCorpusBuilder:
    """Builds interface corpus records from image slice files."""

    def __init__(
        self,
        file_hash_service: FileHashService,
        window_classifier: TimeWindowClassificationService,
    ) -> None:
        self._file_hash_service = file_hash_service
        self._window_classifier = window_classifier

    def build_records(self, base_dir: Path) -> list[dict]:
        records: list[dict] = []
        seen_hashes: set[str] = set()
        sequence_counter: dict[tuple[int, str], int] = {}
        skipped_2019_count = 0

        for year_dir in sorted(base_dir.iterdir()):
            if not year_dir.is_dir():
                continue

            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            if year == 2019:
                skipped_2019_count += 1
                continue

            if year not in INTERFACE_YEARS:
                continue

            for page_type in PAGE_TYPES:
                slice_dir = year_dir / page_type / "slices"

                if not slice_dir.exists():
                    continue

                for image_file_path in sorted(slice_dir.glob("*.png")):
                    file_hash = self._file_hash_service.calculate_md5(image_file_path)

                    if file_hash in seen_hashes:
                        continue

                    seen_hashes.add(file_hash)
                    records.append(
                        self._build_record(
                            image_file_path=image_file_path,
                            year=year,
                            page_type=page_type,
                            file_hash=file_hash,
                            sequence_counter=sequence_counter,
                        )
                    )

        if skipped_2019_count:
            print(f"Interface directories skipped for year 2019: {skipped_2019_count}")

        return records

    def _build_record(
        self,
        image_file_path: Path,
        year: int,
        page_type: str,
        file_hash: str,
        sequence_counter: dict[tuple[int, str], int],
    ) -> dict:
        timestamp = self._extract_timestamp(image_file_path)
        parsed_timestamp = self._parse_timestamp(timestamp)
        window = (
            self._window_classifier.classify_window(parsed_timestamp, year)
            if parsed_timestamp is not None
            else "desconhecida"
        )

        sequence_key = (year, page_type)
        sequence_counter[sequence_key] = sequence_counter.get(sequence_key, 0) + 1

        return {
            "id": f"KBW_{year}_{page_type[:3]}_{sequence_counter[sequence_key]:04d}",
            "ano": year,
            "tipo_pagina": page_type,
            "janela": window,
            "session_ts": timestamp or "",
            "filepath": str(image_file_path),
            "md5": file_hash,
        }

    def _extract_timestamp(self, image_file_path: Path) -> str | None:
        timestamp_match = TIMESTAMP_PATTERN.search(image_file_path.name)
        return timestamp_match.group(1) if timestamp_match else None

    def _parse_timestamp(self, timestamp: str | None) -> datetime | None:
        if not timestamp:
            return None

        try:
            return datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S")
        except ValueError:
            return None


class ComplaintCorpusBuilder:
    """Builds complaint corpus records from MHTML files."""

    def __init__(
        self,
        mhtml_decoder: MhtmlDecoder,
        date_extractor: ComplaintDateExtractionService,
        text_extractor: ComplaintTextExtractionService,
        window_classifier: TimeWindowClassificationService,
    ) -> None:
        self._mhtml_decoder = mhtml_decoder
        self._date_extractor = date_extractor
        self._text_extractor = text_extractor
        self._window_classifier = window_classifier

    def build_records(self, base_dir: Path) -> tuple[list[dict], list[dict]]:
        records: list[dict] = []
        undated_records: list[dict] = []
        ignored_file_count = 0
        seen_text_hashes: set[str] = set()
        sequence_counter: dict[tuple[str, str], int] = {}

        mhtml_file_paths = sorted(base_dir.rglob("*.mhtml"))
        print(f"MHTML files found: {len(mhtml_file_paths)}")

        for mhtml_file_path in mhtml_file_paths:
            try:
                record = self._build_record(
                    mhtml_file_path=mhtml_file_path,
                    seen_text_hashes=seen_text_hashes,
                    sequence_counter=sequence_counter,
                )
            except Exception as error:
                print(f"Unable to decode MHTML file: {mhtml_file_path.name}: {error}")
                continue

            if record is None:
                ignored_file_count += 1
                continue

            if record["complaint_date"] is None:
                undated_records.append(record)
            else:
                records.append(record)

        if ignored_file_count:
            print(f"Files ignored outside configured years: {ignored_file_count}")

        return records, undated_records

    def _build_record(
        self,
        mhtml_file_path: Path,
        seen_text_hashes: set[str],
        sequence_counter: dict[tuple[str, str], int],
    ) -> dict | None:
        decoded_content = self._mhtml_decoder.decode(mhtml_file_path)
        text_result = self._text_extractor.extract_text(decoded_content)
        extracted_date = self._date_extractor.extract_date(decoded_content)

        if text_result.text:
            text_hash = hashlib.md5(text_result.text.encode()).hexdigest()
            if text_hash in seen_text_hashes:
                return None
            seen_text_hashes.add(text_hash)

        year, normalized_date = self._normalize_year_and_date(extracted_date)

        if year is not None and year not in COMPLAINT_YEARS:
            return None

        window = (
            self._window_classifier.classify_window(normalized_date, year)
            if normalized_date is not None and year is not None
            else "desconhecida"
        )

        year_string = str(year) if year else "0000"
        sequence_key = (year_string, window)
        sequence_counter[sequence_key] = sequence_counter.get(sequence_key, 0) + 1
        record_id = f"RA_{year_string}_{window[:3]}_{sequence_counter[sequence_key]:04d}"

        return {
            "id": record_id,
            "ano": year,
            "janela": window,
            "complaint_date": (
                normalized_date.strftime("%Y-%m-%d") if normalized_date else None
            ),
            "filename": mhtml_file_path.name,
            "filepath": str(mhtml_file_path),
            "title": text_result.title,
            "text": text_result.text,
        }

    def _normalize_year_and_date(
        self,
        extracted_date: datetime | None,
    ) -> tuple[int | None, datetime | None]:
        if extracted_date is None:
            return None, None

        year = extracted_date.year

        if year == 2019:
            return 2020, extracted_date.replace(year=2020)

        return year, extracted_date


class CorpusSortingService:
    """Applies output sorting required by the corpus contract."""

    def sort_complaint_records(self, records: list[dict]) -> list[dict]:
        return sorted(
            records,
            key=lambda record: (
                record["ano"] or 0,
                record["janela"],
                record["filename"],
            ),
        )


class CorpusNormalizationOrchestrator:
    """Coordinates the corpus normalization lifecycle."""

    def __init__(
        self,
        interface_corpus_builder: InterfaceCorpusBuilder,
        complaint_corpus_builder: ComplaintCorpusBuilder,
        sorting_service: CorpusSortingService,
        repository: JsonLinesRepositoryProtocol,
    ) -> None:
        self._interface_corpus_builder = interface_corpus_builder
        self._complaint_corpus_builder = complaint_corpus_builder
        self._sorting_service = sorting_service
        self._repository = repository

    def execute(
        self,
        execution_context: CorpusNormalizationExecutionContext,
    ) -> CorpusNormalizationResult:
        output_contract = execution_context.output_contract
        output_contract.output_dir.mkdir(parents=True, exist_ok=True)

        print("Building interface corpus records...")
        interface_records = self._interface_corpus_builder.build_records(
            execution_context.input_contract.interface_capture_dir,
        )

        interface_corpus_path = (
            output_contract.output_dir / output_contract.interface_corpus_file_name
        )
        self._repository.save_records(interface_corpus_path, interface_records)

        print("Building complaint corpus records...")
        complaint_records, undated_complaint_records = (
            self._complaint_corpus_builder.build_records(
                execution_context.input_contract.complaint_capture_dir,
            )
        )
        all_complaint_records = self._sorting_service.sort_complaint_records(
            complaint_records + undated_complaint_records,
        )

        complaint_corpus_path = (
            output_contract.output_dir / output_contract.complaint_corpus_file_name
        )
        self._repository.save_records(complaint_corpus_path, all_complaint_records)

        undated_complaint_path: Path | None = None
        if undated_complaint_records:
            undated_complaint_path = (
                output_contract.output_dir
                / output_contract.undated_complaint_file_name
            )
            self._repository.save_records(
                undated_complaint_path,
                undated_complaint_records,
            )

        return CorpusNormalizationResult(
            interface_corpus_path=interface_corpus_path,
            complaint_corpus_path=complaint_corpus_path,
            undated_complaint_path=undated_complaint_path,
            interface_record_count=len(interface_records),
            complaint_record_count=len(all_complaint_records),
            undated_complaint_record_count=len(undated_complaint_records),
        )


class CorpusNormalizationDependencyFactory:
    """Builds dependencies for the corpus normalization pipeline."""

    def build_orchestrator(self) -> CorpusNormalizationOrchestrator:
        window_classifier = TimeWindowClassificationService()

        return CorpusNormalizationOrchestrator(
            interface_corpus_builder=InterfaceCorpusBuilder(
                file_hash_service=FileHashService(),
                window_classifier=window_classifier,
            ),
            complaint_corpus_builder=ComplaintCorpusBuilder(
                mhtml_decoder=MhtmlDecoder(),
                date_extractor=ComplaintDateExtractionService(),
                text_extractor=ComplaintTextExtractionService(),
                window_classifier=window_classifier,
            ),
            sorting_service=CorpusSortingService(),
            repository=JsonLinesRepository(),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize corpus input files into JSONL output contracts.",
    )
    parser.add_argument("--kabum-dir", type=Path, default=DEFAULT_INTERFACE_CAPTURE_DIR)
    parser.add_argument("--ra-dir", type=Path, default=DEFAULT_COMPLAINT_CAPTURE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_execution_context(args: argparse.Namespace) -> CorpusNormalizationExecutionContext:
    return CorpusNormalizationExecutionContext(
        input_contract=CorpusInputContract(
            interface_capture_dir=args.kabum_dir,
            complaint_capture_dir=args.ra_dir,
        ),
        output_contract=CorpusOutputContract(
            output_dir=args.out_dir,
        ),
    )


def main() -> None:
    args = parse_arguments()
    execution_context = build_execution_context(args)
    orchestrator = CorpusNormalizationDependencyFactory().build_orchestrator()
    result = orchestrator.execute(execution_context)

    print("Execution completed.")
    print(f"Interface corpus: {result.interface_corpus_path}")
    print(f"Complaint corpus: {result.complaint_corpus_path}")

    if result.undated_complaint_path is not None:
        print(f"Undated complaint records: {result.undated_complaint_path}")

    print(
        "Records: "
        f"interface={result.interface_record_count}, "
        f"complaint={result.complaint_record_count}, "
        f"undated_complaint={result.undated_complaint_record_count}"
    )


if __name__ == "__main__":
    main()
