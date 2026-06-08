#!/usr/bin/env python3
"""
Name: text_review_service
Input: reviewed text CSV file and related MHTML files
Output: editable working CSV file, extracted text, and JSON API responses
Usage: run as a Flask application
"""

from __future__ import annotations

import csv
import email
import html as html_module
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Protocol

from flask import Flask, abort, jsonify, make_response, request, send_from_directory


DEFAULT_ENCODING = "utf-8"
DEFAULT_FALLBACK_ENCODING = "latin-1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5050
DEFAULT_STATIC_DIRECTORY = "static"
DEFAULT_CORPUS_DIRECTORY = Path(
    os.environ.get(
        "REVIEW_CORPUS_DIR",
        str(Path.home() / "workspace" / "corpus"),
    )
)
DEFAULT_SOURCE_CAPTURE_DIRECTORY = Path(
    os.environ.get(
        "REVIEW_SOURCE_CAPTURE_DIR",
        str(Path.home() / "workspace" / "capture"),
    )
)
DEFAULT_ORIGINAL_FILE_NAME = "tr_revisao.csv"
WORKING_FILE_NAME_TEMPLATE = "tr_revisao_{date}.csv"

MINDSPACE_COLUMNS = ("M", "I", "N", "D", "S", "P", "A", "C", "E")


class TextReviewServiceError(Exception):
    """Base exception for text review service failures."""


class InvalidRowIndexError(TextReviewServiceError):
    """Raised when a row index is outside the editable range."""


class SourceRowUnavailableError(TextReviewServiceError):
    """Raised when a source row cannot be resolved."""


class TextSourceUnavailableError(TextReviewServiceError):
    """Raised when a text source file cannot be resolved."""


@dataclass(frozen=True)
class ReviewFileContract:
    corpus_dir: Path
    source_capture_dir: Path
    original_file_name: str
    working_file_name: str

    @property
    def original_file_path(self) -> Path:
        return self.corpus_dir / self.original_file_name

    @property
    def working_file_path(self) -> Path:
        return self.corpus_dir / self.working_file_name


@dataclass(frozen=True)
class ReviewServiceConfig:
    base_dir: Path
    static_dir: Path
    file_contract: ReviewFileContract
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


@dataclass(frozen=True)
class CsvTable:
    fieldnames: list[str]
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class ServiceStatus:
    working_csv: str
    original_csv: str
    has_working: bool
    has_original: bool
    today: str


@dataclass(frozen=True)
class TextExtractionResult:
    text: str
    found: bool
    filename: str | None = None


class CsvRepositoryProtocol(Protocol):
    def load_table(self, file_path: Path) -> CsvTable:
        """Load a CSV table."""

    def save_table(
        self,
        file_path: Path,
        fieldnames: Iterable[str],
        rows: Iterable[dict[str, str]],
    ) -> None:
        """Save a CSV table."""


class CsvRepository:
    """Loads and saves CSV files."""

    def load_table(self, file_path: Path) -> CsvTable:
        if not file_path.exists():
            return CsvTable(fieldnames=[], rows=[])

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
            writer = csv.DictWriter(output_file, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)


class WorkingFileBootstrapService:
    """Creates the editable working CSV file when needed."""

    def ensure_working_file(self, file_contract: ReviewFileContract) -> bool:
        file_contract.corpus_dir.mkdir(parents=True, exist_ok=True)

        if file_contract.working_file_path.exists():
            return True

        if file_contract.original_file_path.exists():
            shutil.copy2(
                file_contract.original_file_path,
                file_contract.working_file_path,
            )
            return True

        return False


class ReviewStatusService:
    """Builds status information for the review interface."""

    def __init__(self, bootstrap_service: WorkingFileBootstrapService) -> None:
        self._bootstrap_service = bootstrap_service

    def get_status(self, file_contract: ReviewFileContract, current_date: str) -> ServiceStatus:
        has_working = self._bootstrap_service.ensure_working_file(file_contract)

        return ServiceStatus(
            working_csv=str(file_contract.working_file_path),
            original_csv=str(file_contract.original_file_path),
            has_working=has_working,
            has_original=file_contract.original_file_path.exists(),
            today=current_date,
        )


class ReviewRowsQueryService:
    """Builds editable row payloads from working and original tables."""

    def __init__(
        self,
        bootstrap_service: WorkingFileBootstrapService,
        csv_repository: CsvRepositoryProtocol,
    ) -> None:
        self._bootstrap_service = bootstrap_service
        self._csv_repository = csv_repository

    def get_rows(self, file_contract: ReviewFileContract) -> dict:
        self._bootstrap_service.ensure_working_file(file_contract)
        working_table = self._csv_repository.load_table(file_contract.working_file_path)
        original_table = self._csv_repository.load_table(file_contract.original_file_path)
        original_rows_by_identifier = {
            row["id"]: row
            for row in original_table.rows
            if "id" in row
        }
        payload_rows: list[dict] = []

        for index, row in enumerate(working_table.rows):
            payload_row = dict(row)
            payload_row["_index"] = index
            payload_row["_original"] = original_rows_by_identifier.get(
                payload_row.get("id", ""),
                {},
            )
            payload_rows.append(payload_row)

        return {
            "rows": payload_rows,
            "fieldnames": working_table.fieldnames,
        }


class HtmlTextNormalizationService:
    """Converts HTML payloads into readable plain text."""

    def convert_html_to_text(self, html_text: str) -> str:
        cleaned_text = re.sub(
            r"<script[^>]*>.*?</script>",
            " ",
            html_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned_text = re.sub(
            r"<style[^>]*>.*?</style>",
            " ",
            cleaned_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned_text = re.sub(
            r"<(br|p|div|h[1-6]|li|tr)[^>]*>",
            "\n",
            cleaned_text,
            flags=re.IGNORECASE,
        )
        cleaned_text = re.sub(r"<[^>]+>", "", cleaned_text)
        cleaned_text = html_module.unescape(cleaned_text)
        lines = [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in cleaned_text.splitlines()
        ]
        lines = [line for line in lines if line]

        deduplicated_lines: list[str] = []

        for line in lines:
            if not deduplicated_lines or deduplicated_lines[-1] != line:
                deduplicated_lines.append(line)

        return "\n".join(deduplicated_lines)


class MessageContentExtractionService:
    """Extracts readable text from MHTML files."""

    def __init__(self, html_text_service: HtmlTextNormalizationService) -> None:
        self._html_text_service = html_text_service

    def extract_text_from_mhtml(self, file_path: Path) -> str:
        try:
            raw_payload = file_path.read_bytes()
            message = email.message_from_bytes(raw_payload)
            html_content: str | None = None
            plain_content: str | None = None

            def walk(part) -> None:
                nonlocal html_content, plain_content
                content_type = part.get_content_type()

                if content_type == "text/html" and html_content is None:
                    payload = part.get_payload(decode=True)

                    if payload:
                        html_content = self._decode_payload(part, payload)

                elif content_type == "text/plain" and plain_content is None:
                    payload = part.get_payload(decode=True)

                    if payload:
                        plain_content = self._decode_payload(part, payload)

                if part.is_multipart():
                    for subpart in part.get_payload():
                        walk(subpart)

            walk(message)

            if html_content:
                return self._html_text_service.convert_html_to_text(html_content)

            if plain_content:
                return plain_content.strip()

            return "[Content extraction unavailable.]"

        except Exception as error:
            return f"[File read error: {error}]"

    def _decode_payload(self, part, payload: bytes) -> str:
        charset = part.get_content_charset() or DEFAULT_ENCODING

        try:
            return payload.decode(charset, errors="replace")
        except Exception:
            return payload.decode(DEFAULT_FALLBACK_ENCODING, errors="replace")


class TextSourceResolutionService:
    """Resolves source text files for editable rows."""

    def __init__(
        self,
        bootstrap_service: WorkingFileBootstrapService,
        csv_repository: CsvRepositoryProtocol,
        content_service: MessageContentExtractionService,
    ) -> None:
        self._bootstrap_service = bootstrap_service
        self._csv_repository = csv_repository
        self._content_service = content_service

    def get_text(self, file_contract: ReviewFileContract, index: int) -> TextExtractionResult:
        self._bootstrap_service.ensure_working_file(file_contract)
        working_table = self._csv_repository.load_table(file_contract.working_file_path)

        if index < 0 or index >= len(working_table.rows):
            raise InvalidRowIndexError("Invalid row index.")

        row = working_table.rows[index]
        filename = row.get("filename", "")

        if not filename:
            return TextExtractionResult(
                text="[Filename not found in row.]",
                found=False,
                filename=None,
            )

        source_file_path = file_contract.source_capture_dir / filename

        if not source_file_path.exists():
            alternative_path = file_contract.source_capture_dir / Path(filename).name

            if alternative_path.exists():
                source_file_path = alternative_path
            else:
                return TextExtractionResult(
                    text=(
                        f"[File not found: {filename}]\n\n"
                        f"Expected path: {source_file_path}"
                    ),
                    found=False,
                    filename=filename,
                )

        return TextExtractionResult(
            text=self._content_service.extract_text_from_mhtml(source_file_path),
            found=True,
            filename=filename,
        )


class RowUpdateService:
    """Persists row updates in the working CSV file."""

    def __init__(
        self,
        bootstrap_service: WorkingFileBootstrapService,
        csv_repository: CsvRepositoryProtocol,
    ) -> None:
        self._bootstrap_service = bootstrap_service
        self._csv_repository = csv_repository

    def save_updates(
        self,
        file_contract: ReviewFileContract,
        index: int,
        updates: dict,
    ) -> dict:
        self._bootstrap_service.ensure_working_file(file_contract)
        working_table = self._csv_repository.load_table(file_contract.working_file_path)

        if index is None or index < 0 or index >= len(working_table.rows):
            raise InvalidRowIndexError("Invalid row index.")

        for field_name, value in updates.items():
            if field_name in working_table.fieldnames:
                working_table.rows[index][field_name] = value

        self._csv_repository.save_table(
            file_path=file_contract.working_file_path,
            fieldnames=working_table.fieldnames,
            rows=working_table.rows,
        )

        return {
            "ok": True,
            "saved_index": index,
        }


class RowRevertService:
    """Restores one working row from the original CSV file."""

    def __init__(self, csv_repository: CsvRepositoryProtocol) -> None:
        self._csv_repository = csv_repository

    def revert_row(self, file_contract: ReviewFileContract, index: int) -> dict:
        working_table = self._csv_repository.load_table(file_contract.working_file_path)
        original_table = self._csv_repository.load_table(file_contract.original_file_path)

        if index is None or index < 0 or index >= len(working_table.rows):
            raise InvalidRowIndexError("Invalid row index.")

        row_identifier = working_table.rows[index].get("id", "")
        original_row = next(
            (
                row
                for row in original_table.rows
                if row.get("id") == row_identifier
            ),
            None,
        )

        if original_row is None:
            raise SourceRowUnavailableError(
                f"Original row not found for identifier: {row_identifier}"
            )

        for field_name in working_table.fieldnames:
            if field_name in original_row:
                working_table.rows[index][field_name] = original_row[field_name]

        self._csv_repository.save_table(
            file_path=file_contract.working_file_path,
            fieldnames=working_table.fieldnames,
            rows=working_table.rows,
        )

        return {
            "ok": True,
            "reverted_index": index,
            "row": dict(working_table.rows[index]),
        }


class ErrorResponseFactory:
    """Builds JSON error responses."""

    def build_response(self, error: Exception) -> tuple[dict, int]:
        if isinstance(error, InvalidRowIndexError):
            return {"ok": False, "error": str(error)}, 400

        if isinstance(error, SourceRowUnavailableError):
            return {"ok": False, "error": str(error)}, 404

        if isinstance(error, TextSourceUnavailableError):
            return {"ok": False, "error": str(error)}, 404

        return {"ok": False, "error": "Internal service error."}, 500


class ReviewApplicationContainer:
    """Builds review application services."""

    def __init__(self, config: ReviewServiceConfig) -> None:
        self.config = config
        self.csv_repository = CsvRepository()
        self.bootstrap_service = WorkingFileBootstrapService()
        self.status_service = ReviewStatusService(self.bootstrap_service)
        self.rows_query_service = ReviewRowsQueryService(
            bootstrap_service=self.bootstrap_service,
            csv_repository=self.csv_repository,
        )
        self.text_service = TextSourceResolutionService(
            bootstrap_service=self.bootstrap_service,
            csv_repository=self.csv_repository,
            content_service=MessageContentExtractionService(
                HtmlTextNormalizationService(),
            ),
        )
        self.update_service = RowUpdateService(
            bootstrap_service=self.bootstrap_service,
            csv_repository=self.csv_repository,
        )
        self.revert_service = RowRevertService(
            csv_repository=self.csv_repository,
        )
        self.error_factory = ErrorResponseFactory()


def build_default_config() -> ReviewServiceConfig:
    current_date = date.today().strftime("%Y-%m-%d")
    base_dir = Path(__file__).resolve().parent
    static_dir = base_dir / DEFAULT_STATIC_DIRECTORY

    return ReviewServiceConfig(
        base_dir=base_dir,
        static_dir=static_dir,
        file_contract=ReviewFileContract(
            corpus_dir=DEFAULT_CORPUS_DIRECTORY,
            source_capture_dir=DEFAULT_SOURCE_CAPTURE_DIRECTORY,
            original_file_name=DEFAULT_ORIGINAL_FILE_NAME,
            working_file_name=WORKING_FILE_NAME_TEMPLATE.format(date=current_date),
        ),
    )


def create_application(config: ReviewServiceConfig | None = None) -> Flask:
    active_config = config or build_default_config()
    container = ReviewApplicationContainer(active_config)
    application = Flask(
        __name__,
        static_folder=str(active_config.static_dir),
        static_url_path="/static",
    )

    @application.route("/")
    def index():
        response = make_response(
            send_from_directory(active_config.static_dir, "index.html")
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    @application.route("/api/status")
    def status():
        status_payload = container.status_service.get_status(
            file_contract=active_config.file_contract,
            current_date=date.today().strftime("%Y-%m-%d"),
        )
        return jsonify(
            {
                "working_csv": status_payload.working_csv,
                "original_csv": status_payload.original_csv,
                "has_working": status_payload.has_working,
                "has_original": status_payload.has_original,
                "today": status_payload.today,
            }
        )

    @application.route("/api/rows")
    def get_rows():
        return jsonify(
            container.rows_query_service.get_rows(active_config.file_contract)
        )

    @application.route("/api/text/<int:index>")
    def get_text(index: int):
        try:
            text_payload = container.text_service.get_text(
                file_contract=active_config.file_contract,
                index=index,
            )
            return jsonify(
                {
                    "text": text_payload.text,
                    "filename": text_payload.filename,
                    "found": text_payload.found,
                }
            )
        except Exception as error:
            response_payload, status_code = container.error_factory.build_response(error)
            return jsonify(response_payload), status_code

    @application.route("/api/save", methods=["POST"])
    def save_row():
        try:
            payload = request.get_json(silent=True) or {}
            return jsonify(
                container.update_service.save_updates(
                    file_contract=active_config.file_contract,
                    index=payload.get("index"),
                    updates=payload.get("updates", {}),
                )
            )
        except Exception as error:
            response_payload, status_code = container.error_factory.build_response(error)
            return jsonify(response_payload), status_code

    @application.route("/api/revert", methods=["POST"])
    def revert_row():
        try:
            payload = request.get_json(silent=True) or {}
            return jsonify(
                container.revert_service.revert_row(
                    file_contract=active_config.file_contract,
                    index=payload.get("index"),
                )
            )
        except Exception as error:
            response_payload, status_code = container.error_factory.build_response(error)
            return jsonify(response_payload), status_code

    return application


app = create_application()


def main() -> None:
    config = build_default_config()
    WorkingFileBootstrapService().ensure_working_file(config.file_contract)
    print("Text review service started.")
    print(f"URL: http://localhost:{config.port}")
    print(f"Working file: {config.file_contract.working_file_path}")
    print(f"Source capture directory: {config.file_contract.source_capture_dir}")
    app.run(host=config.host, port=config.port, debug=False)


if __name__ == "__main__":
    main()
