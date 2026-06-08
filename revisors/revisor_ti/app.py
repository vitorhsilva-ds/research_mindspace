#!/usr/bin/env python3
"""
Name: interface_review_service
Input: reviewed interface CSV file and related image files
Output: editable working CSV file and JSON API responses
Usage: run as a Flask application
"""

from __future__ import annotations

import csv
import mimetypes
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Protocol

from flask import Flask, abort, jsonify, make_response, request, send_file, send_from_directory


DEFAULT_ENCODING = "utf-8"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5051
DEFAULT_STATIC_DIRECTORY = "static"
DEFAULT_CORPUS_DIRECTORY = Path(
    os.environ.get(
        "REVIEW_CORPUS_DIR",
        str(Path.home() / "workspace" / "corpus"),
    )
)
DEFAULT_ORIGINAL_FILE_NAME = "ti_revisao.csv"
WORKING_FILE_NAME_TEMPLATE = "ti_revisao_{date}.csv"

MINDSPACE_COLUMNS = ("M", "I", "N", "D", "S", "P", "A", "C", "E")


class ReviewServiceError(Exception):
    """Base exception for review service failures."""


class CsvFileUnavailableError(ReviewServiceError):
    """Raised when a required CSV file is unavailable."""


class InvalidRowIndexError(ReviewServiceError):
    """Raised when a row index is outside the editable range."""


class SourceRowUnavailableError(ReviewServiceError):
    """Raised when a source row cannot be resolved."""


class ImageFileUnavailableError(ReviewServiceError):
    """Raised when an image file cannot be resolved."""


@dataclass(frozen=True)
class ReviewFileContract:
    corpus_dir: Path
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
class ImageResponsePayload:
    file_path: Path
    media_type: str


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


class ImageResolutionService:
    """Resolves image paths for editable rows."""

    def __init__(
        self,
        bootstrap_service: WorkingFileBootstrapService,
        csv_repository: CsvRepositoryProtocol,
    ) -> None:
        self._bootstrap_service = bootstrap_service
        self._csv_repository = csv_repository

    def resolve_image(self, file_contract: ReviewFileContract, index: int) -> ImageResponsePayload:
        self._bootstrap_service.ensure_working_file(file_contract)
        working_table = self._csv_repository.load_table(file_contract.working_file_path)

        if index < 0 or index >= len(working_table.rows):
            raise InvalidRowIndexError("Invalid row index.")

        source_file_path = working_table.rows[index].get("filepath", "")

        if not source_file_path:
            raise ImageFileUnavailableError("Image path is empty.")

        image_file_path = Path(source_file_path).expanduser().resolve()

        if not image_file_path.exists():
            raise ImageFileUnavailableError(f"Image file not found: {image_file_path}")

        media_type, _ = mimetypes.guess_type(str(image_file_path))

        return ImageResponsePayload(
            file_path=image_file_path,
            media_type=media_type or "image/png",
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

        if isinstance(error, CsvFileUnavailableError):
            return {"ok": False, "error": str(error)}, 404

        if isinstance(error, ImageFileUnavailableError):
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
        self.image_service = ImageResolutionService(
            bootstrap_service=self.bootstrap_service,
            csv_repository=self.csv_repository,
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

    @application.route("/api/image/<int:index>")
    def get_image(index: int):
        try:
            image_payload = container.image_service.resolve_image(
                file_contract=active_config.file_contract,
                index=index,
            )
            return send_file(
                image_payload.file_path,
                mimetype=image_payload.media_type,
            )
        except Exception as error:
            abort(container.error_factory.build_response(error)[1], description=str(error))

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
    print("Review service started.")
    print(f"URL: http://localhost:{config.port}")
    print(f"Working file: {config.file_contract.working_file_path}")
    app.run(host=config.host, port=config.port, debug=False)


if __name__ == "__main__":
    main()
