#!/usr/bin/env python3
"""
Name: image_slice_service
Input: local image directories and base64 PNG slice payloads
Output: image responses, slice files, and slice metadata responses
Usage: run as a FastAPI application
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

MEDIA_TYPE_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

DEFAULT_STATIC_DIRECTORY = Path("static")
SLICE_DIRECTORY_NAME = "slices"
SLICE_FILE_EXTENSION = ".png"
SLICE_FILE_PATTERN_TEMPLATE = "{base_name}_slice_*.png"
SLICE_FILE_NAME_TEMPLATE = "{base_name}_slice_{index}.png"
SLICE_INDEX_PATTERN = re.compile(r"_slice_(\d+)\.png$")


class ImageSliceServiceError(Exception):
    """Base exception for image slice service failures."""


class InvalidDirectoryError(ImageSliceServiceError):
    """Raised when a requested directory is invalid."""


class InvalidImageFileError(ImageSliceServiceError):
    """Raised when a requested image file is invalid."""


class InvalidSlicePayloadError(ImageSliceServiceError):
    """Raised when a slice payload cannot be decoded."""


class SaveSliceRequest(BaseModel):
    folder: str = Field(..., min_length=1)
    image_name: str = Field(..., min_length=1)
    image_data: str = Field(..., min_length=1)


@dataclass(frozen=True)
class DirectoryBrowseResult:
    path: str
    parent: str
    dirs: list[str]
    images: list[str]


@dataclass(frozen=True)
class ImageListResult:
    images: list[str]


@dataclass(frozen=True)
class SliceListResult:
    slices: list[str]


@dataclass(frozen=True)
class SliceSaveResult:
    saved: str
    index: int


@dataclass(frozen=True)
class ImageFileResult:
    content: bytes
    media_type: str


class DirectoryBrowsingProtocol(Protocol):
    def browse(self, directory_path: Path) -> DirectoryBrowseResult:
        """Browse a local directory and return image-compatible entries."""


class ImageFileAccessProtocol(Protocol):
    def get_image_file(self, folder: str, filename: str) -> ImageFileResult:
        """Return image bytes and media type for a requested image file."""


class SliceStorageProtocol(Protocol):
    def save_slice(self, request: SaveSliceRequest) -> SliceSaveResult:
        """Persist a PNG slice and return its assigned file name."""

    def list_slices(self, folder: str, image_name: str) -> SliceListResult:
        """List slices associated with an image."""

    def get_slice_file(self, folder: str, filename: str) -> ImageFileResult:
        """Return PNG bytes for a requested slice file."""


class PathResolutionService:
    """Resolves and validates local file-system paths."""

    def resolve_directory(self, path_value: str | Path) -> Path:
        directory_path = Path(path_value).expanduser().resolve()

        if not directory_path.exists() or not directory_path.is_dir():
            raise InvalidDirectoryError("Invalid directory path.")

        return directory_path

    def resolve_image_file(self, folder: str, filename: str) -> Path:
        folder_path = self.resolve_directory(folder)
        image_path = (folder_path / filename).resolve()

        if not image_path.exists() or not image_path.is_file():
            raise InvalidImageFileError("Image file was not found.")

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise InvalidImageFileError("Unsupported image file extension.")

        return image_path

    def resolve_slice_file(self, folder: str, filename: str) -> Path:
        folder_path = self.resolve_directory(folder)
        slice_path = (folder_path / SLICE_DIRECTORY_NAME / filename).resolve()

        if not slice_path.exists() or not slice_path.is_file():
            raise InvalidImageFileError("Slice file was not found.")

        return slice_path


class FileSystemBrowserService:
    """Browses local directories and image files."""

    def __init__(self, path_service: PathResolutionService) -> None:
        self._path_service = path_service

    def browse(self, directory_path: Path) -> DirectoryBrowseResult:
        resolved_path = self._path_service.resolve_directory(directory_path)
        directories = sorted(
            [
                entry.name
                for entry in resolved_path.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            ],
            key=str.lower,
        )
        images = sorted(
            [
                entry.name
                for entry in resolved_path.iterdir()
                if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=str.lower,
        )

        return DirectoryBrowseResult(
            path=str(resolved_path),
            parent=str(resolved_path.parent),
            dirs=directories,
            images=images,
        )


class ImageFileAccessService:
    """Reads image files from the local file system."""

    def __init__(self, path_service: PathResolutionService) -> None:
        self._path_service = path_service

    def list_images(self, folder: str) -> ImageListResult:
        folder_path = self._path_service.resolve_directory(folder)
        images = sorted(
            [
                entry.name
                for entry in folder_path.iterdir()
                if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=str.lower,
        )
        return ImageListResult(images=images)

    def get_image_file(self, folder: str, filename: str) -> ImageFileResult:
        image_path = self._path_service.resolve_image_file(folder, filename)
        media_type = MEDIA_TYPE_BY_EXTENSION.get(
            image_path.suffix.lower(),
            "application/octet-stream",
        )
        return ImageFileResult(
            content=image_path.read_bytes(),
            media_type=media_type,
        )


class Base64ImageDecoder:
    """Decodes base64 image payloads."""

    def decode_png_payload(self, image_data: str) -> bytes:
        _, separator, payload = image_data.partition(",")
        encoded_data = payload if separator else image_data

        try:
            return base64.b64decode(encoded_data, validate=False)
        except Exception as error:
            raise InvalidSlicePayloadError("Invalid base64 image payload.") from error


class SliceIndexService:
    """Computes the next slice index for an image."""

    def get_next_index(self, slices_dir: Path, base_name: str) -> int:
        existing_slice_files = list(
            slices_dir.glob(
                SLICE_FILE_PATTERN_TEMPLATE.format(base_name=base_name)
            )
        )
        indices: list[int] = []

        for slice_file in existing_slice_files:
            match = SLICE_INDEX_PATTERN.search(slice_file.name)

            if match:
                indices.append(int(match.group(1)))

        return max(indices) + 1 if indices else 0


class SliceStorageService:
    """Persists and reads image slices."""

    def __init__(
        self,
        path_service: PathResolutionService,
        decoder: Base64ImageDecoder,
        index_service: SliceIndexService,
    ) -> None:
        self._path_service = path_service
        self._decoder = decoder
        self._index_service = index_service

    def save_slice(self, request: SaveSliceRequest) -> SliceSaveResult:
        folder_path = self._path_service.resolve_directory(request.folder)
        slices_dir = folder_path / SLICE_DIRECTORY_NAME
        slices_dir.mkdir(exist_ok=True)

        base_name = Path(request.image_name).stem
        next_index = self._index_service.get_next_index(slices_dir, base_name)
        slice_filename = SLICE_FILE_NAME_TEMPLATE.format(
            base_name=base_name,
            index=next_index,
        )
        slice_path = slices_dir / slice_filename
        slice_path.write_bytes(
            self._decoder.decode_png_payload(request.image_data)
        )

        return SliceSaveResult(
            saved=slice_filename,
            index=next_index,
        )

    def list_slices(self, folder: str, image_name: str) -> SliceListResult:
        base_name = Path(image_name).stem
        folder_path = self._path_service.resolve_directory(folder)
        slices_dir = folder_path / SLICE_DIRECTORY_NAME

        if not slices_dir.exists():
            return SliceListResult(slices=[])

        slices = sorted(
            slice_file.name
            for slice_file in slices_dir.glob(
                SLICE_FILE_PATTERN_TEMPLATE.format(base_name=base_name)
            )
        )
        return SliceListResult(slices=slices)

    def get_slice_file(self, folder: str, filename: str) -> ImageFileResult:
        slice_path = self._path_service.resolve_slice_file(folder, filename)
        return ImageFileResult(
            content=slice_path.read_bytes(),
            media_type="image/png",
        )


class HttpExceptionMapper:
    """Maps service exceptions to HTTP exceptions."""

    def raise_for_error(self, error: Exception) -> None:
        if isinstance(error, InvalidDirectoryError):
            raise HTTPException(status_code=400, detail=str(error)) from error

        if isinstance(error, InvalidImageFileError):
            raise HTTPException(status_code=404, detail=str(error)) from error

        if isinstance(error, InvalidSlicePayloadError):
            raise HTTPException(status_code=400, detail=str(error)) from error

        raise HTTPException(status_code=500, detail="Internal service error.") from error


class ImageSliceApplicationContainer:
    """Builds application services."""

    def __init__(self) -> None:
        self.path_service = PathResolutionService()
        self.browser_service = FileSystemBrowserService(self.path_service)
        self.image_service = ImageFileAccessService(self.path_service)
        self.slice_service = SliceStorageService(
            path_service=self.path_service,
            decoder=Base64ImageDecoder(),
            index_service=SliceIndexService(),
        )
        self.exception_mapper = HttpExceptionMapper()


def create_application(
    static_directory: Path = DEFAULT_STATIC_DIRECTORY,
) -> FastAPI:
    container = ImageSliceApplicationContainer()
    application = FastAPI(title="Image Slice Service")

    @application.get("/api/browse")
    def browse(path: str = ".") -> dict:
        try:
            result = container.browser_service.browse(Path(path))
            return {
                "path": result.path,
                "parent": result.parent,
                "dirs": result.dirs,
                "images": result.images,
            }
        except Exception as error:
            container.exception_mapper.raise_for_error(error)

    @application.get("/api/list-images")
    def list_images(folder: str) -> dict:
        try:
            result = container.image_service.list_images(folder)
            return {"images": result.images}
        except Exception as error:
            container.exception_mapper.raise_for_error(error)

    @application.get("/api/image")
    def get_image(folder: str, filename: str) -> Response:
        try:
            result = container.image_service.get_image_file(folder, filename)
            return Response(
                content=result.content,
                media_type=result.media_type,
            )
        except Exception as error:
            container.exception_mapper.raise_for_error(error)

    @application.post("/api/save-slice")
    def save_slice(request: SaveSliceRequest) -> dict:
        try:
            result = container.slice_service.save_slice(request)
            return {
                "saved": result.saved,
                "index": result.index,
            }
        except Exception as error:
            container.exception_mapper.raise_for_error(error)

    @application.get("/api/list-slices")
    def list_slices(folder: str, image_name: str) -> dict:
        try:
            result = container.slice_service.list_slices(folder, image_name)
            return {"slices": result.slices}
        except Exception as error:
            container.exception_mapper.raise_for_error(error)

    @application.get("/api/slice-image")
    def get_slice_image(folder: str, filename: str) -> Response:
        try:
            result = container.slice_service.get_slice_file(folder, filename)
            return Response(
                content=result.content,
                media_type=result.media_type,
            )
        except Exception as error:
            container.exception_mapper.raise_for_error(error)

    if static_directory.exists():
        application.mount(
            "/",
            StaticFiles(directory=str(static_directory), html=True),
            name="static",
        )

    return application


app = create_application()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
