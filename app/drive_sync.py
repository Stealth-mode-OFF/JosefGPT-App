from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import importlib.metadata as stdlib_metadata

try:  # Provide compatibility on Python < 3.10 where packages_distributions is missing
    from importlib_metadata import packages_distributions as backport_packages_distributions
except ImportError:  # pragma: no cover - backport not installed
    backport_packages_distributions = None

if not hasattr(stdlib_metadata, "packages_distributions") and backport_packages_distributions:
    stdlib_metadata.packages_distributions = backport_packages_distributions  # type: ignore[attr-defined]

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from app.config import get_settings
from ingest_books import SUPPORTED_SUFFIXES

FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _sanitize_segment(segment: str) -> str:
    cleaned = segment.replace("/", "_").replace("\\", "_").strip()
    return cleaned or "untitled"


def _parse_modified(value: Optional[str]) -> int:
    if not value:
        return 0
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return 0


def _build_drive_service():
    settings = get_settings()
    credentials_path = settings.google_service_account_file
    if credentials_path is None or not credentials_path.exists():
        raise ValueError(
            "Google Drive credentials not found. Set GOOGLE_SERVICE_ACCOUNT_FILE to a valid service account JSON."
        )
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _download_file(service, file_id: str, destination: Path) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def sync_google_drive(folder_ids: Optional[Iterable[str]] = None) -> Dict[str, int]:
    settings = get_settings()
    cache_dir = settings.google_drive_cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_folders = list(folder_ids or settings.google_drive_folder_ids)
    if not target_folders:
        raise ValueError(
            "No Google Drive folders configured. Set GOOGLE_DRIVE_FOLDER_IDS in your environment."
        )

    service = _build_drive_service()
    supported_suffixes = {suffix.lower() for suffix in SUPPORTED_SUFFIXES}
    seen_files = set()

    stats = {
        "folders": 0,
        "downloaded": 0,
        "skipped": 0,
        "removed": 0,
        "errors": 0,
    }

    def walk_folder(folder_id: str, relative_path: Path):
        stats["folders"] += 1
        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for item in response.get("files", []):
                mime_type = item.get("mimeType")
                if mime_type == FOLDER_MIME:
                    sub_name = _sanitize_segment(item.get("name", "folder"))
                    walk_folder(item["id"], relative_path / sub_name)
                    continue

                if not mime_type or mime_type.startswith("application/vnd.google-apps."):
                    stats["skipped"] += 1
                    continue

                name = item.get("name") or "file"
                suffix = Path(name).suffix.lower()
                if suffix not in supported_suffixes:
                    stats["skipped"] += 1
                    continue

                file_id = item["id"]
                file_folder = cache_dir / relative_path / file_id
                file_folder.mkdir(parents=True, exist_ok=True)
                destination = file_folder / _sanitize_segment(name)
                seen_files.add(destination)

                expected_size = int(item.get("size") or 0)
                modified_ts = _parse_modified(item.get("modifiedTime"))

                if destination.exists():
                    stat = destination.stat()
                    if (
                        expected_size
                        and stat.st_size == expected_size
                        and modified_ts
                        and int(stat.st_mtime) == modified_ts
                    ):
                        stats["skipped"] += 1
                        continue

                try:
                    _download_file(service, file_id, destination)
                    if modified_ts:
                        os.utime(destination, (modified_ts, modified_ts))
                    stats["downloaded"] += 1
                except HttpError as exc:
                    stats["errors"] += 1
                    print(f"⚠️ Failed to download {name} ({file_id}): {exc}")

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    for folder_id in target_folders:
        try:
            folder = (
                service.files()
                .get(fileId=folder_id, fields="id, name", supportsAllDrives=True)
                .execute()
            )
        except HttpError as exc:
            stats["errors"] += 1
            print(f"⚠️ Failed to inspect folder {folder_id}: {exc}")
            continue

        folder_name = _sanitize_segment(folder.get("name", folder_id))
        walk_folder(folder_id, Path(folder_name))

    # Remove files no longer present in Drive
    for local_file in cache_dir.rglob("*"):
        if local_file.is_file() and local_file.suffix.lower() in supported_suffixes:
            if local_file not in seen_files:
                try:
                    local_file.unlink()
                    stats["removed"] += 1
                except OSError:
                    pass

    # Clean up empty directories
    for local_dir in sorted(
        (path for path in cache_dir.rglob("*") if path.is_dir()), reverse=True
    ):
        try:
            local_dir.rmdir()
        except OSError:
            continue

    return stats
