"""
Stage 1: Image Ingestion and Manifest Creation

Scans the raw_images folder, assigns sequential IDs, computes file hashes,
and writes a manifest CSV that tracks every image through the pipeline.

The manifest is the single source of truth for which images exist and
their processing status. All subsequent stages read from it.

Usage:
    python -m src.stage1_ingest
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.utils.image_utils import compute_file_hash, get_image_info
from src.utils.logger import setup_logger

logger = logging.getLogger("posas")

MANIFEST_COLUMNS = [
    "image_id",
    "original_filename",
    "original_path",
    "file_hash_sha256",
    "original_width",
    "original_height",
    "original_format",
    "original_mode",
    "original_size_kb",
    "registered_at",
    # Columns added by Stage 2 (preprocessing):
    "processed_path",
    "processed_width",
    "processed_height",
    "processed_size_kb",
    "preprocessed_at",
    # Columns for manual annotation (filled by researcher later):
    "scar_type_label",
    "scar_location",
    "fitzpatrick_type",
    "notes",
]


def load_settings() -> dict:
    """Load settings from config/settings.yaml."""
    settings_path = Path("config/settings.yaml")
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Settings file not found: {settings_path}. "
            f"Run from the project root directory."
        )
    with open(settings_path, "r") as f:
        return yaml.safe_load(f)


def scan_image_folder(folder: str, extensions: list) -> list:
    """Find all image files in the given folder (non-recursive)."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    images = []
    for ext in extensions:
        images.extend(folder_path.glob(f"*{ext}"))
        images.extend(folder_path.glob(f"*{ext.upper()}"))

    seen = set()
    unique = []
    for img in sorted(images):
        resolved = img.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(img)

    return sorted(unique, key=lambda p: p.name.lower())


def build_manifest(image_paths: list, start_id: int = 1) -> list:
    """Build manifest rows from a list of image file paths."""
    manifest = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for idx, path in enumerate(image_paths, start=start_id):
        image_id = f"IMG_{idx:03d}"

        logger.info(f"Registering {image_id}: {path.name}")

        file_hash = compute_file_hash(str(path))
        info = get_image_info(str(path))

        row = {
            "image_id": image_id,
            "original_filename": path.name,
            "original_path": str(path),
            "file_hash_sha256": file_hash,
            "original_width": info["width"],
            "original_height": info["height"],
            "original_format": info["format"],
            "original_mode": info["mode"],
            "original_size_kb": info["file_size_kb"],
            "registered_at": timestamp,
            "processed_path": "",
            "processed_width": "",
            "processed_height": "",
            "processed_size_kb": "",
            "preprocessed_at": "",
            "scar_type_label": "",
            "scar_location": "",
            "fitzpatrick_type": "",
            "notes": "",
        }
        manifest.append(row)

    return manifest


def write_manifest(manifest: list, output_path: str) -> None:
    """Write manifest rows to a CSV file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)

    logger.info(f"Manifest written: {out} ({len(manifest)} images)")


def read_manifest(manifest_path: str) -> list:
    """Read manifest CSV into a list of dicts."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_stage1(settings: dict = None) -> list:
    """Execute Stage 1: scan folder, build manifest, write CSV."""
    if settings is None:
        settings = load_settings()

    raw_folder = settings["paths"]["raw_images"]
    manifest_path = settings["paths"]["manifest"]
    extensions = settings["supported_extensions"]

    logger.info("=" * 60)
    logger.info("STAGE 1: Image Ingestion")
    logger.info("=" * 60)
    logger.info(f"Scanning folder: {raw_folder}")

    image_paths = scan_image_folder(raw_folder, extensions)

    if not image_paths:
        logger.warning(
            f"No images found in {raw_folder}. "
            f"Supported formats: {', '.join(extensions)}"
        )
        return []

    logger.info(f"Found {len(image_paths)} image(s)")

    if Path(manifest_path).exists():
        existing = read_manifest(manifest_path)
        existing_hashes = {row["file_hash_sha256"] for row in existing}
        new_paths = [
            p for p in image_paths
            if compute_file_hash(str(p)) not in existing_hashes
        ]
        if not new_paths:
            logger.info("All images already registered in manifest. Skipping.")
            return existing
        logger.info(
            f"{len(new_paths)} new image(s) to register "
            f"({len(image_paths) - len(new_paths)} already in manifest)"
        )
        new_rows = build_manifest(new_paths, start_id=len(existing) + 1)
        manifest = existing + new_rows
        write_manifest(manifest, manifest_path)

        total_size_mb = sum(
            float(row["original_size_kb"]) for row in new_rows
        ) / 1024
        logger.info(f"Total raw image size: {total_size_mb:.1f} MB")
        logger.info("Stage 1 complete.")
        return manifest

    manifest = build_manifest(image_paths)
    write_manifest(manifest, manifest_path)

    total_size_mb = sum(row["original_size_kb"] for row in manifest) / 1024
    logger.info(f"Total raw image size: {total_size_mb:.1f} MB")
    logger.info("Stage 1 complete.")

    return manifest


if __name__ == "__main__":
    settings = load_settings()
    log_cfg = settings.get("logging", {})
    setup_logger(
        name="posas",
        log_file=log_cfg.get("log_file", "outputs/pipeline.log"),
        level=log_cfg.get("level", "INFO"),
        console_output=log_cfg.get("console_output", True),
    )
    run_stage1(settings)
