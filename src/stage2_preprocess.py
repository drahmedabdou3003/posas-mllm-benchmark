"""
Stage 2: Image Preprocessing

Reads the manifest from Stage 1, processes each raw image
(auto-orient, resize longest edge to 1024px, pad to square, JPEG-85),
saves to processed_images/, and updates the manifest with processed metadata.

Skips images that are already preprocessed (based on processed_path).

Usage:
    python -m src.stage2_preprocess
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.stage1_ingest import MANIFEST_COLUMNS, load_settings, read_manifest
from src.utils.image_utils import preprocess_image
from src.utils.logger import setup_logger

logger = logging.getLogger("posas")


def run_stage2(settings: dict = None) -> list:
    """Execute Stage 2: preprocess all images in the manifest."""
    if settings is None:
        settings = load_settings()

    manifest_path = settings["paths"]["manifest"]
    processed_folder = settings["paths"]["processed_images"]
    prep = settings["preprocessing"]

    logger.info("=" * 60)
    logger.info("STAGE 2: Image Preprocessing")
    logger.info("=" * 60)

    if not Path(manifest_path).exists():
        logger.error(f"Manifest not found: {manifest_path}. Run Stage 1 first.")
        return []

    manifest = read_manifest(manifest_path)
    logger.info(f"Loaded manifest with {len(manifest)} image(s)")

    processed_count = 0
    skipped_count = 0

    for row in manifest:
        image_id = row["image_id"]
        original_path = row["original_path"]

        if row.get("processed_path"):
            logger.debug(f"{image_id}: Already preprocessed, skipping")
            skipped_count += 1
            continue

        if not Path(original_path).exists():
            logger.error(f"{image_id}: Source file missing: {original_path}")
            continue

        stem = Path(original_path).stem
        output_path = str(Path(processed_folder) / f"{image_id}_{stem}.jpg")

        logger.info(f"Processing {image_id}: {Path(original_path).name}")

        try:
            result = preprocess_image(
                input_path=original_path,
                output_path=output_path,
                max_dimension=prep["max_dimension"],
                jpeg_quality=prep["jpeg_quality"],
                pad_color=tuple(prep["pad_color"]),
                auto_orient=prep["auto_orient"],
            )

            row["processed_path"] = result["output_path"]
            row["processed_width"] = result["processed_width"]
            row["processed_height"] = result["processed_height"]
            row["processed_size_kb"] = result["output_file_size_kb"]
            row["preprocessed_at"] = datetime.now(timezone.utc).isoformat()

            orig_kb = float(row["original_size_kb"])
            proc_kb = result["output_file_size_kb"]
            ratio = (1 - proc_kb / orig_kb) * 100 if orig_kb > 0 else 0
            logger.info(
                f"  {image_id}: {int(orig_kb)}KB -> {int(proc_kb)}KB "
                f"({ratio:.0f}% reduction) | "
                f"{result['processed_width']}x{result['processed_height']}px"
            )
            processed_count += 1

        except Exception as e:
            logger.error(f"  {image_id}: Preprocessing failed: {e}")
            continue

    _write_updated_manifest(manifest, manifest_path)

    logger.info("-" * 40)
    logger.info(f"Processed:  {processed_count}")
    logger.info(f"Skipped:    {skipped_count}")
    logger.info(f"Total:      {len(manifest)}")

    total_orig_kb = sum(float(r["original_size_kb"]) for r in manifest)
    total_proc_kb = sum(
        float(r["processed_size_kb"]) for r in manifest if r.get("processed_size_kb")
    )
    if total_orig_kb > 0:
        overall_reduction = (1 - total_proc_kb / total_orig_kb) * 100
        logger.info(
            f"Total size: {total_orig_kb / 1024:.1f}MB -> "
            f"{total_proc_kb / 1024:.1f}MB ({overall_reduction:.0f}% reduction)"
        )

    logger.info("Stage 2 complete.")
    return manifest


def _write_updated_manifest(manifest: list, manifest_path: str) -> None:
    """Overwrite manifest CSV with updated rows."""
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)
    logger.info(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    settings = load_settings()
    log_cfg = settings.get("logging", {})
    setup_logger(
        name="posas",
        log_file=log_cfg.get("log_file", "outputs/pipeline.log"),
        level=log_cfg.get("level", "INFO"),
        console_output=log_cfg.get("console_output", True),
    )
    run_stage2(settings)
