"""
POSAS-MLLM Benchmark — Main Pipeline

Orchestrates all five stages of the evaluation pipeline:
    1. Ingest      — scan raw images, build manifest
    2. Preprocess  — resize + pad to 1024x1024 JPEG, strip EXIF
    3. Dispatch    — send each image to every configured model
    4. Validate    — parse & schema-check every response
    5. Export      — write outputs/results_master.xlsx + .csv

Usage:
    # Run individual stages
    python pipeline.py --stage 1
    python pipeline.py --stage 2

    # Run multiple stages sequentially
    python pipeline.py --stage 1,2
    python pipeline.py --stage 3,4,5 --yes

    # Run everything
    python pipeline.py --stage all --yes

    # Pilot mode (first N images only; N = pilot.max_images in settings.yaml)
    python pipeline.py --stage 3,4,5 --pilot --yes

    # Filter to a specific set of images by their original filename
    python pipeline.py --stage 3,4,5 --images S001.jpg,S002.jpg --yes
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.utils.logger import setup_logger

SETTINGS_FILE = "config/settings.yaml"


def load_settings() -> dict:
    settings_path = Path(SETTINGS_FILE)
    if not settings_path.exists():
        print(f"ERROR: Settings file not found: {settings_path}")
        print("Run this script from the project root directory.")
        sys.exit(1)
    with open(settings_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="POSAS-MLLM Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py --stage 1              # Scan images\n"
            "  python pipeline.py --stage 1,2            # Scan + preprocess\n"
            "  python pipeline.py --stage 3,4,5 --yes    # API + validate + export\n"
            "  python pipeline.py --stage all --yes      # Full pipeline\n"
            "  python pipeline.py --stage 3 --pilot      # Pilot (first N images)\n"
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Stage(s) to run: 1, 2, 3, 4, 5, 'all', or comma-separated (e.g., 1,2)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: process only the first N images (N = pilot.max_images)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-confirm the Stage-3 dry-run cost prompt",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Comma-separated original filenames to process (e.g. S001.jpg,S002.jpg)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings()

    if args.pilot:
        settings["pilot"]["enabled"] = True
        pilot_max = settings["pilot"]["max_images"]
        print(f"\n*** PILOT MODE: Processing max {pilot_max} images ***\n")

    settings["auto_confirm"] = getattr(args, "yes", False)

    if args.images:
        settings["image_filter"] = [f.strip() for f in args.images.split(",")]
        print(f"Image filter: {settings['image_filter']}")
    else:
        settings["image_filter"] = None

    log_cfg = settings.get("logging", {})
    logger = setup_logger(
        name="posas",
        log_file=log_cfg.get("log_file", "outputs/pipeline.log"),
        level=log_cfg.get("level", "INFO"),
        console_output=log_cfg.get("console_output", True),
    )

    if args.stage.lower() == "all":
        stages = [1, 2, 3, 4, 5]
    else:
        try:
            stages = [int(s.strip()) for s in args.stage.split(",")]
        except ValueError:
            logger.error(f"Invalid stage specification: {args.stage}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("POSAS-MLLM BENCHMARK PIPELINE")
    logger.info(f"Stages: {stages}")
    logger.info(f"Pilot mode: {settings['pilot']['enabled']}")
    logger.info("=" * 60)

    for stage_num in sorted(stages):
        if stage_num == 1:
            from src.stage1_ingest import run_stage1
            manifest = run_stage1(settings)
            if not manifest:
                logger.warning("No images found. Stopping pipeline.")
                break

        elif stage_num == 2:
            from src.stage2_preprocess import run_stage2
            run_stage2(settings)

        elif stage_num == 3:
            from src.stage3_dispatch import run_stage3
            run_stage3(settings)

        elif stage_num == 4:
            from src.stage4_validate import run_stage4
            run_stage4(settings)

        elif stage_num == 5:
            from src.stage5_export import run_stage5
            run_stage5(settings)

        else:
            logger.error(f"Unknown stage: {stage_num}. Valid stages: 1-5")

    logger.info("")
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
