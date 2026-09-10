"""
Image preprocessing utilities for the POSAS-MLLM study.

Handles: auto-rotation, resize, padding, compression, hashing.
Designed to preserve clinical scar detail while reducing API costs.
"""

import hashlib
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_image_info(filepath: str) -> dict:
    """
    Extract basic image metadata without loading full pixel data.

    Returns:
        Dict with keys: width, height, format, mode, file_size_kb
    """
    path = Path(filepath)
    file_size_kb = path.stat().st_size / 1024

    with Image.open(filepath) as img:
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "file_size_kb": round(file_size_kb, 1),
        }


def preprocess_image(
    input_path: str,
    output_path: str,
    max_dimension: int = 1024,
    jpeg_quality: int = 85,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    auto_orient: bool = True,
) -> dict:
    """
    Preprocess a scar photograph for API submission.

    Steps (in order):
        1. Auto-rotate based on EXIF orientation
        2. Convert to RGB (handles RGBA, grayscale, palette modes)
        3. Resize so longest edge = max_dimension (Lanczos resampling)
        4. Pad shorter dimension to create a square image
        5. Save as JPEG at specified quality

    Args:
        input_path: Path to original image file.
        output_path: Path to save processed image.
        max_dimension: Target size for longest edge in pixels.
        jpeg_quality: JPEG compression quality (1-100).
        pad_color: RGB tuple for padding (default: black).
        auto_orient: Whether to apply EXIF auto-rotation.

    Returns:
        Dict with preprocessing metadata for the manifest.
    """
    with Image.open(input_path) as img:
        original_size = img.size
        original_format = img.format

        if auto_orient:
            img = ImageOps.exif_transpose(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        if max(width, height) > max_dimension:
            if width >= height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)
        else:
            new_width, new_height = width, height

        if new_width != new_height:
            square_size = max(new_width, new_height)
            padded = Image.new("RGB", (square_size, square_size), pad_color)
            paste_x = (square_size - new_width) // 2
            paste_y = (square_size - new_height) // 2
            padded.paste(img, (paste_x, paste_y))
            img = padded

        processed_size = img.size

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        output = output.with_suffix(".jpg")
        img.save(str(output), "JPEG", quality=jpeg_quality, optimize=True)

    output_size_kb = output.stat().st_size / 1024

    return {
        "original_width": original_size[0],
        "original_height": original_size[1],
        "original_format": original_format,
        "processed_width": processed_size[0],
        "processed_height": processed_size[1],
        "output_file_size_kb": round(output_size_kb, 1),
        "output_path": str(output),
    }
