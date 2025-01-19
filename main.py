#!/usr/bin/env python3

import sys
import os
import gc
import logging
from pathlib import Path

import cv2
import numpy as np

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def ensure_output_dir():
    """
    Ensures that there's an 'output' directory available.
    Returns a Path object pointing to that directory.
    """
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    return out_dir

def validate_grayscale_image(img: np.ndarray) -> None:
    """
    Checks that the image is a valid 8-bit grayscale array.
    Raises ValueError if checks fail.
    """
    if img is None:
        raise ValueError("No image data was loaded.")
    if img.dtype != np.uint8:
        raise ValueError("Image must be of type uint8 (8-bit grayscale).")
    if len(img.shape) != 2:
        raise ValueError("Image must be a 2D grayscale array.")
    if img.size > 100_000_000:
        logging.warning("The image is extremely large; performance may be an issue.")

def compute_kernel_positions(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given structuring element (kernel), confirms it has odd dimensions
    and returns (kernel, row_offsets, col_offsets).
    The row_offsets and col_offsets indicate which positions in the kernel are active (1).
    """
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd for proper centering.")
    row_idxs, col_idxs = np.where(kernel == 1)
    center_y = kh // 2
    center_x = kw // 2
    row_offsets = row_idxs - center_y
    col_offsets = col_idxs - center_x
    return kernel, row_offsets, col_offsets

def dilate_image(img: np.ndarray, kernel: np.ndarray, drows: np.ndarray, dcols: np.ndarray) -> np.ndarray:
    """
    Custom grayscale dilation.
    For each pixel, takes the maximum value of the image region
    under the kernel positions where kernel==1.
    Uses 'symmetric' padding to avoid boundary artifacts.
    """
    h, w = img.shape
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    out = np.zeros_like(img)

    for r_off, c_off in zip(drows, dcols):
        window = padded[
            pad_h + r_off : pad_h + r_off + h,
            pad_w + c_off : pad_w + c_off + w
        ]
        out = np.maximum(out, window)

    return out

def erode_image(img: np.ndarray, kernel: np.ndarray, r_offsets: np.ndarray, c_offsets: np.ndarray) -> np.ndarray:
    """
    Custom grayscale erosion.
    For each pixel, takes the minimum value of the image region
    under the kernel positions where kernel==1.
    Uses 'symmetric' padding to avoid boundary artifacts.
    """
    h, w = img.shape
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    out = np.full_like(img, 255)

    for r_off, c_off in zip(r_offsets, c_offsets):
        window = padded[
            pad_h + r_off : pad_h + r_off + h,
            pad_w + c_off : pad_w + c_off + w
        ]
        out = np.minimum(out, window)

    return out

def open_image(img: np.ndarray, kernel: np.ndarray, r_offsets: np.ndarray, c_offsets: np.ndarray) -> np.ndarray:
    """
    Custom opening = erosion followed by dilation.
    """
    eroded = erode_image(img, kernel, r_offsets, c_offsets)
    opened = dilate_image(eroded, kernel, r_offsets, c_offsets)
    return opened

def close_image(img: np.ndarray, kernel: np.ndarray, r_offsets: np.ndarray, c_offsets: np.ndarray) -> np.ndarray:
    """
    Custom closing = dilation followed by erosion.
    """
    dilated = dilate_image(img, kernel, r_offsets, c_offsets)
    closed = erode_image(dilated, kernel, r_offsets, c_offsets)
    return closed

def validate_operation_result(result: np.ndarray, reference: np.ndarray, operation_name: str) -> np.ndarray:
    """
    Checks that an operation's result is valid:
    same shape, same dtype, finite values.
    """
    if result.shape != reference.shape:
        raise ValueError(f"{operation_name} has changed the image dimensions.")
    if result.dtype != reference.dtype:
        raise ValueError(f"{operation_name} altered the image data type.")
    if not np.isfinite(result).all():
        raise ValueError(f"Found invalid (NaN or Inf) values after {operation_name}.")
    return result

def perform_morphology_operations(img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Given a grayscale image, perform:
      - Dilation
      - Erosion
      - Opening
      - Closing
    Returns a dictionary with each resulting image under its respective key.
    """
    validate_grayscale_image(img)

    # Create a standard 3x3 rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel, row_offsets, col_offsets = compute_kernel_positions(kernel)

    results = dict()
    results["original"] = img.copy()

    # List of operations to apply in a loop
    operations = [
        ("dilation", dilate_image),
        ("erosion", erode_image),
        ("opening", open_image),
        ("closing", close_image),
    ]

    for name, func in operations:
        logging.info(f"Applying {name} ...")
        processed = func(img, kernel, row_offsets, col_offsets)
        validated = validate_operation_result(processed, img, name)
        results[name] = validated

    return results

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <grayscale_image.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        logging.error(f"No such file: {image_path}")
        sys.exit(1)

    out_dir = ensure_output_dir()

    try:
        logging.info(f"Reading image: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Could not load image data.")

        logging.info(f"Processing image of shape {image.shape}")
        results = perform_morphology_operations(image)

        logging.info("Saving results...")
        for name, out_img in results.items():
            save_path = out_dir / f"{name}.jpg"
            cv2.imwrite(str(save_path), out_img)
            logging.info(f"Saved {save_path}")

        logging.info("Done. All operations completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        gc.collect()

if __name__ == "__main__":
    main()
