#!/usr/bin/env python3
"""
PDF -> text via PaddleOCR, returning the result as a string (no output file by default).

Usage:
    python ocr_pdf_to_text_cleanup_return.py my.pdf  # prints the text
    python ocr_pdf_to_text_cleanup_return.py my.pdf -o output.txt  # also writes to file
    python ocr_pdf_to_text_cleanup_return.py my.pdf --keep-intermediate  # keeps temps
"""

import os
import glob
import json
import shutil
from typing import List, Tuple, Optional

from pdf2image import convert_from_path
from paddleocr import PaddleOCR


def pdf_to_images(pdf_path: str, dpi: int = 100) -> Tuple[List[str], str]:
    """
    Convert a PDF file to images (one per page).
    Returns a list of image file paths and the output folder.
    """
    base, _ = os.path.splitext(pdf_path)
    output_folder = f"{base}_pages"
    os.makedirs(output_folder, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths: List[str] = []
    for i, page in enumerate(pages, start=1):
        img_path = os.path.join(output_folder, f"page_{i:03d}.png")
        page.save(img_path, "PNG")
        image_paths.append(img_path)
    return image_paths, output_folder


def ocr_images(
    image_paths: List[str],
    ocr: PaddleOCR,
    img_out_folder: str,
    json_out_folder: str,
) -> None:
    """
    Run OCR on a list of images, saving region images and JSON results.
    """
    os.makedirs(img_out_folder, exist_ok=True)
    os.makedirs(json_out_folder, exist_ok=True)

    for page_idx, img_path in enumerate(image_paths, start=1):
        results = ocr.predict(img_path)
        for res_idx, res in enumerate(results, start=1):
            res.print()
            out_img = os.path.join(img_out_folder, f"page{page_idx:03d}_reg{res_idx:02d}.png")
            res.save_to_img(out_img)
            out_json = os.path.join(json_out_folder, f"page{page_idx:03d}_reg{res_idx:02d}.json")
            res.save_to_json(out_json)


def collate_rec_texts_to_string(json_folder: str) -> str:
    """
    Extract all recognized texts from JSON files and return as a single string.
    """
    all_texts: List[str] = []
    for js_file in sorted(glob.glob(os.path.join(json_folder, "*.json"))):
        with open(js_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            texts = data.get("rec_texts", [])
            all_texts.extend(texts)
    return "\n".join(all_texts)


def ocr_pdf_to_text(
    pdf_path: str,
    output_txt: Optional[str] = None,
    dpi: int = 100,
    cleanup: bool = True,
) -> str:
    """
    Convert PDF to text via OCR.
    - pdf_path: Path to PDF file.
    - output_txt: Optional path to save the extracted text.
    - dpi: Resolution for image conversion (higher = better OCR, slower).
    - cleanup: Whether to remove intermediate files/folders.
    Returns the extracted text as a string.
    """
    image_paths, img_folder = pdf_to_images(pdf_path, dpi=dpi)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )

    # Prepare output folders
    out_dir = os.path.dirname(os.path.abspath(output_txt)) if output_txt else os.getcwd()
    img_out = os.path.join(out_dir, "ocr_images")
    json_out = os.path.join(out_dir, "ocr_json")
    ocr_images(image_paths, ocr, img_out, json_out)

    # Collate recognized text
    text = collate_rec_texts_to_string(json_out)
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as out:
            out.write(text)
        print(f"Wrote {len(text.splitlines())} lines into {output_txt}")

    # Clean up intermediate files if requested
    if cleanup:
        for folder in (img_folder, img_out, json_out):
            shutil.rmtree(folder, ignore_errors=True)
        print("Removed intermediate folders and files.")

    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF → PNGs → PaddleOCR.predict() → text string (with optional file output and cleanup)"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Optional path for the final text file (will be overwritten)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for PDF→image conversion (higher = slower & larger files)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Preserve intermediate image, crop, and JSON files",
    )

    args = parser.parse_args()
    result_text = ocr_pdf_to_text(
        args.pdf_path,
        output_txt=args.output,
        dpi=args.dpi,
        cleanup=not args.keep_intermediate,
    )
    # print(result_text)