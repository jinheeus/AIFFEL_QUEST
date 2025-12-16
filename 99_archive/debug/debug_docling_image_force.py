from docling.document_converter import DocumentConverter
from pdf2image import convert_from_path
import os
import shutil

# Config
INPUT_PDF = "00_data/raw_data/1_alio_raw_files/C0261/2024010502741546-00.pdf"
TEMP_IMG_DIR = "debug_output_C0261/images"
OUTPUT_DIR = "debug_output_C0261"
os.makedirs(TEMP_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_image_force_test():
    print(f"--- Converting PDF to Images: {INPUT_PDF} ---")

    # 1. Convert PDF to images (requires poppler installed on system, mac usually has it or uses simple python fallback if lucky, else might fail)
    # If pdf2image fails, we might need another way or just skip this test.
    try:
        images = convert_from_path(INPUT_PDF, dpi=300)
    except Exception as e:
        print(f"Failed to convert PDF to images: {e}")
        return

    print(f"Converted {len(images)} pages.")

    # 2. Process each image with Docling
    converter = DocumentConverter()

    full_md = ""

    for i, img in enumerate(images):
        print(f"Processing Page {i + 1}...")

        # Save temp image (Docling v2 handles memory images depending on API, but file path is safest)
        img_path = os.path.join(TEMP_IMG_DIR, f"page_{i:03d}.png")
        img.save(img_path, "PNG")

        # Convert
        result = converter.convert(img_path)
        md = result.document.export_to_markdown()

        full_md += f"\n\n<!-- Page {i + 1} -->\n\n" + md

    # 3. Save Combined MD
    out_path = os.path.join(OUTPUT_DIR, "image_force.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"Saved Forced Image OCR result to {out_path}")


if __name__ == "__main__":
    run_image_force_test()
