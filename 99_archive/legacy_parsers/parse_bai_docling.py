import os
import glob
from tqdm import tqdm
from docling.document_converter import DocumentConverter
import argparse

# Configuration
INPUT_DIR = "00_data/raw_data/1_bai_raw_files"
OUTPUT_DIR = "00_data/parsed_data/bai_docling"


def parse_bai_with_docling(limit=None):
    # 1. Setup
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Find Files
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "**/*.pdf"), recursive=True)
    print(f"Found {len(pdf_files)} PDF files in {INPUT_DIR}")

    if limit:
        pdf_files = pdf_files[:limit]
        print(f"Limiting execution to first {limit} files.")

    # 3. Initialize Converter (Load model once)
    print("Initializing Docling Converter...")
    converter = DocumentConverter()

    # 4. Batch Process
    success_count = 0
    skip_count = 0
    error_count = 0

    for pdf_path in tqdm(pdf_files, desc="Parsing BAI PDFs"):
        try:
            # Determine Output Path
            relative_path = os.path.relpath(pdf_path, INPUT_DIR)
            output_subdir = os.path.dirname(relative_path)
            output_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"

            output_full_dir = os.path.join(OUTPUT_DIR, output_subdir)
            os.makedirs(output_full_dir, exist_ok=True)
            output_path = os.path.join(output_full_dir, output_name)

            # Skip if exists
            if os.path.exists(output_path):
                print(f"Skipping (Exists): {output_name}")
                skip_count += 1
                continue

            # Convert
            result = converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()

            # Save
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            success_count += 1

        except Exception as e:
            print(f"Error parsing {pdf_path}: {e}")
            error_count += 1

    print(f"\nProcessing Complete.")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of files to process"
    )
    args = parser.parse_args()
    parse_bai_with_docling(args.limit)
