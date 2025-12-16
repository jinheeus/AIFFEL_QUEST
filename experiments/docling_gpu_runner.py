import os
import glob
import argparse
import time
from tqdm import tqdm
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat

# Default Paths (Adjust as needed)
DEFAULT_INPUT_ROOT = "./00_data/raw_data"
DEFAULT_OUTPUT_ROOT = "./00_data/parsed_data"


def run_docling_batch(input_dir, output_dir, dataset_name="dataset", limit=None):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True)
    if limit:
        pdf_files = pdf_files[:limit]

    print(f"[{dataset_name}] Found {len(pdf_files)} PDF files.")

    # Configure Pipeline for GPU
    print("Initializing Docling with CUDA support...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.accelerator_options.device = "cuda"  # Force Usage of GPU

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    success = 0
    errors = 0
    skipped = 0

    print("Starting conversion...")
    for pdf_path in tqdm(pdf_files, desc=f"Processing {dataset_name}"):
        try:
            # Determin output path structure
            rel_path = os.path.relpath(pdf_path, input_dir)
            out_subdir = os.path.dirname(rel_path)
            out_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"

            full_out_dir = os.path.join(output_dir, out_subdir)
            os.makedirs(full_out_dir, exist_ok=True)
            out_path = os.path.join(full_out_dir, out_name)

            # Skip if already exists and is not empty
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                skipped += 1
                continue

            # Convert
            result = converter.convert(pdf_path)
            md = result.document.export_to_markdown()

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md)

            success += 1

        except Exception as e:
            # print(f"Error processing {pdf_path}: {e}")
            errors += 1

    print(f"[{dataset_name}] Processing Complete.")
    print(f"Success: {success}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docling Batch Runner for GPU")
    parser.add_argument(
        "--input_root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory for raw data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for parsed output",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of files for testing"
    )

    args = parser.parse_args()

    # Define sub-tasks
    # 1. ALIO
    alio_in = os.path.join(args.input_root, "1_alio_raw_files")
    alio_out = os.path.join(args.output_root, "alio_docling")
    if os.path.exists(alio_in):
        run_docling_batch(alio_in, alio_out, "ALIO", args.limit)
    else:
        print(f"Skipping ALIO (Not found at {alio_in})")

    # 2. BAI
    bai_in = os.path.join(args.input_root, "1_bai_raw_files")
    bai_out = os.path.join(args.output_root, "bai_docling")
    if os.path.exists(bai_in):
        run_docling_batch(bai_in, bai_out, "BAI", args.limit)
    else:
        print(f"Skipping BAI (Not found at {bai_in})")
