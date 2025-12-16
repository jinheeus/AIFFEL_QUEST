from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
import os

# Config
INPUT_PATH = "00_data/raw_data/1_alio_raw_files/C0261/2024010502741546-00.pdf"
OUTPUT_BASE = "debug_output_C0261"
os.makedirs(OUTPUT_BASE, exist_ok=True)


def run_test(name, pipeline_options=None):
    print(f"--- Running Test: {name} ---")

    # Configure converter
    format_options = {}
    if pipeline_options:
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

    converter = DocumentConverter(format_options=format_options)

    # Convert
    result = converter.convert(INPUT_PATH)
    md = result.document.export_to_markdown()

    # Save
    out_path = os.path.join(OUTPUT_BASE, f"{name}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved to {out_path}")


# 1. Default
run_test("default")

# 2. Force OCR (Rasterize)
# This helps if the PDF has weird text encoding or invisible layers
opts_ocr = PdfPipelineOptions(do_ocr=True)
opts_ocr.do_table_structure = True
opts_ocr.table_structure_options.mode = TableFormerMode.ACCURATE  # Use ACCURATE model
run_test("ocr_accurate", opts_ocr)

# 3. No OCR but Accurate Tables
opts_tables = PdfPipelineOptions(do_ocr=False)
opts_tables.do_table_structure = True
opts_tables.table_structure_options.mode = TableFormerMode.ACCURATE
run_test("tables_accurate", opts_tables)
