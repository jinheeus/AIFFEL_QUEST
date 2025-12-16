import os
import glob
import re
import argparse
import shutil
from tqdm import tqdm
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from clean_pdf_watermark import remove_watermark_and_convert  # Import cleaning logic

# Settings
CORRUPTION_THRESHOLD = (
    0.05  # If > 5% of non-whitespace chars are CJK Ideographs (Hanja/Chinese)
)


def is_corrupted(file_path):
    """
    Detects if a markdown file is likely corrupted by bad OCR (Chinese hallucination).
    Logic: Count CJK Unified Ideographs (\u4e00-\u9fff).
    Korean docs can contain Hanja, but ratio shouldn't be extremely high if it's modern text.
    The corrupted example had almost purely Hanja/Gibberish.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return False, 0.0

    # Remove whitespace and common markdown syntax
    clean_text = re.sub(r"[\s\n\t\#\-\|]", "", text)
    if not clean_text:
        return False, 0.0

    # Count CJK Unified Ideographs (Includes Hanja and Chinese)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", clean_text))

    # Count Hangul (Syllables + Jamo)
    hangul_count = len(
        re.findall(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", clean_text)
    )

    total_chars = len(clean_text)
    if total_chars == 0:
        return False, 0.0

    cjk_ratio = cjk_count / total_chars

    # Heuristic: High CJK ratio AND Low Hangul ratio
    # If a document is 5% Hanja, it might be valid. But if it has very little Hangul, it's bad.
    if cjk_ratio > CORRUPTION_THRESHOLD and hangul_count < cjk_count:
        return True, cjk_ratio

    return False, cjk_ratio


def run_fix_batch(parsed_root, raw_root):
    print(f"Scanning {parsed_root} for corrupted files...")

    bad_files = []

    # 1. Detect
    md_files = glob.glob(os.path.join(parsed_root, "**/*.md"), recursive=True)
    for md_path in tqdm(md_files, desc="Detecting Corruption"):
        corrupted, score = is_corrupted(md_path)
        if corrupted:
            # Find corresponding PDF
            # parsed/alio_docling/C0105/file.md -> raw/1_alio_raw_files/C0105/file.pdf
            rel_path = os.path.relpath(
                md_path, parsed_root
            )  # alio_docling/C0105/file.md

            # Subdir mapping check
            if "alio_docling" in rel_path:
                raw_rel = rel_path.replace("alio_docling", "1_alio_raw_files").replace(
                    ".md", ".pdf"
                )
            elif "bai_docling" in rel_path:
                raw_rel = rel_path.replace("bai_docling", "1_bai_raw_files").replace(
                    ".md", ".pdf"
                )
            else:
                continue

            raw_path = os.path.join(raw_root, raw_rel)

            if os.path.exists(raw_path):
                bad_files.append((raw_path, md_path))
                # print(f"Found Bad File: {md_path} (CJK Ratio: {score:.2f})")
            else:
                print(f"Warning: Source PDF not found for {md_path}")

    print(f"Found {len(bad_files)} corrupted files to fix.")

    if not bad_files:
        return

    # 2. Re-run with Correct Options
    print("Initializing Docling with Correct Korean OCR Options...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.lang = ["ko"]  # CRITICAL FIX
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.accelerator_options.device = "cuda"

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    success = 0
    errors = 0

    for pdf_path, md_path in tqdm(bad_files, desc="Re-parsing Corrupted Files"):
        temp_cleaned_pdf = pdf_path.replace(".pdf", "_cleaned.pdf")
        try:
            # 1. Cleaner: Remove Watermark
            print(f"  -> Cleaning Watermark: {pdf_path}")
            cleaned_ok = remove_watermark_and_convert(pdf_path, temp_cleaned_pdf)

            target_pdf = temp_cleaned_pdf if cleaned_ok else pdf_path

            # Backup original bad file
            shutil.move(md_path, md_path + ".bad")

            # 2. Re-parse
            result = converter.convert(target_pdf)
            md = result.document.export_to_markdown()

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)

            success += 1

        except Exception as e:
            print(f"Error checking {pdf_path}: {e}")
            errors += 1
            # Restore backup if failed
            if os.path.exists(md_path + ".bad"):
                shutil.move(md_path + ".bad", md_path)

        finally:
            # Cleanup temp file
            if os.path.exists(temp_cleaned_pdf):
                os.remove(temp_cleaned_pdf)

    print(f"Fix Complete. Success: {success}, Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_root", default="./00_data/parsed_data")
    parser.add_argument("--raw_root", default="./00_data/raw_data")
    args = parser.parse_args()

    run_fix_batch(args.parsed_root, args.raw_root)
