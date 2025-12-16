import pdfplumber
import os
import glob
from collections import Counter
from tqdm import tqdm
import re

# Try importing spacing library
try:
    from pykospacing import Spacing

    spacer = Spacing()
    HAS_SPACING_LIB = True
except ImportError:
    HAS_SPACING_LIB = False
    print("Warning: pykospacing not found. Spacing correction will be skipped.")

INPUT_DIR = "00_data/raw_data/1_alio_raw_files"
OUTPUT_DIR = "00_data/parsed_data/alio_local"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_font_sizes(pdf):
    """Analyze font sizes across the PDF to heuristically determine headers."""
    sizes = []
    # Analyze first 5 pages max to save time
    for page in pdf.pages[:5]:
        for char in page.chars:
            sizes.append(round(char["size"], 1))

    if not sizes:
        return 14.0, 12.0, 10.0  # Default H1, H2, Body

    counts = Counter(sizes)
    sorted_fonts = sorted(counts.items(), key=lambda x: x[0], reverse=True)
    # Filter very rare large fonts (>2 chars)
    valid_fonts = [size for size, freq in sorted_fonts if freq > 5]

    if not valid_fonts:
        return 14.0, 12.0, 10.0

    h1 = valid_fonts[0]
    h2 = valid_fonts[1] if len(valid_fonts) > 1 else h1
    try:
        body = counts.most_common(1)[0][0]
    except:
        body = 10.0

    return h1, h2, body


def clean_text_block(text):
    if not text:
        return ""
    # Remove excessive newlines
    text = re.sub(r"\n+", " ", text).strip()
    return text


def parse_pdf_hybrid(pdf_path):
    try:
        relative_path = os.path.relpath(pdf_path, INPUT_DIR)
        output_subdir = os.path.dirname(relative_path)
        output_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
        output_full_dir = os.path.join(OUTPUT_DIR, output_subdir)
        os.makedirs(output_full_dir, exist_ok=True)
        output_path = os.path.join(output_full_dir, output_name)

        if os.path.exists(output_path):
            return f"Skipped (Exists): {pdf_path}"

        full_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                pass  # Empty
            else:
                h1_size, h2_size, body_size = analyze_font_sizes(pdf)

                for i, page in enumerate(pdf.pages):
                    full_text += f"\n\n<!-- Page {i + 1} -->\n"

                    # 1. Extract Tables first to identify their bbox
                    tables = page.find_tables()
                    table_bboxes = [t.bbox for t in tables]

                    # 2. Extract Text (excluding tables) is hard in pdfplumber automatically
                    # Strategy: Extract text line by line, check if inside table.
                    # Simpler Strategy: Extract tables, convert to markdown.
                    # Then extract text normally, but how to mix?
                    # pdfplumber doesn't support easy "exclude table" text extraction.
                    # Workaround: effective hybrid is hard.

                    # Let's simple append: Text then Tables? No, context lost.
                    # Best effort: `extract_text_lines` and filter collisions?

                    # For now, let's do:
                    # 1. Page text (layout=True maintains some structure which is good for tables but duplicates text)
                    # 2. Actually, pdfplumber's `extract_text` handles basic layout.
                    # But we want Markdown Tables.

                    # Let's try: Extract words. Reconstruct? Too complex.
                    # LET'S DO: Page.extract_text() + Append Markdown Tables at bottom of page?
                    # Or: Filter tables out of text?

                    # User wants "Table as Markdown".
                    # If we use `page.extract_tables()`, we get data.
                    # If we use `page.extract_text()`, we get text representation of table (often bad).

                    # COMPROMISE STRATEGY for Reliability:
                    # 1. Extract Full Text (cleaning spacing).
                    # 2. Extract Tables formatted as Markdown.
                    # 3. Append Tables at the end of the page block.
                    # This duplicates data (bad text table + good markdown table), but ensures we don't lose info.
                    # It's better than missing text.

                    # Extract raw text for content
                    raw_text = page.extract_text() or ""

                    # Fix Spacing on raw text
                    if HAS_SPACING_LIB and len(raw_text) > 50:
                        # Split by lines to avoid memory issues
                        lines = raw_text.split("\n")
                        fixed_lines = []
                        for line in lines:
                            if len(line.strip()) > 5:
                                try:
                                    fixed_lines.append(spacer(line))
                                except:
                                    fixed_lines.append(line)
                            else:
                                fixed_lines.append(line)
                        raw_text = "\n".join(fixed_lines)

                    # Headers Heuristic (Post-processing check?)
                    # Hard to map back to text lines without words.
                    # Let's apply headers based on raw_text? No, we lost font size info in extract_text().
                    # We need to iterate over words/lines manually if we want headers.

                    # Simpler: Just save the text + markdown tables. Headers are nice-to-have,
                    # but Spacing + Tables is the core requirement.
                    # LlamaParse does headers well. Local is "Best Effort".

                    full_text += raw_text + "\n"

                    # Append Tables
                    extracted_tables = page.extract_tables()
                    if extracted_tables:
                        full_text += "\n\n### [Tables Extracted]\n"
                        for table in extracted_tables:
                            if not table:
                                continue
                            clean_table = [
                                [
                                    str(cell or "").replace("\n", " ").strip()
                                    for cell in row
                                ]
                                for row in table
                            ]
                            if not clean_table:
                                continue

                            # Md format
                            if len(clean_table) > 0:
                                headers = clean_table[0]
                                full_text += "| " + " | ".join(headers) + " |\n"
                                full_text += (
                                    "| " + " | ".join(["---"] * len(headers)) + " |\n"
                                )
                                for row in clean_table[1:]:
                                    full_text += "| " + " | ".join(row) + " |\n"
                                full_text += "\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        return f"Success: {pdf_path}"

    except Exception as e:
        return f"Error: {pdf_path} {str(e)}"


def main():
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "**/*.pdf"), recursive=True)
    print(f"Found {len(pdf_files)} PDF files in {INPUT_DIR}")

    for pdf_path in tqdm(pdf_files):
        parse_pdf_hybrid(pdf_path)


if __name__ == "__main__":
    main()
