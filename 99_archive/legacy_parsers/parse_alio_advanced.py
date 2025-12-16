import fitz  # PyMuPDF
import os
import re
from collections import Counter

# Try importing spacing library, if available
try:
    from pykospacing import Spacing

    spacer = Spacing()
    HAS_SPACING_LIB = True
except ImportError:
    HAS_SPACING_LIB = False
    print("Warning: pykospacing not found. Spacing correction will be skipped.")

target_file = "00_data/raw_data/1_alio_raw_files/C0105/2021011902148051-01.pdf"
output_file = "test_local_advanced.md"


def get_font_size_histogram(doc):
    """Analyze font sizes to determine header structure."""
    font_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 0:  # text
                for l in b["lines"]:
                    for s in l["spans"]:
                        font_sizes.append(round(s["size"], 1))
    return Counter(font_sizes)


def parse_with_structure(pdf_path, output_path):
    print(f"--- Parsing with Structure & Spacing: {pdf_path} ---")
    doc = fitz.open(pdf_path)

    # 1. Analyze Font Sizes
    font_counter = get_font_size_histogram(doc)
    # Assume largest significant font (not singular outliers) is Title/H1
    # We sort by size desc
    sorted_fonts = sorted(font_counter.items(), key=lambda x: x[0], reverse=True)
    # Filter very rare large fonts (maybe logo text)
    valid_fonts = [size for size, freq in sorted_fonts if freq > 1]

    if not valid_fonts:
        base_size = 10.0
        h1_size = 14.0
        h2_size = 12.0
    else:
        # Heuristic:
        # Largest frequent = H1
        # Second largest = H2
        # Most frequent = Body (approx)
        h1_size = valid_fonts[0]
        h2_size = valid_fonts[1] if len(valid_fonts) > 1 else h1_size
        body_size = font_counter.most_common(1)[0][0]

    print(f"Detected Font Sizes -> H1: {h1_size}, H2: {h2_size}, Body: {body_size}")

    full_text = ""

    for page in doc:
        page_text = ""
        blocks = page.get_text("dict")["blocks"]

        # Sort blocks vertically
        blocks.sort(key=lambda b: b["bbox"][1])

        for b in blocks:
            if b["type"] == 0:
                block_text = ""
                block_size = 0.0
                count = 0

                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"]
                        size = s["size"]

                        # Accumulate average size for block to decide Header
                        block_size += size
                        count += 1

                        block_text += text
                    block_text += " "  # Space between spans in a line?

                if count > 0:
                    avg_size = block_size / count

                    # Apply Spacing Correction
                    if HAS_SPACING_LIB:
                        # Only apply if it looks like Korean and largely unspaced
                        # Simple check: valid korean chars / spaces
                        # But pykospacing is robust.
                        # We must strip spaces first to let pykospacing do its job?
                        # Or just pass it. PyKoSpacing fixes spacing errors.
                        # Warning: It is slow.
                        block_text = spacer(block_text)

                    # Apply Header Levels
                    prefix = ""
                    if avg_size >= h1_size - 0.5:  # Tolerance
                        prefix = "# "
                    elif avg_size >= h2_size - 0.5 and avg_size < h1_size:
                        prefix = "## "
                    elif avg_size > body_size + 1.0:
                        prefix = "### "

                    # Clean up
                    clean_line = block_text.strip()
                    if clean_line:
                        page_text += f"\n{prefix}{clean_line}\n"

        full_text += f"\n\n<!-- Page {page.number + 1} -->\n"
        full_text += page_text

        print(f"Processed Page {page.number + 1}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved to {output_file}")

    # Preview
    print("\n--- Preview (First 500 chars) ---")
    print(full_text[:500])


if __name__ == "__main__":
    parse_with_structure(target_file, output_file)
