import pdfplumber
import os

target_file = "00_data/raw_data/1_alio_raw_files/C0105/2021011902148051-01.pdf"
output_file = "test_table_extract.md"


def extract_tables(pdf_path, output_path):
    print(f"--- Testing pdfplumber Table Extraction on {pdf_path} ---")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text += f"\n\n# Page {i + 1}\n\n"

                # Check for tables
                tables = page.extract_tables()
                if tables:
                    text += f"\n<!-- Tables found: {len(tables)} -->\n"
                    for table in tables:
                        # Convert to Markdown Table
                        if not table:
                            continue

                        # Filter out None
                        clean_table = [
                            [str(cell or "").replace("\n", " ") for cell in row]
                            for row in table
                        ]

                        # Header
                        if len(clean_table) > 0:
                            headers = clean_table[0]
                            text += "| " + " | ".join(headers) + " |\n"
                            text += "| " + " | ".join(["---"] * len(headers)) + " |\n"

                            for row in clean_table[1:]:
                                text += "| " + " | ".join(row) + " |\n"
                            text += "\n"
                else:
                    text += "(No tables found on this page)\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved table extraction result to {output_path}")
        print(text[:500])

    except Exception as e:
        print(f"pdfplumber failed: {e}")


if __name__ == "__main__":
    if os.path.exists(target_file):
        extract_tables(target_file, output_file)
    else:
        print("File not found")
