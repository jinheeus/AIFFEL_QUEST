import fitz  # PyMuPDF
import pdfplumber
import os

target_file = "00_data/raw_data/1_alio_raw_files/C0105/2021011902148051-01.pdf"
output_pymupdf = "test_pymupdf.md"
output_pdfplumber = "test_pdfplumber.md"


def test_pymupdf(pdf_path, output_path):
    print(f"--- Testing PyMuPDF on {pdf_path} ---")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        # get_text("markdown") is available in recent versions, otherwise "text"
        # We'll try basic text block extraction for structure approximation
        text += f"\n\n# Page {page.number + 1}\n\n"
        text += page.get_text("text")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved PyMuPDF result to {output_path}")
    # Print first 500 chars
    print(text[:500])


def test_pdfplumber(pdf_path, output_path):
    print(f"\n--- Testing pdfplumber on {pdf_path} ---")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text += f"\n\n# Page {i + 1}\n\n"
            extracted = page.extract_text()
            if extracted:
                text += extracted
            else:
                text += "(No text extracted)"

            # Simple table check
            tables = page.extract_tables()
            if tables:
                text += "\n\n[Found Table]\n"
                for table in tables:
                    for row in table:
                        text += " | ".join([str(c) if c else "" for c in row]) + "\n"
                    text += "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved pdfplumber result to {output_path}")
    # Print first 500 chars
    print(text[:500])


if __name__ == "__main__":
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
    else:
        try:
            test_pymupdf(target_file, output_pymupdf)
        except Exception as e:
            print(f"PyMuPDF failed: {e}")

        try:
            test_pdfplumber(target_file, output_pdfplumber)
        except Exception as e:
            print(f"pdfplumber failed: {e}")
