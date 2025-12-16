from docling.document_converter import DocumentConverter
import os


def test():
    # Target PDF
    pdf_path = "/Users/bychoi/develop/aura/aiffelthon/00_data/raw_data/1_bai_raw_files/3038-00.pdf"

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Converting {pdf_path} using Docling...")

    # Initialize Converter
    converter = DocumentConverter()

    # Convert
    result = converter.convert(pdf_path)

    # Export Markdown
    output_md = result.document.export_to_markdown()

    # Print first 2000 chars to check headers and spacing
    print("--- Docling Output Sample ---")
    print(output_md[:2000])
    print("---------------------------")

    # Save to file for inspection
    with open("docling_test_output.md", "w", encoding="utf-8") as f:
        f.write(output_md)
    print("Full result saved to docling_test_output.md")


if __name__ == "__main__":
    test()
