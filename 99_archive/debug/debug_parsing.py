import os
import glob
from dotenv import load_dotenv
from llama_parse import LlamaParse
import time

# Load environment variables
load_dotenv()


def debug_main():
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("❌ Error: LLAMA_CLOUD_API_KEY not found.")
        return

    print(f"✅ API Key found: {api_key[:5]}...")

    # Use the file known to work
    test_file = "00_data/raw_data/1_bai_raw_files/2538-00.pdf"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        # Try finding ANY pdf
        pdfs = glob.glob("00_data/raw_data/1_bai_raw_files/*.pdf")
        if pdfs:
            test_file = pdfs[0]
            print(f"⚠️ Using alternative file: {test_file}")
        else:
            print("❌ No PDF files found.")
            return

    print(f"🚀 Initializing LlamaParse (Sync Mode)...")
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True,  # ENABLE VERBOSE
        language="ko",
    )

    print(f"📂 Starting to parse: {test_file}")
    start_time = time.time()

    try:
        # PURE SYNC CALL - No asyncio, no nest_asyncio
        documents = parser.load_data(test_file)

        end_time = time.time()
        print(f"✅ Parse Complete in {end_time - start_time:.2f} seconds")

        if documents:
            print(f"📄 Generated {len(documents)} documents")
            print(f"📝 Content Preview: {documents[0].text[:200]}...")

            # Try saving
            with open("debug_output.md", "w", encoding="utf-8") as f:
                f.write(documents[0].text)
            print("💾 Saved to debug_output.md")
        else:
            print("⚠️ Parsed successfully but no documents returned.")

    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_main()
