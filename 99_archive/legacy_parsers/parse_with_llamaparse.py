import os
import glob
import time
from dotenv import load_dotenv
from llama_parse import LlamaParse
from tqdm import tqdm

# Load environment variables
load_dotenv()


def main():
    # 1. Configuration
    # Load all available API KEYs
    api_keys = []

    # Check for the main key
    # main_key = os.getenv("LLAMA_CLOUD_API_KEY")
    # if main_key:
    #     api_keys.append(main_key)

    # Check for additional keys (LLAMA_CLOUD_API_KEY_2, _3, ...)
    for i in range(3, 20):  # Check up to 20 keys
        key = os.getenv(f"LLAMA_CLOUD_API_KEY_{i}")
        if key:
            api_keys.append(key)

    if not api_keys:
        print("Error: No LLAMA_CLOUD_API_KEY found in .env files.")
        return

    print(f"Loaded {len(api_keys)} API Keys.")
    current_key_index = 0

    input_dir = "00_data/raw_data/1_bai_raw_files/"
    output_dir = "00_data/parsed_data/llama_parse/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Get Files
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")

    # Helper to get current parser
    def get_parser(key_idx):
        return LlamaParse(
            api_key=api_keys[key_idx],
            result_type="markdown",
            verbose=True,
            language="ko",
        )

    # 3. Initialize Parser
    parser = get_parser(current_key_index)

    # Signal handler for timeout
    import signal

    def handler(signum, frame):
        raise TimeoutError("Parsing timed out")

    signal.signal(signal.SIGALRM, handler)

    # 4. Batch Process
    results = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "empty": 0,
        "timeout": 0,
        "quota_exceeded": 0,
    }

    print("Starting batch parsing with Key Rotation...")

    files_count = len(pdf_files)
    for i, pdf_file in enumerate(pdf_files):
        try:
            file_name = os.path.basename(pdf_file)
            print(
                f"[{i + 1}/{files_count}] Processing: {file_name} (Key #{current_key_index + 1})"
            )

            file_base = os.path.splitext(file_name)[0]
            output_file = os.path.join(output_dir, f"{file_base}.md")

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"  -> Skipping (Already exists)")
                results["skipped"] += 1
                continue

            # Retry loop for checking quota/timeout
            max_retries = len(api_keys)  # Try each key once for a file if quota fails
            attempts = 0
            success = False

            while attempts < max_retries and not success:
                # Set alarm for timeout
                signal.alarm(1800)  # 30 mins
                try:
                    # Sync parse
                    documents = parser.load_data(pdf_file)
                    signal.alarm(0)  # Disable alarm

                    if documents:
                        full_text = "\n\n".join([doc.text for doc in documents])
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(full_text)
                        print(f"  -> Success")
                        results["success"] += 1
                        success = True
                    else:
                        print(f"  -> Warning: No content extracted")
                        results["empty"] += 1
                        success = True  # Treated as success (not an API error)

                except TimeoutError:
                    print(f"  -> Timeout (skipped file)")
                    results["timeout"] += 1
                    success = True  # Give up on this file, move to next
                    # Don't rotate key for timeout, it's a file issue

                except Exception as e:
                    signal.alarm(0)  # Disable alarm
                    error_msg = str(e)
                    # Check for Quota/Credit errors
                    if (
                        "429" in error_msg
                        or "402" in error_msg
                        or "Credit" in error_msg
                        or "Quota" in error_msg
                    ):
                        print(
                            f"  -> Quota Exceeded on Key #{current_key_index + 1}: {e}"
                        )

                        # Rotate Key
                        current_key_index += 1
                        if current_key_index >= len(api_keys):
                            print("  -> ALL KEYS EXHAUSTED. Stopping.")
                            return

                        print(f"  -> Switching to Key #{current_key_index + 1}")
                        parser = get_parser(current_key_index)
                        attempts += 1  # Try again with new key
                    else:
                        print(f"  -> Error: {e}")
                        results["error"] += 1
                        success = True  # Give up on this file
                        time.sleep(1)

        except Exception as e:  # Outer Catch
            print(f"  -> Critical Error: {e}")
            # This outer catch is for errors that prevent processing of the current file
            # and might indicate a more severe issue.
            # The original code had a `break` here, but the instruction snippet
            # seems to imply a `time.sleep(1)` for general errors.
            # I'll keep the `time.sleep(1)` here for consistency with the original intent
            # of handling general errors, but note that the instruction snippet was incomplete.
            time.sleep(1)

    # 5. Summary
    print("\n" + "=" * 30)
    print("Processing Complete")
    print(f"Total Files: {len(pdf_files)}")
    print(f"Success: {results['success']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Errors: {results['error']}")
    print(f"Empty: {results['empty']}")
    print("=" * 30)


if __name__ == "__main__":
    main()
