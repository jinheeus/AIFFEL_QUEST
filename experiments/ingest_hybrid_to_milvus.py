import os
import glob
import json
import re
import argparse
from tqdm import tqdm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import MilvusClient, DataType
import concurrent.futures

# --- Detection Logic (Corrupted File Filter) ---
CORRUPTION_THRESHOLD = 0.05


def is_corrupted(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return True, 0.0  # Treat read errors as "skip"
    clean_text = re.sub(r"[\s\n\t\#\-\|]", "", text)
    if not clean_text:
        return True, 0.0
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", clean_text))
    hangul_count = len(
        re.findall(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", clean_text)
    )
    total_chars = len(clean_text)
    if total_chars == 0:
        return True, 0.0
    cjk_ratio = cjk_count / total_chars
    if cjk_ratio > CORRUPTION_THRESHOLD and hangul_count < cjk_count:
        return True, cjk_ratio
    return False, cjk_ratio


# --- Helper: Load Metadata ---
def load_metadata_resources(raw_root):
    print("📚 Loading Metadata Resources...")

    # 1. Load Mapping (IDX <-> Filename)
    mapping_path = os.path.join(raw_root, "id_to_filename_mapping_v2.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)["idx_to_meta"]

    # Invert Mapping: Filename -> [IDX, IDX, ...]
    filename_to_idxs = {}
    for idx_str, meta in mapping_data.items():
        fname = meta.get("filename")
        if not fname:
            continue
        if fname not in filename_to_idxs:
            filename_to_idxs[fname] = []
        filename_to_idxs[fname].append(int(idx_str))

    print(f" -> Mapped {len(filename_to_idxs)} filenames to IDXs.")

    # 2. Load Contents (Analysis Data)
    contents_path = os.path.join(raw_root, "contents.json")
    with open(contents_path, "r", encoding="utf-8") as f:
        contents_list = json.load(f)

    # Index Contents by IDX
    contents_by_idx = {item["idx"]: item for item in contents_list}
    print(f" -> Loaded {len(contents_by_idx)} content records.")

    return filename_to_idxs, contents_by_idx


def build_hybrid_context(filename_base, filename_to_idxs, contents_by_idx):
    """
    Constructs a rich context string from associated JSON metadata.
    """
    idxs = filename_to_idxs.get(filename_base, [])
    if not idxs:
        return "", []

    context_parts = []

    for idx in idxs:
        record = contents_by_idx.get(idx)
        if not record:
            continue

        # Format:
        # [Audit Case #1] Title: ...
        # Problems: ...
        # Action: ...
        block = f"[Audit Case #{idx}]\n"
        if record.get("title"):
            block += f"Title: {record['title']}\n"
        if record.get("problems"):
            block += f"Problems: {record['problems']}\n"
        # if record.get("action"): block += f"Action: {record['action']}\n" # Optional inclusion

        context_parts.append(block)

    return "\n---\n".join(context_parts), idxs


def _flush(client, collection, embeddings, batch):
    """
    Called within a ThreadPoolExecutor.
    NOTE: Embeddings are done synchronously on Main Thread (GPU) to avoid fork issues.
    This function handles the Network I/O (Insert).
    """
    # 1. Embed (This is GPU intensive, so we do it on Main Thread ideally,
    # but here we moved everything to helper to simplify threading structure.
    # Actually, BGE-M3/CUDA inside thread might be tricky.
    # BETTER PATTERN: Embed on Main, Insert on Thread.
    pass


# --- Main Ingestion ---
def ingest_hybrid(
    parsed_root, raw_root, collection_name, uri, token, batch_size=50, reset=False
):
    print(f"🚀 Starting HYBRID Ingestion to: {collection_name}")

    # 1. Prepare Resources
    filename_to_idxs, contents_by_idx = load_metadata_resources(raw_root)

    # 2. Milvus Init
    client = MilvusClient(uri=uri, token=token)

    if reset and client.has_collection(collection_name):
        print(f"🧹 Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)

    if client.has_collection(collection_name):
        print(f"⚠️ Collection {collection_name} exists. Appending...")
    else:
        print(f"🔨 Creating Collection {collection_name}...")
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field(
            "parent_text", DataType.VARCHAR, max_length=65535
        )  # Hybrid Context lives here
        schema.add_field("source_type", DataType.VARCHAR, max_length=100)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=200)
        # JSON field for related_idxs might be needed or dynamic.
        # Since enable_dynamic_field=True, extra fields like related_idxs can be inserted automatically if not in schema?
        # Ideally, explicit is better for query performance if filtered.
        # But 'related_idxs' is a list. Milvus JSON/Array support varies.
        # Dynamic field handles it as JSON.

        index_params = client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")

        client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

    # 3. Embedding Model
    print("🤖 Loading Embedding Model (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Setup ThreadPool for Async Insertion
    # We use a single worker to preserve order (though order doesn't strictly matter for RAG)
    # and to limit network congestion.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures = []

    def dispatch_insert(batch_to_insert):
        # This function runs in a thread.
        # It assumes 'batch_to_insert' already has 'vector' populated.
        try:
            client.insert(collection_name, batch_to_insert)
        except Exception as e:
            print(f"❌ Async Insert Error: {e}")

    # 4. Processing
    md_files = glob.glob(os.path.join(parsed_root, "**/*.md"), recursive=True)
    print(f"Found {len(md_files)} parsed files.")

    chunk_batch = []

    for md_path in tqdm(md_files, desc="Hybrid Ingestion"):
        # A. Filter Corruption
        if is_corrupted(md_path)[0]:
            continue

        try:
            filename = os.path.basename(md_path)
            filename_base = os.path.splitext(filename)[0]  # e.g. 2551-00

            # B. Build Hybrid Context
            hybrid_context, related_idxs = build_hybrid_context(
                filename_base, filename_to_idxs, contents_by_idx
            )

            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            source_type = "ALIO" if "alio" in md_path.lower() else "BAI"

            # C. Split & Merge
            # Splitter setup
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )

            parent_docs = markdown_splitter.split_text(content)

            for p_doc in parent_docs:
                p_text = p_doc.page_content

                # *** HYBRID MERGE ***
                # Prepend the JSON Context to the Parent Text
                # The LLM will see: [Audit Case Metadata] + [Actual PDF Content]
                enriched_parent_text = f"=== METADATA CONTEXT ===\n{hybrid_context}\n\n=== DOCUMENT CONTENT ===\n{p_text}"

                # Truncate to avoid Milvus VarChar limit (65535 bytes)
                # Korean char = 3 bytes, so 60k chars >> 65k bytes.
                # Safe limit: 65535 / 3 ~= 21845. Let's use 15000 to be safe.
                MAX_LEN = 15000
                if len(enriched_parent_text) > MAX_LEN:
                    enriched_parent_text = (
                        enriched_parent_text[:MAX_LEN] + "...(TRUNCATED)"
                    )

                child_chunks = text_splitter.split_text(
                    p_text
                )  # Chunk the *original* content, but link to enriched parent

                for c_text in child_chunks:
                    chunk_data = {
                        "text": c_text,
                        "parent_text": enriched_parent_text,  # Key for Hybrid RAG
                        "source_type": source_type,
                        "doc_id": filename,
                        # "related_idxs": related_idxs, # Check schema support if needed
                    }
                    chunk_batch.append(chunk_data)

                    if len(chunk_batch) >= batch_size:
                        # 1. Embed on Main Thread (GPU is here)
                        texts = [x["text"] for x in chunk_batch]
                        vectors = embeddings.embed_documents(texts)
                        for i, v in enumerate(vectors):
                            chunk_batch[i]["vector"] = v

                        # 2. Dispatch Insert to Thread (Network I/O)
                        # Copy batch to avoid mutation
                        batch_copy = list(chunk_batch)
                        futures.append(executor.submit(dispatch_insert, batch_copy))

                        chunk_batch = []

        except Exception as e:
            print(f"Error {filename}: {e}")

    # Final Batch
    if chunk_batch:
        texts = [x["text"] for x in chunk_batch]
        vectors = embeddings.embed_documents(texts)
        for i, v in enumerate(vectors):
            chunk_batch[i]["vector"] = v
        futures.append(executor.submit(dispatch_insert, chunk_batch))

    # Wait for all uploads
    print("⏳ Waiting for background uploads...")
    for f in concurrent.futures.as_completed(futures):
        pass  # Just wait

    executor.shutdown()
    print("✅ Hybrid Ingestion Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_root", default="./00_data/parsed_data")
    parser.add_argument("--raw_root", default="./00_data/raw_data")
    parser.add_argument("--collection_name", default="markdown_rag_hybrid_v1")
    parser.add_argument("--uri", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--reset", action="store_true", help="Drop collection before ingestion"
    )

    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for embedding/insertion"
    )

    args = parser.parse_args()
    ingest_hybrid(
        args.parsed_root,
        args.raw_root,
        args.collection_name,
        args.uri,
        args.token,
        batch_size=args.batch_size,
        reset=args.reset,
    )
