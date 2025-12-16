import os
import glob
import re
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime

# LangChain / Embedding
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Milvus
from pymilvus import MilvusClient, DataType

# --- Detection Logic (Copied from fix_parsing_errors.py) ---
CORRUPTION_THRESHOLD = (
    0.05  # If > 5% of non-whitespace chars are CJK Ideographs (Hanja/Chinese)
)


def is_corrupted(file_path):
    """
    Detects if a markdown file is likely corrupted by bad OCR (Chinese hallucination).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return True, 0.0  # Treat read errors as "skip"

    # Remove whitespace and common markdown syntax
    clean_text = re.sub(r"[\s\n\t\#\-\|]", "", text)
    if not clean_text:
        return True, 0.0  # Empty file

    # Count CJK Unified Ideographs (Includes Hanja and Chinese)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", clean_text))
    # Count Hangul (Syllables + Jamo)
    hangul_count = len(
        re.findall(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", clean_text)
    )

    total_chars = len(clean_text)
    if total_chars == 0:
        return True, 0.0

    cjk_ratio = cjk_count / total_chars

    # Skip if high CJK ratio AND low Hangul count
    if cjk_ratio > CORRUPTION_THRESHOLD and hangul_count < cjk_count:
        return True, cjk_ratio

    return False, cjk_ratio


# --- Ingestion Logic ---


def ingest_to_milvus(
    input_dir: str, collection_name: str, uri: str, token: str, batch_size: int = 50
):
    print(f"🚀 Starting Ingestion to Milvus Collection: {collection_name}")
    print(f"📂 Scanning Directory: {input_dir}")

    # 1. Initialize Milvus Client
    client = MilvusClient(uri=uri, token=token)  # Cloud connection

    # 2. Define Schema (Parent-Child)
    if client.has_collection(collection_name):
        print(
            f"⚠️ Collection {collection_name} exists. Appending data (or drop first if needed)."
        )
    else:
        print(f"🔨 Creating Collection {collection_name}...")
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024
        )  # BGE-M3 dim
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="parent_text", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="source_type", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=200)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )

        client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

    # 3. Initialize Embedding Model (Local GPU)
    print("🤖 Loading Embedding Model (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},  # T4 GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Process Files
    md_files = glob.glob(os.path.join(input_dir, "**/*.md"), recursive=True)
    print(f"Found {len(md_files)} Markdown files.")

    chunk_batch = []
    skipped_count = 0

    for md_path in tqdm(md_files, desc="Ingesting"):
        # A. Check Corruption
        bad_file, score = is_corrupted(md_path)
        if bad_file:
            # print(f"🚫 Skipping Corrupted File: {md_path} (Score: {score:.2f})")
            skipped_count += 1
            continue

        # B. Read & Extract Metadata
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract basic metadata from path
            # structure: .../alio_docling/C0105/filename.md
            parts = md_path.split(os.sep)
            filename = os.path.basename(md_path)

            # Simple heuristic for source_type
            source_type = "UNKNOWN"
            if "alio" in md_path.lower():
                source_type = "ALIO"
            elif "bai" in md_path.lower():
                source_type = "BAI"

            # C. Split (Parent-Child)
            # Level 1: Headers (Parent)
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            md_header_splits = markdown_splitter.split_text(content)

            # Level 2: Recursive Token Split (Child)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )

            for parent_doc in md_header_splits:
                parent_text = parent_doc.page_content
                parent_meta = parent_doc.metadata
                breadcrumbs = str(parent_meta)  # Simple string representation

                child_chunks = text_splitter.split_text(parent_text)

                for child_text in child_chunks:
                    # Enrich Metadata
                    chunk_data = {
                        "text": child_text,
                        "parent_text": parent_text,  # Context for LLM
                        "source_type": source_type,
                        "doc_id": filename,
                        # "breadcrumbs": breadcrumbs # Optional dynamic field
                    }
                    chunk_batch.append(chunk_data)

                    # D. Flush Batch
                    if len(chunk_batch) >= batch_size:
                        _flush_batch(client, collection_name, embeddings, chunk_batch)
                        chunk_batch = []

        except Exception as e:
            print(f"Error processing {md_path}: {e}")
            continue

    # Final Flush
    if chunk_batch:
        _flush_batch(client, collection_name, embeddings, chunk_batch)

    print(f"✅ Ingestion Complete. Skipped {skipped_count} corrupted files.")


def _flush_batch(client, collection_name, embeddings, batch_data):
    texts = [item["text"] for item in batch_data]
    try:
        vectors = embeddings.embed_documents(texts)

        insert_list = []
        for i, item in enumerate(batch_data):
            record = item.copy()
            record["vector"] = vectors[i]
            insert_list.append(record)

        client.insert(collection_name=collection_name, data=insert_list)

    except Exception as e:
        print(f"Error inserting batch: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./00_data/parsed_data")
    parser.add_argument("--collection_name", default="markdown_rag_parent_child_v1")
    # T4 Environment usually provides these via ENV or args.
    # Providing placeholders for user to fill or pass via CLI.
    parser.add_argument("--uri", required=True, help="Milvus Public Endpoint URI")
    parser.add_argument("--token", required=True, help="Milvus API Token")

    args = parser.parse_args()

    ingest_to_milvus(args.input_dir, args.collection_name, args.uri, args.token)
