import pandas as pd
import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection
from config import Config
import ast
import json

# Force MPS if available
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

print(f"Using Device: {device}")


def get_embedding_model():
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


def main():
    # 1. Load Queries
    csv_path = "retrieval.csv"
    if not os.path.exists(csv_path):
        print("retrieval.csv not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} queries from {csv_path}")

    # 2. Connect to Milvus
    connections.connect("default", uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
    collection_name = Config.MILVUS_COLLECTION_NAME_MARKDOWN
    collection = Collection(collection_name)
    collection.load()
    print(
        f"Connected to Collection: {collection_name} (Rows: {collection.num_entities})"
    )

    # 3. Embedding Model
    embed_model = get_embedding_model()

    # 4. Run Retrieval
    retrieved_results = []

    # We need to fill 'documents' column with list of dicts: [{'content':..., 'metadata':...}]
    # The evaluation notebook expects 'documents' column to be a list of maps.

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    for idx, row in df.iterrows():
        query = row["question"]
        if pd.isna(query):
            retrieved_results.append([])
            continue

        # Embed
        query_vec = embed_model.embed_query(query)

        # Search
        results = collection.search(
            data=[query_vec],
            anns_field="vector",
            param=search_params,
            limit=5,
            output_fields=[
                "text",
                "parent_text",
                "file_name",
                "source_type",
                "breadcrumbs",
            ],
        )

        # Format
        docs = []
        for hits in results:
            for hit in hits:
                # Format as dict for the evaluation script
                doc_content = {
                    "content": hit.entity.get("text"),
                    "file": hit.entity.get("file_name"),
                    "source": hit.entity.get("source_type"),
                    "breadcrumbs": hit.entity.get("breadcrumbs"),
                }
                docs.append(doc_content)

        retrieved_results.append(docs)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} queries...")

    # 5. Save Results
    df["documents"] = retrieved_results
    output_path = "retrieval_with_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved retrieval results to {output_path}")


if __name__ == "__main__":
    main()
