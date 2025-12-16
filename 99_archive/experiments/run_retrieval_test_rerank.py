import pandas as pd
import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection
from sentence_transformers import CrossEncoder
from config import Config

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

    # 3. Models
    embed_model = get_embedding_model()

    # Reranker (Same as pipeline_v2.py)
    try:
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        print("Loaded Reranker: BAAI/bge-reranker-v2-m3")
    except:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Loaded Fallback Reranker: ms-marco-MiniLM-L-6-v2")

    # 4. Run Retrieval + Rerank
    retrieved_results = []

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    # Top-K settings
    SEARCH_K = 50  # Retrieve 50 candidates
    TOP_K = 5  # Rerank and select Top 5

    for idx, row in df.iterrows():
        query = row["question"]
        if pd.isna(query):
            retrieved_results.append([])
            continue

        # Embed
        query_vec = embed_model.embed_query(query)

        # Search (Top-50)
        results = collection.search(
            data=[query_vec],
            anns_field="vector",
            param=search_params,
            limit=SEARCH_K,
            output_fields=[
                "text",
                "parent_text",
                "file_name",
                "source_type",
                "breadcrumbs",
            ],
        )

        candidates = []
        hits = results[0]  # Single query

        for hit in hits:
            # Prepare text for reranker: Query + (Breadcrumbs + ParentText)
            breadcrumbs = hit.entity.get("breadcrumbs") or ""
            parent_text = hit.entity.get("parent_text") or ""
            child_text = hit.entity.get("text") or ""

            # Using same logic as pipeline_v2: prioritizing parent_text context
            context_text = parent_text if parent_text else child_text
            formatted_text = f"[Path: {breadcrumbs}]\n{context_text}"

            candidates.append({"formatted_text": formatted_text, "hit": hit})

        if not candidates:
            retrieved_results.append([])
            continue

        # Rerank
        pairs = [[query, c["formatted_text"]] for c in candidates]
        scores = reranker.predict(pairs)

        # Sort
        scored_candidates = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        # Select Top-K
        final_docs = []
        for cand, score in scored_candidates[:TOP_K]:
            hit = cand["hit"]
            doc_content = {
                "content": hit.entity.get(
                    "text"
                ),  # Evaluator still sees Child Text? Or Parent?
                # Evaluation metric checks "Does this document verify the answer?"
                # If we return Child Text, it might verify specific facts.
                # If we return Parent Text, it verifies broader context.
                # Let's return Parent Text as "content" to help the Evaluator see the full context
                # (since that's what we feed the LLM).
                # BUT wait, the previous run used "text" (Child).
                # If we switch to "parent_text" here, the score might jump just because of context window.
                # Let's stick to what the RAG pipeline provides.
                # Pipeline provides `parent_text` (or context_text).
                # So let's use `parent_text` for evaluation content.
                "content": hit.entity.get("parent_text") or hit.entity.get("text"),
                "file": hit.entity.get("file_name"),
                "source": hit.entity.get("source_type"),
                "breadcrumbs": hit.entity.get("breadcrumbs"),
                "rerank_score": float(score),
            }
            final_docs.append(doc_content)

        retrieved_results.append(final_docs)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} queries...")

    # 5. Save Results
    df["documents"] = retrieved_results
    output_path = "retrieval_with_results.csv"  # Overwrite same file for scoring script
    df.to_csv(output_path, index=False)
    print(f"Saved Reranked results to {output_path}")


if __name__ == "__main__":
    main()
