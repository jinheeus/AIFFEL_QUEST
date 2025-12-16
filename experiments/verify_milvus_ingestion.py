import sys
import os
from pymilvus import MilvusClient

# Add root directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

URI = Config.MILVUS_URI
TOKEN = Config.MILVUS_TOKEN
# Looking at previous files, user provided URI/TOKEN via args usually.
# I'll rely on hardcoded URI from previous notebooks or env vars if set.
# Wait, I should probably check ingest_hybrid_to_milvus.py args or use the ones from there.
# Let's try to get them from env first, if not, I might need to ask or look at previous `ingest_hybrid_to_milvus.py` calls to see if I can find the values.
# Actually, I'll assume they are in .env or I can use the values I saw in the notebook earlier?
# The user passed them as args in previous runs. I don't have them in my history explicitly as env vars.
# However, I can try to read `.env` if it exists.

# Let's try to read explicit values if I can find them.
# Re-reading `agentic_rag_code.ipynb` showed a URI: "https://in03-6919b557b41d797.serverless.gcp-us-west1.cloud.zilliz.com"
# But TOKEN was empty string in that notebook snippet.
# User previously ran `ingest_hybrid_to_milvus.py` with arguments.
# I will try to use the URI from the notebook and a placeholder for token, but likely I need the token.
# Let's check `03_agentic_rag/.env` or similar if it exists.
# Or better, I will just write a script that expects arguments, and run it with the arguments if I can find them, or ask user.
# Wait, I can see `run_app.sh` or similar to see how env vars are loaded?
# Let's look for `.env` file first.

# For now, I will create a script that TAKES arguments.
import argparse


def verify_milvus(collection_name):
    print(f"Connecting to Milvus: {URI}")
    client = MilvusClient(uri=URI, token=TOKEN)

    if not client.has_collection(collection_name):
        print(f"❌ Collection {collection_name} NOT FOUND!")
        return

    print(f"✅ Collection {collection_name} exists.")

    # Get Count
    res = client.query(
        collection_name=collection_name, filter="", output_fields=["count(*)"]
    )
    # query output for count(*) depends on version, usually it returns count.
    # Actually client.query with count(*) is specific.
    # Better: client.get_collection_stats() or just `client.query` with limit 1 and `output_fields=["count(*)"]` is not standard in pymilvus generic client sometimes.
    # The simplest way to count in standard MilvusClient is `client.query(..., output_fields=["count(*)"])` if supported, or `num_entities` property.

    # In milvus-lite / serverless, num_entities might be approximate.
    # Let's try getting collection info.
    desc = client.describe_collection(collection_name)
    print(f"Collection Info: {desc}")

    # Query 3 random items
    print("\n🔍 Sampling 3 items...")
    results = client.query(
        collection_name=collection_name,
        filter="id >= 0",
        output_fields=["doc_id", "source_type", "parent_text", "text"],
        limit=3,
    )

    for i, res in enumerate(results):
        print(f"\n[Item {i}]")
        print(f" - ID: {res.get('id')}")
        print(f" - Doc ID: {res.get('doc_id')}")
        print(f" - Source: {res.get('source_type')}")
        pt = res.get("parent_text", "")
        print(f" - Parent Text ({len(pt)} chars): {pt[:200]} ...")
        print(
            f" - Child Text ({len(res.get('text', ''))} chars): {res.get('text', '')[:100]} ..."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--uri", required=True)
    # parser.add_argument("--token", required=True)
    parser.add_argument("--collection_name", default="markdown_rag_hybrid_v1")
    args = parser.parse_args()

    verify_milvus(args.collection_name)
