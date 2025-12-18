import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Import the pipeline class from the standalone script (assuming it's in the same dir)
# We need to adapt this because we can't easily import from the standalone script if it's not a module.
# proper way: I will copy the minimal logic here to ensure I'm testing the SAME code structure.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import MilvusClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from statistics import mean


class Config:
    MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_demo.db")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEVICE = "cpu"  # Force CPU for local debug to avoid CUDA errors if not available


def debug_run():
    print("🚀 Starting Local Debug...")

    # 1. Test OpenAI Connection
    if not Config.OPENAI_API_KEY:
        print("❌ OpenAI API Key is MISSING in environment!")
        return
    print("✅ OpenAI API Key found.")

    # 2. Test Milvus Connection
    print(f"🔌 Connecting to Milvus: {Config.MILVUS_URI}")
    try:
        client = MilvusClient(uri=Config.MILVUS_URI, token=Config.MILVUS_TOKEN)
        # Just check connection by describing collection or simple query
        res = client.query(
            collection_name="audit_rag_hybrid_v1",
            filter="id >= 0",
            output_fields=["doc_id"],
            limit=1,
        )
        print("✅ Milvus Connected. Sample Query Success.")
    except Exception as e:
        print(f"❌ Milvus Connection Failed: {e}")
        return

    # 3. Initialize Pipeline Components (Mocking the StandalonePipeline structure)
    print("⚙️ Loading Embedding Model (BGE-M3)...")
    try:
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": Config.DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ Embedding Model Loaded.")
    except Exception as e:
        print(f"❌ Embedding Model Failed: {e}")
        return

    # 4. Perform Search
    query = "출장비 부정 수령 사례"
    print(f"🔎 Searching for: '{query}'")

    try:
        q_vec = embedding.embed_query(query)
        res = client.search(
            collection_name="audit_rag_hybrid_v1",
            data=[q_vec],
            limit=3,
            output_fields=["text", "parent_text", "doc_id"],
        )
        docs = [h["entity"]["text"] for h in res[0]]
        print(f"✅ Retrieved {len(docs)} documents.")
        if not docs:
            print("⚠️ No documents found! This explains the 0.0 score.")
            return
        print(f"   Sample Doc: {docs[0][:100]}...")
    except Exception as e:
        print(f"❌ Search Failed: {e}")
        return

    # 5. Test LLM Evaluation (The likely culprit for 0.0 if search works)
    print("⚖️ Testing LLM Evaluation...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)

    system_prompt = """
    당신은 감사문서 기반 질문-문서 유사도 평가 전문가입니다.
    ... (Simplified for Debug) ...
    [출력 형식]
    반드시 JSON 형식으로만 출력: {{"topic_match": {{"decision": true, "reason": "..."}}}}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "[Question]\n{question}\n\n[Document]\n{document}"),
        ]
    )
    chain = prompt | llm

    try:
        print("   Invoking LLM...")
        res = chain.invoke({"question": query, "document": docs[0]})
        print(f"   LLM Raw Output:\n{res.content}")

        # Parse Check
        content = res.content.replace("```json", "").replace("```", "").strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            content = content[start:end]

        data = json.loads(content)
        print("✅ JSON Parsing Success:", data.keys())

    except Exception as e:
        print(f"❌ LLM Evaluation Failed: {e}")


if __name__ == "__main__":
    debug_run()
