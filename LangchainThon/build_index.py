import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from rag.vectorstore import get_vectorstore

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")


def load_all_pdfs(pdf_dir):
    docs = []
    for name in os.listdir(pdf_dir):
        if not name.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, name)
        loader = PyPDFLoader(path)

        pdf_docs = loader.load()

        for d in pdf_docs:
            d.metadata["source_pdf"] = name

        docs.extend(pdf_docs)

    return docs


def build_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(docs)


def main():
    print(f"📂 Loading PDFs from: {DOCS_DIR}")
    docs = load_all_pdfs(DOCS_DIR)

    print(f"➡ documents loaded: {len(docs)}")

    chunks = build_chunks(docs)
    print(f"✂ chunks created: {len(chunks)}")

    vs = get_vectorstore(CHROMA_DIR)

    print("🧠 Saving embeddings to Chroma...")
    vs.add_documents(chunks)

    print("✅ DONE. Vector DB ready at:", CHROMA_DIR)


if __name__ == "__main__":
    main()
