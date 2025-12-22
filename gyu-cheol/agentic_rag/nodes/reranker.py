from state import GraphState
from utils.models import encoder_model

def reranker(state: GraphState) -> dict:
    documents = state["documents"]
    validated_docs = state.get("validated_documents", [])
    
    current_valid_count = len(validated_docs)
    needed_count = 5 - current_valid_count
    
    if needed_count <= 0:
        return {"documents": []}
    
    if not documents:
        return {"documents": []}

    print(f"[ INFO ] RERANKING: SELECTING TOP {needed_count} (ALREADY HAVE {current_valid_count})")

    pairs = [[state["question"], doc.page_content] for doc in documents]
    scores = encoder_model.predict(pairs)
    
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    top_k_docs = [doc for doc, _ in scored_docs[:needed_count]]
    
    return {"documents": top_k_docs}