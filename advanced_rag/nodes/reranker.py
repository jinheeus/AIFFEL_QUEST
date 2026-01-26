from state import GraphState
from utils.models import encoder_model

def reranker(state: GraphState) -> dict:
    documents = state["documents"]
    if not documents:
        return {"documents": []}

    pairs = [[state["question"], doc.page_content] for doc in documents]
    scores = encoder_model.predict(pairs)
    
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return {"documents": [doc for doc, _ in scored_docs[:5]]}