from langchain_core.documents import Document
from state import GraphState
from utils.vectorstore import vector_db, documents_db

def retriever(state: GraphState) -> dict:
    query = state["question"]

    target_k = 5
    
    hit_ids = []
    
    try:
        results = vector_db.similarity_search(
            query, 
            k=target_k
        )
        
        for doc in results:
            idx = doc.metadata.get("idx")
            if idx and idx not in hit_ids:
                hit_ids.append(idx)
                
    except Exception:
        pass

    final_docs = []

    if hit_ids:
        try:
            collection = documents_db.col
            
            expr = f"idx in {hit_ids}"
            res = collection.query(expr=expr, output_fields=["*"], limit=len(hit_ids))
            
            res_map = {item.get('idx'): item for item in res}
            
            for idx in hit_ids:
                if idx in res_map:
                    item = res_map[idx]
                    text = item.get('text')
                    
                    metadata = item.copy()
                    if 'text' in metadata: del metadata['text']
                    if 'vector' in metadata: del metadata['vector']
                    
                    metadata['source'] = "VEC"
                    
                    if text:
                        final_docs.append(Document(page_content=text, metadata=metadata))
                        
        except Exception:
            pass

    unique_docs_map = {doc.metadata.get("idx"): doc for doc in final_docs}
    unique_docs = list(unique_docs_map.values())
    
    return {
        "documents": unique_docs,
        "search_stats": {
            "vector": len(unique_docs),
            "unique": len(unique_docs)
        }
    }