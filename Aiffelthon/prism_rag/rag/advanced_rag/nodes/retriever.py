from langchain_core.documents import Document
from state import GraphState
from utils.vectorstore import vector_dbs, global_bm25_retriever, documents_db

def retriever(state: GraphState) -> dict:
    query = state["question"]
    target_fields = state["selected_fields"]
    date_filter = state.get("filter_date", "") 

    vector_k = 20
    bm25_k = 60 if date_filter else 20
    
    vector_hit_ids = set()
    bm25_hit_ids = set()
    newly_retrieved_docs = []

    for field in target_fields:
        if field in vector_dbs:
            try:
                results = vector_dbs[field].similarity_search(
                    query, 
                    k=vector_k,
                    expr=date_filter if date_filter else None
                )
                for doc in results:
                    idx = doc.metadata.get("idx")
                    if idx:
                        vector_hit_ids.add(idx)
            except Exception:
                pass
    
    global_bm25_retriever.k = bm25_k
    try:
        bm25_docs = global_bm25_retriever.invoke(query)
        for doc in bm25_docs:
            idx = doc.metadata.get("idx")
            if idx:
                bm25_hit_ids.add(idx)
    except Exception:
        pass

    all_target_ids = list(vector_hit_ids | bm25_hit_ids)
    
    if all_target_ids:
        try:
            collection = documents_db.col
            
            expr = f"idx in {all_target_ids}"
            if date_filter:
                expr = f"idx in {all_target_ids} and ({date_filter})"
            
            res = collection.query(expr=expr, output_fields=["*"], limit=len(all_target_ids))
            
            for item in res:
                text = item.get('text')
                idx = item.get('idx')
                
                is_vec = idx in vector_hit_ids
                is_bm25 = idx in bm25_hit_ids
                
                source_tag = "UNK"
                if is_vec and is_bm25:
                    source_tag = "BOTH"
                elif is_vec:
                    source_tag = "VEC"
                elif is_bm25:
                    source_tag = "BM25"
                
                metadata = item.copy()
                if 'text' in metadata: del metadata['text']
                if 'vector' in metadata: del metadata['vector']
                metadata['source'] = source_tag 
                
                if text:
                    newly_retrieved_docs.append(Document(page_content=text, metadata=metadata))
        except Exception:
            pass

    unique_docs_map = {doc.metadata.get("idx"): doc for doc in newly_retrieved_docs}
    unique_docs = list(unique_docs_map.values())
    
    if len(unique_docs) > 40:
        unique_docs = unique_docs[:40]
    
    vec_count = sum(1 for doc in unique_docs if doc.metadata.get('source') in ['VEC', 'BOTH'])
    bm25_count = sum(1 for doc in unique_docs if doc.metadata.get('source') in ['BM25', 'BOTH'])

    return {
        "documents": unique_docs,
        "search_stats": {
            "vector": vec_count,
            "bm25": bm25_count,
            "unique": len(unique_docs)
        }
    }