from typing import List, TypedDict, Dict, Any, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    original_question: str
    question: str
    rewrite_cot: List[str]
    filter_date: str
    selected_fields: List[str] 
    selected_fields_cot: List[str]
    documents: List[Document]
    search_stats: Dict[str, int]
    validated_documents: List[Document]
    validation_results: List[Dict[str, Any]] 
    retry_count: int
    answer: Optional[str]
    answer_cot: List[str]