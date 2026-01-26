from typing import List, TypedDict, Dict, Any, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    search_stats: Dict[str, Any]
    answer: Optional[str]
    answer_cot: Optional[List[str]]