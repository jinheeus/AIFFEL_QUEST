from typing import List, TypedDict, Dict, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    filter_date: str
    selected_fields: List[str] 
    selected_fields_cot: List[str]
    documents: List[Document]
    search_stats: Dict[str, int]
    answer: Optional[str]
    answer_cot: List[str]