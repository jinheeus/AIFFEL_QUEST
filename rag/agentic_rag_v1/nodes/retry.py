from typing import Literal
from state import GraphState

def retry(state: GraphState) -> Literal["retry", "end"]:
    valid_docs = state.get("validated_documents", [])
    current_count = len(valid_docs)
    retry_count = state.get("retry_count", 0)

    print(f"\n[ CHECK ] VALID_DOCS_COLLECTED: {current_count} / 5")

    if current_count >= 5 or retry_count >= 3:
        return "end"
    
    return "retry"