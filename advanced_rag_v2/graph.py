from langgraph.graph import StateGraph, END
from state import GraphState
from nodes.date_extractor import date_extractor
from nodes.field_selector import field_selector
from nodes.retriever import retriever
from nodes.reranker import reranker
from nodes.generator import generator

workflow = StateGraph(GraphState)

workflow.add_node("date_extract", date_extractor)
workflow.add_node("field_select", field_selector)
workflow.add_node("retrieve", retriever)
workflow.add_node("rerank", reranker)
workflow.add_node("generate", generator)

workflow.set_entry_point("date_extract")
workflow.add_edge("date_extract", "field_select")
workflow.add_edge("field_select", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()