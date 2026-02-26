from langgraph.graph import StateGraph, END
from state import GraphState
from nodes.retriever import retriever
from nodes.generator import generator 

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retriever)
workflow.add_node("generate", generator)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()