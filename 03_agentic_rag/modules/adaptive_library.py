from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from state import AgentState
from model_factory import ModelFactory

# --- 1. Grade Documents ---


class GradeResult(BaseModel):
    """Relevance check result"""

    reasoning: str = Field(description="Why the document is relevant or not")
    relevant: str = Field(description="'yes' or 'no'")


def grade_documents(state: AgentState) -> AgentState:
    """
    [Node] Retrieved documents grading.
    If document is irrelevant, we will rewrite the query.
    """
    print(f"\n[Node] grade_documents: 문서 적합성 평가 중...")

    docs = state.get("documents", [])
    if not docs or docs == ["검색 결과가 없습니다."]:
        print(" -> 문서 없음. 재검색 필요.")
        return {"documents": [], "grade_status": "fail"}

    # We only check the first few docs to save time, or concatenate them
    context_sample = "\n\n".join([str(d)[:500] for d in docs[:3]])
    query = state["query"]

    llm = ModelFactory.get_rag_model(level="light")  # Use light model for grading

    parser = JsonOutputParser(pydantic_object=GradeResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strict grader assessing the relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        
        Output JSON format:
        {{
            "reasoning": "...",
            "relevant": "yes" or "no"
        }}
        """,
            ),
            (
                "human",
                "Retrieved document: \n\n {context} \n\n User question: {question}",
            ),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"context": context_sample, "question": query})
        score = result["relevant"].lower()
        print(f" -> Grading Result: {score} ({result['reasoning']})")

        if score == "yes":
            return {"grade_status": "success"}
        else:
            return {"grade_status": "fail"}

    except Exception as e:
        print(f" -> Grading Error: {e}")
        # Default to success to avoid infinite loops on error, or fail if strictly needed
        return {"grade_status": "success"}


# --- 2. Rewrite Query ---


def rewrite_query(state: AgentState) -> AgentState:
    """
    [Node] Rewrite the query to improve retrieval.
    """
    print(f"\n[Node] rewrite_query: 질문 재작성 중...")

    query = state["query"]
    llm = ModelFactory.get_rag_model(level="light")

    msg = [
        (
            "system",
            "You are a helpful assistant that optimizes search queries. Look at the initial query and formulate a better query for a semantic search engine. Output ONLY the rewritten query string.",
        ),
        ("human", f"Initial Query: {query}"),
    ]

    response = llm.invoke(msg)
    new_query = response.content.strip()

    print(f" -> Originally: '{query}'")
    print(f" -> Rewritten: '{new_query}'")

    # Update the query for the next retrieval step
    # We might want to keep original query in history, but for simple loop we update 'sub_queries' or 'query'
    # Let's update 'sub_queries' to be safe for the retriever logic

    return {
        "sub_queries": [new_query],
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }
