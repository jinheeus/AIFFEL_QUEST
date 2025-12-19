from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Literal

from model_factory import ModelFactory


# --- LLM 초기화 ---
# 평가(Evaluation) 모델은 구조화된 출력(Structured Output)을 잘 지원하는 모델(GPT-4o-mini 등)을 사용합니다.
llm = ModelFactory.get_eval_model(level="light", temperature=0)


# --- 1. 문서 평가기 (Retrieval Grader) ---
class GradeRetrieval(BaseModel):
    """검색된 문서의 관련성(Relevance) 점수."""

    binary_score: str = Field(
        description="문서가 질문과 관련이 있는지 여부, 'yes' 또는 'no'"
    )


retrieval_grader_system = """[Role]
You are a grader assessing relevance of a retrieved document to a user question.

[Goal]
Check if the document contains keywords or semantic meaning related to the user question.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

[Output]
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader_chain = retrieval_grader_prompt | llm.with_structured_output(
    GradeRetrieval
)


def grade_documents(question: str, documents: List[Document]) -> dict:
    """
    검색된 문서들의 관련성(Relevance)을 평가합니다.
    """
    print("--- [Modular RAG] Grading Documents ---")

    filtered_docs = []
    relevant_found = False

    for d in documents:
        # Compatibility handling
        if isinstance(d, str):
            content = d
            doc_id = "unknown"
        else:
            content = d.page_content
            content = d.page_content
            # Try to get meaningful ID or snippet
            doc_id = (
                d.metadata.get("source")
                or d.metadata.get("doc_id")
                or content[:30] + "..."
            )

        score = retrieval_grader_chain.invoke(
            {"question": question, "document": content}
        )
        grade = score.binary_score

        if grade == "yes":
            print(f" -> Document Relevant: {doc_id}")
            filtered_docs.append(d)
            relevant_found = True
        else:
            print(f" -> Document Irrelevant: {doc_id}")

    return {
        "documents": filtered_docs,
        "is_retrieval_success": "yes" if relevant_found else "no",
    }


# --- 2. 환각 평가기 (Hallucination Grader) ---
class GradeHallucinations(BaseModel):
    """답변의 환각 여부(Groundedness) 점수."""

    binary_score: str = Field(
        description="답변이 사실(Facts)에 기반하는지 여부, 'yes' 또는 'no'"
    )


hallucination_grader_system = """[Role]
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

[Goal]
Check if the answer is grounded in the provided documents.
'yes' means the answer is fully supported by the documents.
'no' means the answer contains information not present in the documents (hallucination).

[Output]
Give a binary score 'yes' or 'no'."""

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader_chain = hallucination_grader_prompt | llm.with_structured_output(
    GradeHallucinations
)


def grade_hallucination(generation: str, documents: List[Document]) -> str:
    """
    생성된 답변이 문서에 근거(Grounded)하고 있는지 확인합니다.
    """
    print("--- [Modular RAG] Grading Hallucination ---")

    # Context format
    # Handle both string and Document objects
    docs_text = []
    for d in documents:
        if isinstance(d, str):
            docs_text.append(d)
        else:
            docs_text.append(d.page_content)

    context = "\n\n".join(docs_text)

    score = hallucination_grader_chain.invoke(
        {"documents": context, "generation": generation}
    )
    return score.binary_score


# --- 3. 답변 유용성 평가기 (Answer Grader) ---
class GradeAnswer(BaseModel):
    """답변이 질문을 해결했는지(Utility) 점수."""

    binary_score: str = Field(
        description="답변이 질문을 다루고/해결하고 있는지 여부, 'yes' 또는 'no'"
    )


answer_grader_system = """[Role]
You are a grader assessing whether an answer addresses / resolves a question.

[Goal]
Ensure the answer actually responds to the user's intent.

[Output]
Give a binary score 'yes' or 'no'."""

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader_chain = answer_grader_prompt | llm.with_structured_output(GradeAnswer)


def grade_answer(question: str, generation: str) -> str:
    """
    답변이 사용자에게 유용한지(Helpfulness) 확인합니다.
    """
    print("--- [Modular RAG] Grading Answer Utility ---")

    score = answer_grader_chain.invoke({"question": question, "generation": generation})
    return score.binary_score
