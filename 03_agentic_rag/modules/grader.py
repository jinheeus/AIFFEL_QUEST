from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Literal

# Import ModelFactory (Assume it exists in parent or sibling, adjusting path relative to this file)
# Since this is in modules/, we need to import from ..model_factory if it exists there,
# or assume it's available via sys path. Let's use relative import based on existing structure.
try:
    from model_factory import ModelFactory
except ImportError:
    # Fallback/Mock if running standalone validation
    from langchain_openai import ChatOpenAI

    class ModelFactory:
        @staticmethod
        def get_rag_model(level="heavy", temperature=0):
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# --- LLM Initialization ---
# Use 'eval' model (GPT-4o-mini / Gemini Flash) for grading as it supports structured output better than HCX.
llm = ModelFactory.get_eval_model(level="light", temperature=0)


# --- 1. Retrieval Grader (Effectiveness check) ---
class GradeRetrieval(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
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
    Grades the relevance of retrieved documents.
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


# --- 2. Hallucination Grader (Groundedness check) ---
class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
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
    Checks if the generation is grounded in the documents.
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


# --- 3. Answer Grader (Helpfulness check) ---
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
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
    Checks if the answer helps the user.
    """
    print("--- [Modular RAG] Grading Answer Utility ---")

    score = answer_grader_chain.invoke({"question": question, "generation": generation})
    return score.binary_score
