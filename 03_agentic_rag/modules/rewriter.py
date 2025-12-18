from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import ModelFactory (relative import fallback)
try:
    from model_factory import ModelFactory
except ImportError:
    from langchain_openai import ChatOpenAI

    class ModelFactory:
        @staticmethod
        def get_rag_model(level="heavy", temperature=0):
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# --- LLM Initialization ---
llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

# --- Query Rewriter ---
rewriter_system = """[Role]
You are a professional audit search optimizer. 
Your job is to rewrite a user's question to be more effective for a semantic/keyword search engine containing audit reports.

[Goal]
Transform the input question into a better search query.
- Focus on audit keywords: "violation", "misuse", "embezzlement", "insufficient monitoring".
- Remove conversational filler.
- If the query is vague, infer the likely intent (finding similar cases).

[Output]
Output ONLY the rewritten query string. No explanations."""

rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewriter_system),
        (
            "human",
            "Original Question: {question} \n\n Formulate an improved search query:",
        ),
    ]
)

rewriter_chain = rewriter_prompt | llm | StrOutputParser()


def rewrite_query(question: str) -> str:
    """
    Rewrites the question to improve retrieval.
    """
    print(f"--- [Modular RAG] Rewriting Query: '{question}' ---")

    better_query = rewriter_chain.invoke({"question": question})
    print(f" -> Optimized Query: '{better_query}'")

    return better_query
