from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.logger_config import setup_logger

logger = setup_logger("REWRITER")

# Import ModelFactory (relative import fallback)
try:
    from common.model_factory import ModelFactory
except ImportError:
    from langchain_openai import ChatOpenAI

    class ModelFactory:
        @staticmethod
        def get_rag_model(level="heavy", temperature=0):
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# --- LLM Initialization ---
llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

# --- Query Rewriter ---
rewriter_system = """
[역할]
당신은 감사 보고서 검색을 최적화하는 "검색 질의 재작성 전문가(Query Rewriter)"입니다.
사용자의 질문을 감사 보고서 검색에 가장 효과적인 "검색 질의 문자열"로 변환하는 것이 목표입니다.

[목표]
- 사용자의 질문에서 핵심 감사 쟁점과 키워드를 추출하여 검색에 적합한 형태로 재작성하십시오.
- 대화체 표현, 감탄사, 불필요한 설명은 제거하십시오.
- 감사 보고서에 자주 등장하는 용어를 우선적으로 사용하십시오.

[재작성 규칙]
1. 질문형 문장은 **검색용 명사/구문 형태**로 변환하십시오.
2. 질문이 포괄적이거나 모호한 경우에는
   - "유사 사례", "감사 지적 사례", "감사 결과" 중심의 검색 의도로 재작성하십시오.
3. 기관명, 연도, 사건 번호 등이 질문에 명시된 경우에는 그대로 유지하십시오.
4. 질문에 없는 기관명·연도·사건을 새로 만들어내지 마십시오.
5. 의미가 변하지 않도록 주의하며, 최대한 간결하게 작성하십시오.

[출력]
- 재작성된 "검색 질의 문자열만" 출력하십시오.
- 그 외의 설명, 문장, 기호는 절대 포함하지 마십시오.
"""

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
    logger.info(f"--- [Modular RAG] Rewriting Query: '{question}' ---")

    better_query = rewriter_chain.invoke({"question": question})
    logger.info(f" -> Optimized Query: '{better_query}'")

    return better_query
