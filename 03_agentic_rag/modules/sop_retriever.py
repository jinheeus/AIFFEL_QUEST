from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from state import AgentState
from model_factory import ModelFactory


# --- Data Models (Pydantic) ---
class FactOutput(BaseModel):
    subject: str = Field(description="행위 주체 (예: OO본부, 담당자)")
    action: str = Field(description="핵심 감사 대상 행위")
    amount: str = Field(description="관련 금액 (없으면 'N/A')")
    date: str = Field(description="행위 시점")


class ComplianceOutput(BaseModel):
    status: str = Field(description="판정 결과 (위반/준수/판단불가)")
    reasoning: str = Field(description="규정과 사실관계를 대조한 논리")
    matched_regulation: str = Field(
        description="적용된 근거 규정/지침 (LLM 내부 지식 활용)"
    )


class DispositionOutput(BaseModel):
    disposition: str = Field(
        description="처분 유형 (주의, 경고, 시정, 징계, 변상, 권고 등)"
    )
    detail: str = Field(description="구체적인 처분 내용 및 근거")


# --- Prompts ---
FACT_PROMPT = """
[Role]
You are a meticulous Audit Fact Finder.

[Task]
Extract the key facts (Subject, Action, Amount, Date) from the retrieved documents.
Focus ONLY on the specific incident related to the user's query.

[Query]
{query}

[Documents]
{context}
"""

REGULATION_PROMPT = """
[Role]
You are a Korean Public Audit Law Expert (규정 전문가).

[Task]
Based on the extracted facts, identify the specific **Laws (법령)**, **Regulations (규정)**, or **Guidelines (지침)** that apply.
Use your internal knowledge to cite the most relevant clause (e.g., '국가를 당사자로 하는 계약에 관한 법률 제27조').

[Facts]
{facts}
"""

COMPLIANCE_PROMPT = """
[Role]
You are an Audit Judge.

[Task]
Compare the **Facts** against the **Regulations**.
Determine if the action identifies a **Violation (위반)** or is **Compliant (준수)**.

[Facts]
{facts}

[Regulations]
{regulations}
"""

DISPOSITION_PROMPT = """
[Role]
You are an Audit Sentencing Committee.

[Task]
Based on the violation and standard audit criteria, determine the appropriate **Disposition (처분)**.
Common dispositions: 시정(Correction), 주의(Caution), 경고(Warning), 징계(Disciplinary Action), 변상(Reimbursement).

[Violation Context]
{compliance_result}
"""


def sop_retriever(state: AgentState) -> dict:
    """
    SOP Execution Node (4-Step Chain):
    1. Fact Extraction
    2. Regulation Matching (internal knowledge)
    3. Compliance Check
    4. Disposition Decision
    """
    print("--- [Node] SOP Generator (4-Step Chain) ---")
    query = state.get("search_query") or state["query"]
    docs = state.get("documents", [])

    # Handle Document objects
    docs_text = []
    for d in docs:
        if hasattr(d, "page_content"):
            docs_text.append(d.page_content)
        else:
            docs_text.append(str(d))

    context_text = "\n\n".join(docs_text)[:10000]  # Limit context length

    llm = ModelFactory.get_rag_model(
        level="heavy", temperature=0
    )  # Use heavy for reasoning

    # Step 1: Fact Extraction
    print(" -> 1. Extracting Facts...")
    fact_chain = (
        ChatPromptTemplate.from_template(FACT_PROMPT)
        | llm
        | JsonOutputParser(pydantic_object=FactOutput)
    )
    try:
        facts = fact_chain.invoke({"query": query, "context": context_text})
    except:
        facts = {
            "subject": "Unknown",
            "action": "General Query",
            "amount": "-",
            "date": "-",
        }

    # Step 2: Regulation Matching
    print(" -> 2. Matching Regulations...")
    reg_chain = (
        ChatPromptTemplate.from_template(REGULATION_PROMPT) | llm | StrOutputParser()
    )
    regs = reg_chain.invoke({"facts": str(facts)})

    # Step 3: Compliance Check
    print(" -> 3. Checking Compliance...")
    comp_chain = (
        ChatPromptTemplate.from_template(COMPLIANCE_PROMPT)
        | llm
        | JsonOutputParser(pydantic_object=ComplianceOutput)
    )
    try:
        compliance = comp_chain.invoke({"facts": str(facts), "regulations": regs})
    except:
        compliance = {
            "status": "Unknown",
            "reasoning": "Error in logic",
            "matched_regulation": regs,
        }

    # Step 4: Disposition
    print(" -> 4. Determining Disposition...")
    disp_chain = (
        ChatPromptTemplate.from_template(DISPOSITION_PROMPT)
        | llm
        | JsonOutputParser(pydantic_object=DispositionOutput)
    )
    try:
        disposition = disp_chain.invoke({"compliance_result": str(compliance)})
    except:
        disposition = {"disposition": "Refer to Manual", "detail": "Logic Error"}

    # Format Final Output for Generator
    sop_result = f"""
[SOP Analysis Result]
1. **Facts**: {facts}
2. **Regulations**: {compliance["matched_regulation"]}
3. **Compliance**: {compliance["status"]} ({compliance["reasoning"]})
4. **Disposition**: {disposition["disposition"]} - {disposition["detail"]}
"""
    print(f" -> SOP Result: {compliance['status']} / {disposition['disposition']}")

    # Save detailed results to state (optional, for debugging)

    return {"sop_context": sop_result}
