from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal
from state import AgentState
from config import Config
from model_factory import ModelFactory


# 1. 출력 스키마
class SupervisorPlan(BaseModel):
    reasoning: str = Field(description="계획에 대한 단계별 추론 과정")
    category: Literal["chit_chat", "audit_qa", "research", "judgment"] = Field(
        description="사용자 질문의 유형 분류 결과"
    )
    plan: List[str] = Field(
        description="실행할 단계 리스트 (예: ['research_worker', 'audit_worker'])"
    )
    next_worker: str = Field(
        description="가장 먼저 호출할 워커 이름 (예: 'research_worker')"
    )
    filters: dict = Field(
        default={},
        description="메타데이터 필터 (예: {'source_type': 'BAI', 'focus_field': 'problems'})",
    )


# 2. Prompt
system_prompt = """
You are the **Supervisor Agent** (Audit Manager) for an AI Audit Assistant system.
Your goal is to analyze the user's request, plan the necessary steps, and delegate to the appropriate workers.
You can also determine if the search should be restricted using **Metadata Filters**.

### Workers Available:
1. **ChatWorker**: Handles greetings, casual conversation, and general questions NOT related to audit regulations or cases. (e.g. "Hello", "Who are you?")
2. **ResearchWorker**: Performs hybrid retrieval (Vector + Keyword) to find relevant audit cases, regulations, and guidelines. Use this for general audit inquiries. (e.g. "Find cases about overtime pay")
3. **AuditWorker**: Executes the **Standard Audit Procedure (SOP)**. Use this when the user asks for a specific judgment, compliance check, or detailed analysis of a situation against regulations. (e.g. "Is this contract violation?")
4. **AdversaryWorker**: Runs a simulated trial (Defense vs Prosecution). Use this ONLY when the AuditWorker explicitly flags a 'Violation' and the user wants to simulate a defense.

### Planning Rules:
- If the query is simple chit-chat, route to **ChatWorker**. Plan: `["chat_worker"]`. Next: `chat_worker`.
- If the query asks for information/cases, route to **ResearchWorker**. Plan: `["research_worker", "answer_generator"]`. Next: `research_worker`.
- If the query requires judgment/analysis (SOP), you **MUST** gather facts first. Plan: `["research_worker", "audit_worker"]`. Next: `research_worker`. **Category MUST be 'judgment'**.

### Metadata Filters:
- `source_type`: "BAI" (Audit Board) or "ALIO" (Public Inst).
- `focus_field`: "problems" (Violation), "action" (Disposition), "standards" (Rules), "outline" (Summary).

### Output Format:
You must output a single JSON object. Do not wrap it in markdown block (like ```json ... ```), just return the raw JSON.
The JSON must have the following keys:
- `reasoning`: Brief explanation of the plan.
- `category`: One of ["chit_chat", "audit_qa", "research", "judgment"].
- `plan`: List of worker names to execute in order.
- `next_worker`: The name of the first worker to trigger.
- `filters`: A dictionary of metadata filters.

Example:
{{
  "reasoning": "User asks for violation cases in BAI reports.",
  "category": "research",
  "plan": ["research_worker", "answer_generator"],
  "next_worker": "research_worker",
  "filters": {{
    "source_type": "BAI",
    "focus_field": "problems"
  }}
}}
"""


def supervisor_node(state: AgentState):
    print("--- [Supervisor] 계획 수립 및 라우팅 (Planning & Routing) ---")

    llm = ModelFactory.get_rag_model(level="heavy")  # 계획 수립을 위한 고성능 모델 사용
    parser = JsonOutputParser(pydantic_object=SupervisorPlan)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{query}")]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"query": state["query"]})

        print(f" -> Plan: {result['plan']}")
        print(f" -> Next: {result['next_worker']}")

        return {
            "category": result["category"],
            "plan": result["plan"],
            "next_step": result["next_worker"],
            "metadata_filters": result.get("filters", {}),
        }

    except Exception as e:
        print(f"Error in Supervisor: {e}")
        # Fallback
        return {
            "category": "search",
            "plan": ["research_worker"],
            "next_step": "research_worker",
        }
