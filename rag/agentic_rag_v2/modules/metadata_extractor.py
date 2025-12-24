from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from state import AgentState
from common.model_factory import ModelFactory

# --- Initialize LLM ---
llm = ModelFactory.get_rag_model(level="heavy", temperature=0)

# --- Prompts ---
extractor_system = """
[Role]
You are a 'Metadata Extractor' for an Audit Retrieval System.
Your goal is to extract specific filtering criteria from the user's natural language query.

[Target Fields]
Extract values for the following keys only. If a filter is not mentioned, do not include it.

1. `company_code`: Look for Alio company codes (e.g., C0105, C0001) or infer from company names if implied (e.g. "Airport Corp" -> "C0105").
   - Known Mappings:
     - 인천국제공항공사 (Incheon Airport) -> C0105
     - 한국가스안전공사 (Gas Safety) -> C0302
     - 부산항만공사 (Busan Port) -> C0203
     - 한국건설교통신기술협회 -> (No code, ignore)
   - If user explicitly mentions a code like "C1234", use it.

2. `year`: Valid 4-digit year (e.g., 2021, 2022). Extract if user asks for specific year's data.

3. `source_type`: 
   - 'BAI_JSON' if user implies Audit Board (감사원) or general audit.
   - (Currently only BAI_JSON is active, so default to empty unless specific distinction needed).

4. `idx`: If user refers to a specific Case ID (e.g. "Case 3", "Index 100"), extract as integer.

[Output Format]
Return a JSON object with a `filters` key.
Example:
Input: "Incheon Airport 2021 audit results?"
Output: {{"filters": {{"company_code": "C0105", "date": "2021"}}}}

Input: "Show me corruption cases"
Output: {{"filters": {{}}}}

[Rules]
- Only return the JSON.
- Do not hallucinate codes. Only use what is explicitly in query or the known mappings above.
- Date should be partial match string for 'date' field (e.g. "2021" will match "2021.03.05").
"""

extractor_user = """
[Query]
{query}
"""

extractor_prompt = ChatPromptTemplate.from_messages(
    [("system", extractor_system), ("human", extractor_user)]
)

extractor_chain = extractor_prompt | llm | JsonOutputParser()


def extract_metadata(state: AgentState) -> dict:
    print("--- [Node] Metadata Extractor ---")
    query = state.get("search_query") or state["query"]

    try:
        result = extractor_chain.invoke({"query": query})
        filters = result.get("filters", {})

        # Clean up empty keys
        filters = {k: v for k, v in filters.items() if v}

        if filters:
            print(f" -> Extracted Filters: {filters}")
        else:
            print(" -> No filters extracted.")

        return {"metadata_filters": filters}

    except Exception as e:
        print(f" -> Extraction Failed: {e}")
        return {"metadata_filters": {}}
