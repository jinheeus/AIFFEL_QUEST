from typing import Any, Dict, List
import pandas as pd
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .shared import rag_pipeline

# Access standard LLM from pipeline
llm = rag_pipeline.llm


class StatEngineV2:
    def __init__(self, milvus_client=None):
        self.milvus_client = milvus_client
        self.collection_name = "problems"  # Ensure we use the correct collection
        self.df = None

    def load_data(self):
        """Loads data from Milvus into a Pandas DataFrame."""
        if self.df is not None:
            return self.df

        # Fetch basic metadata for stats
        # We fetch all data for now (assuming manageable size) or a representative sample
        # In a real scenario, we might use a SQL query or limit
        try:
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="idx >= 0",  # Get all (idx is Int64 in 'problems')
                output_fields=["idx", "cat", "sub_cat", "site", "date", "text"],
            )
            self.df = pd.DataFrame(results)
            # Rename for clarity
            self.df.rename(columns={"site": "org", "cat": "category"}, inplace=True)

            # Ensure date is datetime
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
            self.df["year"] = self.df["date"].dt.year
            print(f"[StatEngineV2] Loaded {len(self.df)} records.")
        except Exception as e:
            print(f"[StatEngineV2] Error loading data: {e}")
            self.df = pd.DataFrame()  # Empty fallback

        return self.df

    def analyze(self, query: str) -> str:
        """
        Analyzes the query and dataframe using Code Interpreter pattern.
        """
        df = self.load_data()

        if df.empty:
            return "데이터를 불러올 수 없어 통계 분석이 불가능합니다."

        # 1. Provide Schema info
        schema_info = df.dtypes.to_string()
        preview = df.head(3).to_string(index=False)
        categories = df["sub_cat"].unique()[:10]

        # 2. Prompt for Python Code
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Python Data Analyst.
You have access to a pandas DataFrame named `df`.
The user will ask a statistical question. Your job is to write a Python function `analyze_data(df)` that:
1. Performs the necessary pandas operations (filtering, groupby, aggregation) to answer the question.
2. Returns the result as a formatted string (Korean).

[DataFrame Schema]
{schema_info}

[Important Field Descriptions]
- `text`: Contains the full description of the issue (e.g. "횡령 금액 1억원", "징계 처분" details). 
  - USE THIS for keyword matching (e.g. `df[df['text'].str.contains('횡령')]`) or extracting numbers using regex.
- `sub_cat`: Sub-category of the problem.
- `year`: Extracted year from date.

[DataFrame Preview]
{preview}

[Constraints]
- **CRITICAL**: Do NOT use matplotlib, seaborn, or any visualization libraries. Code must only calculate and return text.
- **CRITICAL**: Use the existing `df['year']` column for grouping. Do NOT use `df['date'].dt.year`.
- **CRITICAL**: Do NOT use `groupby(...).size().max().index`. The result of `max()` is a number, not a Series.
  - To find the *category* with the most items, use `df['col'].value_counts().idxmax()` or `df.groupby('col').size().idxmax()`.
  - Alternatively, use `df.groupby('col').size().sort_values(ascending=False).head(1)`.
- The function match strictly be named `analyze_data`.
- The function must take `df` as an argument.
- Return a string summary of the result.
- Handle potential errors (e.g., empty result) gracefully.
- ONLY output the python code block. No explanation.
""",
                ),
                ("human", "{query}"),
            ]
        )

        try:
            chain = prompt | llm | StrOutputParser()
            code_response = chain.invoke(
                {"schema_info": schema_info, "preview": preview, "query": query}
            )

            # Clean formatting (```python ... ```)
            code_block = (
                code_response.replace("```python", "").replace("```", "").strip()
            )

            # Sanitization: Remove visualization code despite instructions
            code_lines = code_block.split("\n")
            sanitized_lines = []
            for line in code_lines:
                if "matplotlib" in line or "plt." in line or "pyplot" in line:
                    continue
                sanitized_lines.append(line)
            code_block = "\n".join(sanitized_lines)

            print(f"[StatEngineV2] Generated Code:\n{code_block}")

            # 3. Execute Code
            local_vars = {"pd": pd, "df": df, "datetime": datetime}
            # We use exec. In production, need sandboxing.
            exec(code_block, {}, local_vars)

            # 4. Get Result
            analyze_data_func = local_vars.get("analyze_data")
            if analyze_data_func:
                result = analyze_data_func(df)
                return result
            else:
                return "분석 함수 생성에 실패했습니다."

        except Exception as e:
            return f"통계 분석 중 오류가 발생했습니다: {str(e)}"
