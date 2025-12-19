import sqlite3
import re
import sys
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Ensure parent directory is in path to import config if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

# Try loading .env from project root
load_dotenv(os.path.join(project_root, ".env"))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from langchain_naver import ChatClovaX
except ImportError:
    print("Warning: langchain_naver not found. Please install it.")
    ChatClovaX = None

try:
    from config import Config

    MODEL_NAME = Config.LLM_MODEL

    # Validation
    if not os.getenv("CLOVASTUDIO_API_KEY"):
        # Try to fallback to Config key if available
        if hasattr(Config, "CLOVANSTUDIO_API_KEY") and Config.CLOVANSTUDIO_API_KEY:
            os.environ["CLOVASTUDIO_API_KEY"] = Config.CLOVANSTUDIO_API_KEY
            print("   [SQL Retriever] Set CLOVASTUDIO_API_KEY from Config.")

except ImportError:
    MODEL_NAME = "HCX-DASH-001"  # Fallback


class SQLRetriever:
    def __init__(self, db_path: str = "audit_metadata.db"):
        self.db_path = db_path

        # Check API Key
        if not os.getenv("CLOVASTUDIO_API_KEY") and not os.getenv(
            "NCP_CLOVASTUDIO_API_KEY"
        ):
            print("❌ Error: CLOVASTUDIO_API_KEY not found in environment.")
            print(f"Current Keys: {[k for k in os.environ.keys() if 'CLOVA' in k]}")

        if ChatClovaX:
            # Using ChatClovaX as per project standard
            # temperature=0 is sometimes invalid, use small float
            self.llm = ChatClovaX(model=MODEL_NAME, temperature=0.05, max_tokens=2048)
        else:
            raise ImportError("ChatClovaX is required for SQLRetriever.")

        # Schema for the LLM to understand
        self.schema_info = """
Table: audits
Columns:
- id (INTEGER): Primary Key
- idx (INTEGER): Unique ID for the audit case
- date (TEXT): Date of the audit case (YYYY-MM-DD or YYYY.MM.DD)
- title (TEXT): Title of the audit case
- site (TEXT): Source site (e.g., 'ALIO 공공기관 경영정보 공개시스템', '감사원')
- company (TEXT): Company name involved (e.g., '인천국제공항공사')
- company_code (TEXT): Company code if available
- category (TEXT): General category string
- cat (TEXT): Main category
- sub_cat (TEXT): Sub category
- file_path (TEXT): Path to the original file
- download_url (TEXT): URL to download the file
- problem (TEXT): The problem description
- action (TEXT): The action taken
Data Range: 2021-01-04 to 2024-06-28
"""

        self.prompt = ChatPromptTemplate.from_template("""
You are a SQL expert. Convert the user's natural language query into a SQLite query using the schema below.
Ensure the SQL query is efficient and correct.
Return ONLY the SQL query. Do not include markdown formatting (like ```sql).

{schema}

Current Date: 2025-12-18

[Conversation Context]
{context}

Rules:
1. **Context Resolution**: If the user refers to "this", "that", "the first case", "item #3", "the link", etc., look at [Conversation Context]. Find the corresponding `idx` or `title` and filter by it (e.g., `WHERE idx = 123`).
2. For "latest" or "recent" queries, **ALWAYS use ORDER BY date DESC**. **NEVER add a WHERE clause for date (e.g. date > ...)** unless the user explicitly asks for a specific date range (e.g. "2024년 이후"). "Latest" means sorting, not filtering.
3. **SQLite Compatibility**:
   - The function `year()` does NOT exist.
   - To filter by year, **ALWAYS use `date LIKE 'YYYY-%'`** (e.g. '2024-%').
   - Do NOT use `strftime('%Y', date) = 2024` (Integer comparison fails).
   - If you must use `strftime`, ensure you compare with a STRING: `strftime('%Y', date) = '2024'`.
4. **Target Columns**: Search for organization/company names in the `company` column, NOT the `site` column.
5. **Entity Resolution**: Convert short names or typos to the full official name if possible.
6. Select all columns (*) unless specified otherwise.
7. **Syntax Warning**: Ensure `WHERE` comes before `ORDER BY`, and `ORDER BY` comes before `LIMIT`.
8. **No Semicolons in Subqueries**: Do NOT put a semicolon `;` inside a subquery or nested SELECT. Only put ONE semicolon at the very end of the main query.
9. **Strict Filtering**: Do NOT add `WHERE` clauses for columns (like `category`, `problem`) unless the user explicitly asks for them.
10. **Correct Column Usage**:
    - **Companies/Organizations**: Use `company LIKE '%Name%'` (e.g. 'Incheon Airport' -> `company LIKE '%인천국제공항공사%'`).
    - **Source Sites**: Use `site = '감사원'` ONLY if the user asks for "Board of Audit" or "BAI".
11. **Simplicity First**: 
    - For "latest 3 items", simply use `ORDER BY date DESC LIMIT 3`.
    - **NEVER** add `date LIKE 'YYYY-%'` or any other year filter unless the user explicitly wrote a year (e.g. "2023년") in the query.
    - **No Redundant Filters**: Do NOT add `WHERE date >= '2021-...'` or `date <= '2024-...'`. If looking for all time, just omit the date clause.

User Query: {query}
SQL Query:
""")
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Executes the SQL query and returns list of dicts."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return []

    def _clean_sql(self, sql: str) -> str:
        """Removes markdown and whitespace."""
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    def retrieve(
        self, query: str, context: Optional[List[Document]] = None
    ) -> List[Document]:
        """
        Translates NL query to SQL and executes it, returning Documents.
        Args:
            query: User's natural language query.
            context: List of previous Documents (for resolving 'it', '#2', etc.)
        """
        print(f"   [SQL Retriever] Processing: {query}")

        # Format Context
        context_str = "No context available."
        if context:
            formatted = []
            for i, doc in enumerate(context, 1):
                idx = doc.metadata.get("idx", "Unknown")
                title = doc.metadata.get("title", "Unknown")
                formatted.append(f"Item #{i}: IDX={idx}, Title={title}")
            context_str = "\n".join(formatted)
            print(f"   [SQL Retriever] Context Provided ({len(context)} docs)")

        # 1. Generate SQL
        generated_sql = self.chain.invoke(
            {"schema": self.schema_info, "query": query, "context": context_str}
        )
        cleaned_sql = self._clean_sql(generated_sql)
        print(f"   [SQL Retriever] Generated SQL: {cleaned_sql}")

        # 2. Execute SQL
        results = self._execute_query(cleaned_sql)
        print(f"   [SQL Retriever] Found {len(results)} records.")

        # 3. Convert to Documents
        documents = []
        for row in results:
            content = f"Title: {row.get('title')}\nDate: {row.get('date')}\nCompany: {row.get('company')}\nProblem: {row.get('problem')}\nAction: {row.get('action')}"

            # Metadata
            metadata = {k: v for k, v in row.items() if k not in ["problem", "action"]}
            metadata["source"] = "sql_database"

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents


if __name__ == "__main__":
    # Test
    try:
        retriever = SQLRetriever()
        docs = retriever.retrieve("인천국제공항공사 최신 2건 알려줘")
        for doc in docs:
            print(f"[{doc.metadata.get('date')}] {doc.metadata.get('title')}")
    except Exception as e:
        print(f"Test Failed: {e}")
