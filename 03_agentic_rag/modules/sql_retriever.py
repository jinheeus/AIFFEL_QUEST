import sqlite3
import re  # noqa: F401 (Imported but unused, keeping just in case for future regex needs, or remove if strictly cleaning)
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

from langchain_naver import ChatClovaX
from config import Config


class SQLRetriever:
    def __init__(self, db_path: str = "audit_metadata.db"):
        self.db_path = db_path

        # Check API Key
        if not os.getenv("CLOVASTUDIO_API_KEY") and not os.getenv(
            "NCP_CLOVASTUDIO_API_KEY"
        ):
            print("❌ Error: CLOVASTUDIO_API_KEY not found in environment.")
            print(f"Current Keys: {[k for k in os.environ.keys() if 'CLOVA' in k]}")

        # ChatClovaX 초기화
        # temperature=0이 가끔 튀는 경우가 있어 작은 float 값을 사용합니다.
        self.llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.05, max_tokens=2048)

        # LLM이 이해하기 위한 스키마 정보 (Schema for LLM)
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
4. **Target Columns (CRITICAL)**:
   - **Organization/Company Names**: Search in `company` column (e.g., "Incheon Airport" -> `company LIKE '%인천국제공항공사%'`).
   - **Topics/Subjects (e.g., Contract, Safety, Budget)**: Search in `title`, `problem`, `cat`, or `sub_cat`. **NEVER** search for topics in `company`.
     - Example: "contract cases" -> `(title LIKE '%계약%' OR problem LIKE '%계약%' OR cat LIKE '%계약%' OR sub_cat LIKE '%계약%')`
5. Select all columns (*) unless specified otherwise.
6. **Syntax Warning**: Ensure `WHERE` comes before `ORDER BY`, and `ORDER BY` comes before `LIMIT`.
7. **No Semicolons in Subqueries**: Do NOT put a semicolon `;` inside a subquery or nested SELECT. Only put ONE semicolon at the very end of the main query.
8. **Correct Column Usage**:
    - **Companies/Organizations**: Use `company LIKE '%Name%'`.
    - **Source Sites**: Use `site = '감사원'` ONLY if the user asks for "Board of Audit" or "BAI".
9. **Simplicity First**: 
    - For "latest 3 items", simply use `ORDER BY date DESC LIMIT 3`.
    - **NEVER** add `date LIKE 'YYYY-%'` unless explicitly asked.

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
        자연어 질문(NL)을 SQL로 변환하여 실행하고, 결과를 Document 리스트로 반환합니다.
        Args:
            query: 사용자의 자연어 질문
            context: 이전 문서 리스트 ('그거', '#2' 등의 참조 해결용)
        """
        print(f"   [SQL Retriever] 처리 중: {query}")

        # 컨텍스트 포맷팅 (Format Context)
        context_str = "No context available."
        if context:
            formatted = []
            for i, doc in enumerate(context, 1):
                idx = doc.metadata.get("idx", "Unknown")
                title = doc.metadata.get("title", "Unknown")
                formatted.append(f"Item #{i}: IDX={idx}, Title={title}")
            context_str = "\n".join(formatted)
            print(f"   [SQL Retriever] 컨텍스트 제공됨 ({len(context)} docs)")

        # 1. SQL 생성 (Generate SQL)
        generated_sql = self.chain.invoke(
            {"schema": self.schema_info, "query": query, "context": context_str}
        )
        cleaned_sql = self._clean_sql(generated_sql)
        print(f"   [SQL Retriever] 생성된 SQL: {cleaned_sql}")

        # 2. SQL 실행 (Execute SQL)
        results = self._execute_query(cleaned_sql)
        print(f"   [SQL Retriever] 검색 결과: {len(results)}건 발견.")

        # 3. 문서 변환 (Convert to Documents)
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
