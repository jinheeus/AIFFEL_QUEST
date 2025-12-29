import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_naver import ChatClovaX

from common.config import Config
from common.logger_config import setup_logger

# Get project root for db path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
# agentic_rag_v2 -> rag -> project_root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# Load .env (Though usually loaded by main, keeping it for standalone safety if needed, but cleaner)
# Assuming project root is correctly set in environment or running from root
load_dotenv()

logger = setup_logger("SQL_RETRIEVER")


class SQLRetriever:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Resolve common/audit_metadata.db relative to project root
            self.db_path = os.path.join(project_root, "common", "audit_metadata.db")
        else:
            self.db_path = db_path

        # Check API Key
        if not os.getenv("CLOVASTUDIO_API_KEY") and not os.getenv(
            "NCP_CLOVASTUDIO_API_KEY"
        ):
            logger.error("❌ Error: CLOVASTUDIO_API_KEY not found in environment.")
            # Masking keys for log safety
            keys = [k for k in os.environ.keys() if "CLOVA" in k]
            logger.info(f"Current CLOVA related env vars: {keys}")

        # ChatClovaX 초기화
        self.llm = ChatClovaX(model=Config.LLM_MODEL, temperature=0.05, max_tokens=2048)

        # Schema definition for the LLM
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
당신은 "SQL 전문가"입니다. 사용자의 자연어 질문을 아래 스키마를 사용해 "SQLite 쿼리"로 변환하십시오.
{schema}

현재 날짜: {current_date}
[대화 컨텍스트]
{context}

[규칙 - 핵심]
1. "키워드 추출 및 필터링 (필수)":
   - 사용자의 질문에서 '기관명'이 아닌 모든 핵심 명사(예: 횡령, 레일바이크, 안전, 계약 등)는 검색 키워드로 간주합니다.
   - 키워드가 발견되면 반드시 `WHERE` 절을 사용하여 `title`, `problem`, `cat`, `sub_cat` 컬럼에서 `LIKE` 검색을 수행하십시오.
   - 예: "레일바이크 관련 3개" -> `WHERE (title LIKE '%레일바이크%' OR problem LIKE '%레일바이크%' OR cat LIKE '%레일바이크%' OR sub_cat LIKE '%레일바이크%')`

2. "최신/최근(latest/recent) 정렬":
   - "최신", "최근", 혹은 단순히 결과의 순서가 중요해 보이는 경우 `ORDER BY date DESC`를 추가하십시오.
   - 단, `ORDER BY`가 `WHERE` 필터링을 생략하는 이유가 되어서는 안 됩니다.

3. "개수 제한 (LIMIT)":
   - "n개 알려줘", "n건" 등의 요청이 있으면 쿼리 맨 마지막에 `LIMIT n`을 붙이십시오.

4. "단순성 및 정확성":
   - 사용자가 언급한 키워드는 하나도 빠짐없이 `WHERE` 절에 반영해야 합니다. 
   - '최신 3개'라고 해서 키워드 없이 전체에서 3개만 가져오는 오류를 범하지 마십시오.

[SQLite 구문 규칙]
- `year()` 사용 금지, `date LIKE '2023-%'` 또는 `strftime('%Y', date) = '2023'` 사용.
- `LIMIT`는 항상 `ORDER BY` 뒤에 위치.
- 세미콜론은 전체 쿼리 끝에 한 번만.
- 모든 SQL은 반드시 다음 순서에 맞춰 조립하십시오:
    SELECT * FROM audits
    WHERE (키워드 필터링)
    ORDER BY date DESC (최신순 요청 시)
    LIMIT n (개수 요청 시 - 무조건 마지막)

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
            logger.error(f"Error executing SQL: {e}")
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
        """
        logger.info(f"Processing Query: {query}")

        # Context Formatting
        context_str = "No context available."
        if context:
            formatted = []
            for i, doc in enumerate(context, 1):
                idx = doc.metadata.get("idx", "Unknown")
                title = doc.metadata.get("title", "Unknown")
                formatted.append(f"Item #{i}: IDX={idx}, Title={title}")
            context_str = "\\n".join(formatted)
            logger.info(f"Context Provided ({len(context)} docs)")

        # 1. SQL 생성 (Generate SQL)
        current_date = datetime.now().strftime("%Y-%m-%d")
        generated_sql = self.chain.invoke(
            {
                "schema": self.schema_info,
                "query": query,
                "context": context_str,
                "current_date": current_date,
            }
        )
        cleaned_sql = self._clean_sql(generated_sql)
        logger.info(f"Generated SQL: {cleaned_sql}")

        # 2. SQL 실행 (Execute SQL)
        results = self._execute_query(cleaned_sql)
        logger.info(f"Found {len(results)} results")

        # 3. 문서 변환 (Convert to Documents)
        documents = []
        for row in results:
            content = f"Title: {row.get('title')}\\nDate: {row.get('date')}\\nCompany: {row.get('company')}\\nProblem: {row.get('problem')}\\nAction: {row.get('action')}"

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
        logger.error(f"Test Failed: {e}")
