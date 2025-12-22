import sqlite3
import re  # noqa: F401 (Imported but unused, keeping just in case for future regex needs, or remove if strictly cleaning)
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
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
당신은 "SQL 전문가"입니다. 사용자의 자연어 질문을 아래 스키마를 사용해 "SQLite 쿼리"로 변환하십시오.
쿼리는 "효율적이고 정확"해야 합니다.
"반드시 SQL 쿼리만 출력"하십시오. 설명, 주석, 마크다운(예: ```sql)은 절대 포함하지 마십시오.

{schema}

현재 날짜: {current_date}

[대화 컨텍스트]
{context}

[규칙]
1. "컨텍스트 지시어 해소(Context Resolution)":
   - 사용자가 "이거/저거/그거", "첫 번째 사건", "3번 항목", "그 링크"처럼 지시어를 쓰면, [대화 컨텍스트]에서 해당 항목을 찾아 **idx 또는 title**로 식별하고 필터링하십시오.
     예: `WHERE idx = 123` 또는 `WHERE title LIKE '%...%'`
   - 사용자가 이전에 정정한 내용이 있다면(예: '도로공사 아니고 가스공사야'), 부인된 엔터티는 **무시**하고 **정정된** 내용을 사용하십시오.
   - 컨텍스트에서 특정 항목을 확정할 수 없으면, 임의로 추측하여 idx를 만들지 말고, 가능한 가장 보수적인(=과도한 필터를 피하는) 쿼리를 작성하십시오.

2. "최신/최근(latest/recent) 처리 규칙(정렬 우선)":
   - "최신", "최근", "latest", "recent" 요청이면 "항상" `ORDER BY date DESC`를 사용하십시오.
   - 사용자가 기간을 명시하지 않은 경우, "날짜 범위 WHERE 조건을 절대 추가하지 마십시오."
     ("최신"은 필터링이 아니라 정렬입니다.)
   - 사용자가 "2024년 이후"처럼 기간을 명시한 경우에만 날짜 조건을 추가할 수 있습니다.

3. "SQLite 호환성(연도 처리 포함)":
   - `year()` 함수는 존재하지 않습니다.
   - 연도 필터가 필요하면 기본 원칙은 `date LIKE 'YYYY-%'`를 사용하십시오.
   - `strftime`를 꼭 써야 한다면 "문자열로 비교"하십시오:
     `strftime('%Y', date) = '2024'` (정수 비교 금지)

4. "기관/회사명 검색 컬럼 고정":
   - 기관/회사명은 `site`가 아니라 "반드시 `company` 컬럼"에서 검색하십시오.

    5. "주제/이슈 검색 데이터 범위 확장 (Multi-Column Search)":
       - 계약, 안전, 예산, 횡령 등 주제나 이슈를 검색할 때는 **반드시** `title`, `problem`, `cat`, `sub_cat` 컬럼을 모두 검사하십시오.
       - 이유: DB의 `title`은 "공공임대주택 실태"처럼 포괄적일 수 있지만, `cat`, `sub_cat`이나 `problem`에 "횡령", "비리" 같은 핵심 키워드가 있을 수 있습니다.
       - 작성 예시:
         `WHERE (title LIKE '%횡령%' OR problem LIKE '%횡령%' OR cat LIKE '%횡령%' OR sub_cat LIKE '%횡령%')`
       - **절대로** `company` 컬럼에서 주제를 검색하지 마십시오.

6. "엔터티 정규화(Entity Resolution)":
   - 짧은 표기/오타가 있으면, 가능하면 공식 명칭으로 정규화할 수 있습니다.
   - 단, 확실하지 않은 경우 임의로 기관명을 만들어내지 마십시오.

7. "기본 선택 컬럼":
   - 사용자가 특정 컬럼만 요청하지 않는 한 `SELECT *`를 사용하십시오.

8. "구문 순서 경고":
   - `WHERE` → `ORDER BY` → `LIMIT` 순서를 반드시 지키십시오.

9. "서브쿼리 세미콜론 금지":
   - 서브쿼리/중첩 SELECT 내부에는 세미콜론 `;`을 넣지 마십시오.
   - 전체 쿼리의 맨 끝에만 세미콜론을 "정확히 한 번" 넣으십시오.

10. "엄격한 필터링(사용자 명시 없으면 금지)":
    - 사용자가 명시적으로 요구하지 않는 한 `category`, `problem` 등 다른 컬럼으로 추가 `WHERE` 조건을 만들지 마십시오 (단, 주제 검색 시에는 예외).

11. "컬럼 사용 규칙(정확히 준수)":
    - "기관/회사명": `company LIKE '%기관명%'`
      예: 'Incheon Airport' → `company LIKE '%인천국제공항공사%'`
    - "출처 사이트(site)": 사용자가 "감사원", "BAI", "Board of Audit"를 명시한 경우에만
      `site = '감사원'` 조건을 사용할 수 있습니다.

12. "단순성 우선(Simplicity First)":
    - 예: "최신 3개" → `ORDER BY date DESC LIMIT 3`
    - 사용자가 연도를 직접 적지 않았다면(예: "2023년") 연도 필터를 "절대 추가하지 마십시오."
    - 불필요한 날짜 조건(`date >= ...`, `date <= ...`)은 만들지 마십시오.

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
            context_str = "\\n".join(formatted)
            print(f"   [SQL Retriever] 컨텍스트 제공됨 ({len(context)} docs)")

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
        print(f"   [SQL Retriever] 생성된 SQL: {cleaned_sql}")

        # 2. SQL 실행 (Execute SQL)
        results = self._execute_query(cleaned_sql)
        print(f"   [SQL Retriever] 검색 결과: {len(results)}건 발견.")

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
        print(f"Test Failed: {e}")
