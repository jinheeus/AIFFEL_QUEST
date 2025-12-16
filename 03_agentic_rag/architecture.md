# Agentic RAG Architecture (Supervisor-Worker Model)

본 문서는 **Supervisor-Worker 패턴**을 기반으로 설계된 고도화된 Agentic RAG 시스템의 아키텍처를 설명합니다. 이 구조는 중앙의 관리자(Supervisor)가 사용자의 복잡한 의도를 파악하고, 전문화된 작업자(Worker)들에게 업무를 위임하는 계층적 형태를 띱니다.

## 1. Architecture Overview (아키텍처 개요)

시스템은 **Think -> Plan -> Execute -> Reflect**의 사이클을 따릅니다.

### System Flow Diagram

```mermaid
graph TD
    User([User]) --> Supervisor
    
    subgraph "Brain (Supervisor Agent)"
        Supervisor["Supervisor Node"]
        Supervisor -->|Review| Plan["Planning & Routing"]
        Supervisor -->|Select Tool| Filter["Metadata Filter"]
    end
    
    Plan -->|Chit-chat| ChatWorker
    Plan -->|Research / Judgment| ResearchWorker
    
    ResearchWorker -.->|Context Handoff (Judgment)| AuditWorker
    
    subgraph "General"
        ChatWorker["ChatWorker"]
    end
    
    subgraph "Research (Advanced Adaptive RAG)"
        ResearchWorker["ResearchWorker"]
        FieldSelector["**Field Selector**"]
        Retriever["Hybrid Retriever"]
        Validator["**Validator**"]
        StrategyDecider["**Strategy Decider**"]
        GraphDB[("Graph DB")]
        VectorDB[("Milvus DB")]
        
        ResearchWorker --> FieldSelector
        FieldSelector --> Retriever
        Retriever -->|Search| VectorDB
        Retriever -->|Search| GraphDB
        
        Retriever --> Validator
        Validator -->|"Valid (Yes)"| GraphDB
        Validator -- "Invalid (No)" --> StrategyDecider
        StrategyDecider -->|"Rewrite Query / Update Fields"| Retriever
        
        %% Highlights for New Logic
        classDef newLogic fill:#ffcccc,stroke:#ff0000,stroke-width:2px;
        class FieldSelector,Validator,StrategyDecider newLogic;
    end
    
    subgraph "Audit SOP (Standard Procedure)"
        AuditWorker["AuditWorker"]
        FactExtraction["1. 팩트 추출"]
        RegulationMatching["2. 규정 매칭"]
        ComplianceCheck["3. 준수 여부 판단"]
        Disposition["4. 처분 결정"]
        
        AuditWorker --> FactExtraction
        FactExtraction --> RegulationMatching
        RegulationMatching --> ComplianceCheck
        ComplianceCheck --> Disposition
        
        Disposition -->|Violation Detected?| AdversarialCheck{"재판 필요 여부"}
    end
    
    subgraph "Adversarial Simulation (Trial)"
        AdversarialCheck -- Yes --> DefenseAgent["피감기관 (변호)"]
        DefenseAgent --> ProsecutionAgent["감사관 (검사)"]
        ProsecutionAgent --> Judge["위원회 (판결)"]
    end
    
    subgraph "Output & Reflection"
        Generator["Answer Generator"]
        Reflector["Reflector"]
        
        ChatWorker --> Generator
        GraphDB --> Generator
        Disposition -- No Violation --> Generator
        Judge --> Generator
        
        Generator --> Reflector
        Reflector -->|Pass| FinalAnswer(["Final Answer"])
        Reflector -->|Fail| Generator
    end
```

---

## 2. Agent Roles (에이전트 역할 상세)

### 1. Supervisor (관리자 에이전트)
- **Role**: 시스템의 **두뇌(Brain)** 역할을 수행합니다. 모든 사용자 요청의 진입점(Entry Point)입니다.
- **Capabilities**:
    1.  **Intent Classification**: 질문이 단순 대화인지, 정보 검색인지, 심층 감사인지 분류합니다.
    2.  **Planning (ReAct)**: 문제를 해결하기 위한 단계적 계획(Plan)을 수립합니다. (예: `['research_worker', 'answer_generator']`)
    3.  **Metadata Filtering (Tool)**: 사용자의 질문에서 특정 필드(예: '조치사항', '판단기준')나 출처(예: '감사원')를 파악하여, 검색 범위를 좁히는 필터를 생성합니다.

### 2. ChatWorker (대화 작업자)
- **Role**: 감사 업무와 무관한 **일상 대화(Chit-chat)**를 담당합니다.
- **Logic**: 복잡한 RAG 파이프라인이나 SOP를 거치지 않고, 가벼운 LLM(Light Model)을 사용하여 빠르고 자연스럽게 응답합니다. 리소스 낭비를 방지하는 역할을 합니다.

### 3. ResearchWorker (조사 작업자 & **Advanced Adaptive RAG** 🌟)
- **Role**: 정보 검색 요청을 처리하며, **지능적으로 검색 전략을 수정**합니다.
- **Logic (Advanced Adaptive Loop)**:
    1.  **Field Selector [NEW]**: 질문의 의도를 분석하여 검색할 메타데이터 필드를 선정합니다. (예: "조치사항 중심 검색")
    2.  **Hybrid Retriever**: 선택된 필드와 쿼리를 이용하여 1차 검색을 수행합니다.
    3.  **Validator [NEW]**: 검색된 문서만으로 답변이 가능한지 **COT(단계적 사고)**로 검증합니다.
        - **Fail**: **Strategy Decider**로 이동합니다.
        - **Pass**: 다음 단계로 진행합니다.
    4.  **Strategy Decider [NEW]**: 검증 실패 원인을 분석하여 전략을 결정합니다.
        - **Rewrite Query**: 키워드가 부적절하면 질문을 재작성합니다.
        - **Update Fields**: 정보가 부족하면 검색 필드를 확장합니다.
    5.  **Graph Retrieval**: Neo4j 지식 그래프 탐색.

### 4. AuditWorker (감사 작업자)
- **Role**: **"규정 위반인가?"** 판단.
- **Logic (SOP)**: Fact Extraction -> Regulation Matching -> Compliance Check -> Disposition.

### 5. AdversaryWorker (적대적 시뮬레이션)
- **Role**: 위반 건에 대한 **가상 재판(Trial)**.
- **Logic**: 변호(Defense) vs 검사(Prosecution) -> 판결(Verdict).

### 6. Hybrid Ingestion (Data Pipeline)
- **Concept**: 기존 파일 정보(PDF)와 사람이 검증한 메타데이터(JSON)를 병합하여 검색 품질을 극대화합니다.
- **Source**:
    - **Unstructured**: Docling Parsed Markdown (본문 텍스트)
    - **Structured**: `contents.json` (감사 쟁점, 조치사항, 판단기준 등 요약 정보)
- **Process**:
    1.  **Merge**: 파일명(Filename)을 기준으로 JSON의 핵심 필드(`problems`, `action` 등)를 추출.
    2.  **Context Injection**: 추출된 정보를 각 청크(Chunk)의 최상단에 주입(Prepend)하여, LLM이 문맥을 놓치지 않도록 함.
    3.  **Indexing**: `markdown_rag_hybrid_v1` 컬렉션에 적재.

---

## 3. Key Features (핵심 기술)

### Adaptive Retrieval (Self-Correction)
- **개요**: 검색 결과가 마음에 들지 않으면 에이전트가 스스로 질문을 고쳐 쓰고 다시 검색합니다.
- **효과**: 잘못된 검색어로 인한 답변 실패를 원천 차단합니다.

### Metadata Filtering (Field Selector)
- **개요**: "조치사항만 알려줘" 같은 요청에 대응하여 검색 범위를 좁힙니다.

### Intelligent Reflection (자기 성찰)
- **개요**: 최종 답변이 질문에 적합한지 검토하고, 부족하면 재생성합니다.
