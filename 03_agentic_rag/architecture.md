# Modular Agentic RAG Architecture (CRAG + Self-RAG + SOP)

This document describes the high-level architecture of the Modular Agentic RAG system for the AURA Audit Assistant. It combines **Corrective RAG (CRAG)**, **Self-RAG**, and **Standard Operating Procedure (SOP)** integration to ensure high precision, regulatory compliance, and hallucination-free generation.

## Agentic Flow (White-Box Logic View)

```mermaid
graph TD
    %% --- Subgraph: Router Logic ---
    subgraph Router_Box ["Router Component"]
        direction TB
        Input(("User Input")) --> Keyword{"Contains<br>'Hello'?"}
        Keyword -- Yes --> ChatNode["Chat Worker<br/>Simple Response"]
        Keyword -- No --> Classify["LLM Classifier<br/>Intent & Pivot"]
        Classify --> CheckInt{"Intent?"}
        Classify --> PivotCheck{{"New Topic?<br/>(Context Clear)"}}
    end

    %% --- Subgraph: Fast Track (SQL) ---
    subgraph Fast_Box ["Fast Retrieval"]
        direction TB
        SQL["SQL Retriever<br/>Metadata/Date Search"]
    end

    %% --- Subgraph: RAG Preparation ---
    subgraph Planner_Box ["Field Selector (Planner)"]
        direction TB
        Analyze["CoT Analysis<br/>Step-by-Step Plan"] --> Extract["Metadata Extractor<br/>Filter: Year/Source"]
        Extract --> Refine["Search Query<br/>Keyword Refinement"]
    end

    %% --- Subgraph: Hybrid Retrieval ---
    subgraph Retriever_Box ["Hybrid Retriever Engine"]
        direction TB
        Fork((Fork)) --> Sparse["Sparse Search<br/>BM25 + Kiwi"]
        Fork --> Dense["Dense Search<br/>Milvus + Embedding"]
        Sparse --> Fusion["RRF Fusion<br/>Reciprocal Rank"]
        Dense --> Fusion
        Fusion --> Rerank["Cross Encoder<br/>BGE-Reranker"]
        Rerank --> TopK{"Top-5?"}
    end

    %% --- Subgraph: SOP Execution ---
    subgraph SOP_Box ["SOP Engine"]
        direction TB
        Fact["Fact Extraction<br/>5W1H Analysis"] --> Match["Regulation Match<br/>Law/Precedent"]
        Match --> Compliance["Compliance Check<br/>Violation Detection"]
        Compliance --> Dispo["Disposition<br/>Action Decision"]
    end

    %% --- Subgraph: Verification & Memory ---
    subgraph Verifier_Box ["Verification Loop"]
        direction TB
        HalluCheck{"Hallucination<br/>Grounded?"} 
        UtilCheck{"Utility<br/>Helpful?"}
        Memory["Summary Memory<br/>Compress & Save"]
    end

    %% --- Main Flow Connections ---
    CheckInt -- Chat --> ChatNode
    CheckInt -- Fast --> SQL
    SQL --> Generate
    
    CheckInt -- Deep --> Analyze
    
    Refine --> Fork
    TopK --> GradeLogic{"Grader<br/>Relevant?"}
    
    GradeLogic -- Yes --> Fact
    GradeLogic -- No --> Rewrite["Query Rewriter"]
    Rewrite --> Fork
    
    Dispo --> Generate["Generator<br/>Draft Answer"]
    Generate --> HalluCheck
    
    HalluCheck -- Fail --> Generate
    HalluCheck -- Pass --> UtilCheck
    
    UtilCheck -- Fail --> Rewrite
    UtilCheck -- Pass --> Memory
    
    Memory --> EndNode((End))
    ChatNode --> EndNode

    %% --- Styling ---
    style Router_Box fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Fast_Box fill:#e0f7fa,stroke:#006064,stroke-width:2px
    style Planner_Box fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style Retriever_Box fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    style SOP_Box fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style Verifier_Box fill:#ffebee,stroke:#b71c1c,stroke-width:2px
```

## Core Components



## Data Flow
1. **Query** -> `Field Selector` -> Structured Metadata.
2. **Metadata + Query** -> `Hybrid Retriever` -> Raw Docs.
3. **Raw Docs** -> `Retrieval Grader` -> Filtered Relevant Docs.
4. **Relevant Docs** -> `SOP Retriever` -> SOP/Rules.
5. **Docs + SOP** -> `Generator` -> Draft Answer.
6. **Draft Answer** -> `Verification Loop` -> Final Answer.

## Context Persistence Strategy (Memory Pivot)
To handle multi-turn conversations effectively, the Router employs a **Pivot Detection** mechanism:
- **New Topic (Pivot)**: If the user asks about a new entity (e.g., "Switching from Incheon to Gas Corp"), the Router sets `is_new_topic=True`. This **clears** the `persist_documents` to ensure a fresh search without context pollution.
- **Follow-up**: If the user asks for more details (e.g., "Give me the file for #1"), the Router sets `is_new_topic=False`. The previous `documents` are maintained in `persist_documents` so the SQL Retriever can resolve references like "Item #1".
