# High-Context RAG Architecture

This document illustrates the "Split & Merge" architecture used in the High-Context RAG system.

```mermaid
flowchart TD
    User([User Query]) --> Pipeline[HighContextRAGPipeline]
    
    subgraph "Step 1: Router & Search"
        Pipeline -->|Parallel Search| SearchProblems[Search 'problems']
        Pipeline -->|Parallel Search| SearchStandards[Search 'standards']
        
        SearchProblems -->|Top-K Chunks| Results[Search Results]
        SearchStandards -->|Top-K Chunks| Results
    end
    
    subgraph "Step 2: ID Extraction"
        Results -->|Extract unique 'idx'| IDList[Candidate IDs]
    end
    
    subgraph "Step 3: Context Hydration"
        IDList -->|Query by ID| Milvus{Milvus DB}
        
        Milvus -->|Fetch| Col1[problems]
        Milvus -->|Fetch| Col2[standards]
        Milvus -->|Fetch| Col3[action]
        Milvus -->|Fetch| Col4[outline]
        Milvus -->|Fetch| Col5[opinion]
        Milvus -->|Fetch| Col6[title]
        
        Col1 & Col2 & Col3 & Col4 & Col5 & Col6 --> HydratedData[Hydrated Data Fragments]
    end
    
    subgraph "Step 4: Reconstruction"
        HydratedData -->|Assemble| FullDocs[Reconstructed Full Documents]
    end
    
    subgraph "Step 5: Reranking"
        FullDocs -->|Cross-Encoder| Reranker[Reranker]
        Reranker -->|Top-K Docs| FinalContext[Final Context]
    end
    
    subgraph "Step 6: Generation"
        FinalContext -->|Context + Query| LLM[ChatClovaX]
        LLM --> Answer([Final Answer])
    end
    
    style User fill:#f9f,stroke:#333,stroke-width:2px
    style Answer fill:#9f9,stroke:#333,stroke-width:2px
    style Milvus fill:#bbf,stroke:#333,stroke-width:2px
```

## Key Components
1.  **Split Search**: We only search `problems` and `standards` collections first because they contain the most distinctive keywords ("Hooks").
2.  **ID-Based Hydration**: Once we find a relevant chunk, we use its `idx` to pull *all* related information from other collections, ensuring the LLM sees the full picture.
3.  **Reconstruction**: Fragments are reassembled into a structured format (Title, Outline, Problems, Standards, Action, Opinion) before being passed to the LLM.
