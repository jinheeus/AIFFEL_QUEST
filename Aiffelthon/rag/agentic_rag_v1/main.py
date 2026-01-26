from graph import app

def run_rag():
    print("\n" + "="*60)
    print("[ SYSTEM ] AGENTIC RAG SYSTEM STARTED")
    print("[ INFO   ] EXIT: 'q' or 'quit'")
    print("="*60)
    
    while True:
        user_input = input("\n[ QUESTION ] > ")
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\n[ SYSTEM ] RAG SYSTEM TERMINATED")
            print("="*60)
            break
            
        if not user_input.strip():
            continue
            
        inputs = {"question": user_input}
        final_docs = []
        
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    print("-" * 60)
                    print(f"[ NODE     ] {key.upper()}")
                    
                    if key == "date_extract":
                        extracted_date = value.get("filter_date", "")
                        original_q = value.get("original_question", "")
                        if original_q:
                            print(f"[ ORIGINAL ] Q: {original_q}")
                        print(f"[ FILTER   ] DATE: {extracted_date if extracted_date else 'NO_CONDITION'}")
                    
                    elif key == "rewrite":
                        new_q = value.get("question", "")
                        cot = value.get("rewrite_cot", [])
                        print(f"[ REWRITE  ] NEW QUERY: {new_q}")
                        if cot:
                            print(f"[ STRATEGY ]")
                            for step in cot:
                                print(f"   - {step}")

                    elif key == "field_select":
                        fields = value.get("selected_fields", [])
                        cot = value.get("selected_fields_cot", [])
                        print(f"[ FIELDS   ] {', '.join(fields)}")
                        if cot:
                            print(f"[ REASONING ]")
                            for step in cot:
                                print(f"   - {step}")
                    
                    elif key == "retrieve":
                        docs = value.get("documents", [])
                        print(f"[ RETRIEVE ] UNIQUE DOCUMENTS FOUND: {len(docs)}")
                        for i, doc in enumerate(docs):
                            source = doc.metadata.get("source", "UNK")
                            snippet = doc.page_content.replace('\n', ' ')[:80]
                            print(f"   [{source:^4}] {i+1:02d}. [idx:{doc.metadata.get('idx')}] {snippet}...")
                    
                    elif key == "rerank":
                        ranked_docs = value.get("documents", [])
                        print(f"[ RERANK   ] RERANKING COMPLETED")
                        for i, doc in enumerate(ranked_docs):
                            source = doc.metadata.get("source", "UNK")
                            snippet = doc.page_content.replace('\n', ' ')[:80]
                            print(f"   [*] [{source:^4}] {i+1:02d}. [idx:{doc.metadata.get('idx')}] {snippet}...")

                    elif key == "validate":
                        results = value.get("validation_results", [])
                        final_docs = value.get("validated_documents", [])
                        retry_count = value.get("retry_count", 0)
                        
                        valid_count = sum(1 for r in results if r.get("is_valid") == "yes")
                        print(f"[ VALIDATE ] VALID: {valid_count} / TOTAL CHECKED: {len(results)} (RETRY: {retry_count}/3)")
                        
                        for i, res in enumerate(results):
                            status = res.get("is_valid", "no").upper()
                            cot = res.get("validator_cot", [])
                            print(f"   {i+1:02d}. PRED: {status}")
                            if cot:
                                print(f"       [ LOGIC ]")
                                for step in cot:
                                    print(f"       - {step}")

                    elif key == "generate":
                        answer = value.get("answer")
                        cot = value.get("answer_cot", [])
                        
                        print(f"[ THINKING PROCESS (CoT) ]")
                        if cot:
                            for step in cot:
                                print(f"   - {step}")
                        else:
                            print("   (No CoT information available)")
                        
                        print(f"\n[ FINAL ANSWER ]")
                        print(f"   {answer}")
                        
                    print("-" * 60)

            print("\n" + "="*60)
            print(f"[ REFERENCE ] TOP {len(final_docs)} VALIDATED DOCUMENTS USED")
            print("="*60)
            
            if final_docs:
                for i, doc in enumerate(final_docs):
                    idx = doc.metadata.get("idx", "UNK")
                    source = doc.metadata.get("source", "UNK")
                    
                    valid_reason = "Approved by Validator"
                    if "validator_cot" in doc.metadata and doc.metadata["validator_cot"]:
                         valid_reason = doc.metadata["validator_cot"][-1]

                    print(f"\nREF {i+1} | Source: {source} | ID: {idx}")
                    print(f"REASON: {valid_reason}")
                    print(f"CONTENT: {doc.page_content[:200].replace(chr(10), ' ')}...") 
                    print("-" * 30)
            else:
                print("No valid documents found satisfying strict criteria.")
                
            print("="*60)

        except Exception as e:
            print(f"\n[ ERROR    ] {e}")

if __name__ == "__main__":
    run_rag()