from graph import app

def run_rag():
    print("\n" + "="*50)
    print("[ SYSTEM ] RAG_SYSTEM_STARTED")
    print("[ INFO   ] EXIT: 'q' or 'quit'")
    print("="*50)
    
    while True:
        user_input = input("\n[ QUESTION ] > ")
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\n[ SYSTEM ] RAG_SYSTEM_TERMINATED")
            print("="*50)
            break
            
        if not user_input.strip():
            continue
            
        inputs = {"question": user_input}
        
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    print("-" * 50)
                    print(f"[ NODE     ] {key.upper()}")
                    
                    if key == "date_extract":
                        extracted_date = value.get("filter_date", "")
                        print(f"[ FILTER   ] DATE: {extracted_date if extracted_date else 'NO_CONDITION'}")
                    
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
                        
                        vec_docs = [d for d in docs if d.metadata.get('source') in ['VEC', 'BOTH']]
                        bm25_docs = [d for d in docs if d.metadata.get('source') in ['BM25', 'BOTH']]
                        
                        print(f"\n[ 1. VECTOR SEARCH RESULT ] COUNT: {len(vec_docs)}")
                        for i, doc in enumerate(vec_docs):
                            snippet = doc.page_content.replace('\n', ' ')[:80]
                            print(f"   {i+1:02d}. [idx:{doc.metadata.get('idx')}] {snippet}...")

                        print(f"\n[ 2. BM25 SEARCH RESULT   ] COUNT: {len(bm25_docs)}")
                        for i, doc in enumerate(bm25_docs):
                            snippet = doc.page_content.replace('\n', ' ')[:80]
                            print(f"   {i+1:02d}. [idx:{doc.metadata.get('idx')}] {snippet}...")

                        print(f"\n[ 3. FINAL UNIQUE MERGED  ] COUNT: {len(docs)}")
                        for i, doc in enumerate(docs):
                            source = doc.metadata.get("source", "UNK")
                            snippet = doc.page_content.replace('\n', ' ')[:80]
                            print(f"   [{source:^4}] {i+1:02d}. [idx:{doc.metadata.get('idx')}] {snippet}...")
                    
                    elif key == "rerank":
                        ranked_docs = value.get("documents", [])
                        print(f"[ RERANK   ] RERANKING COMPLETED ([*] IS TOP 5)")
                        for i, doc in enumerate(ranked_docs):
                            mark = "[*]" if i < 5 else "[ ]"
                            source = doc.metadata.get("source", "UNK")
                            snippet = doc.page_content.replace('\n', ' ')[:80]
                            print(f"   {mark} [{source:^4}] {i+1:02d}. [idx:{doc.metadata.get('idx')}] {snippet}...")
                            
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
                        
                    print("-" * 50)
                        
        except Exception as e:
            print(f"\n[ ERROR    ] {e}")

if __name__ == "__main__":
    run_rag()