from graph import app

def run_rag():
    print("\n" + "="*60)
    print("[ SYSTEM ] NAIVE_RAG_SYSTEM_STARTED (Retriever + Generator)")
    print("[ INFO   ] EXIT: 'q' or 'quit'")
    print("="*60)
    
    while True:
        user_input = input("\n[ QUESTION ] > ")
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\n[ SYSTEM ] RAG_SYSTEM_TERMINATED")
            print("="*60)
            break
            
        if not user_input.strip():
            continue
            
        inputs = {"question": user_input}
        
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    print("-" * 60)
                    print(f"[ NODE     ] {key.upper()}")
                    
                    if key == "retrieve":
                        docs = value.get("documents", [])
                        stats = value.get("search_stats", {})
                        
                        count = stats.get('vector', 0)
                        print(f"[ STATS    ] RETRIEVED: {count} DOCS")
                        
                        print(f"\n[ SEARCH RESULTS ]")
                        if not docs:
                            print("   No documents found.")
                        else:
                            for i, doc in enumerate(docs):
                                idx = doc.metadata.get('idx')
                                snippet = doc.page_content.replace('\n', ' ')[:100]
                                print(f"   {i+1:02d}. [idx:{idx}] {snippet}...")
                    
                    elif key == "generate":
                        answer = value.get("answer")
                        cot = value.get("answer_cot", [])
                        
                        print(f"\n[ THINKING PROCESS (CoT) ]")
                        if cot:
                            for step in cot:
                                print(f"   {step}")
                        else:
                            print("   (No CoT information available)")
                            
                        print(f"\n[ FINAL ANSWER ]")
                        print(f"   {answer}")
                        
                    print("-" * 60)
                        
        except Exception as e:
            print(f"\n[ ERROR    ] {e}")

if __name__ == "__main__":
    run_rag()