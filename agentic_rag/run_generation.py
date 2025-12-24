import pandas as pd
from graph import app

def run_generation_process(input_csv_path, output_csv_path):
    print(f"Loading from {input_csv_path}...")
    df = pd.read_csv(input_csv_path, encoding="utf-8")
    
    all_contexts = []
    all_answers = []

    total_queries = len(df)
    print(f"Starting Generation for {total_queries} queries...")

    for i, query in enumerate(df["question"], start=1):
        try:
            state = app.invoke({"question": query})
            
            if state.get("validated_documents"):
                results = state.get("validated_documents")
            else:
                results = state.get("documents", [])
            
            retrieved_contexts = [doc.page_content for doc in results]
            all_contexts.append(retrieved_contexts)

            final_answer = state.get("answer") or ""
            all_answers.append(final_answer)

        except Exception as e:
            print(f"Query {i} failed: {e}")
            all_contexts.append([])
            all_answers.append("")

        if i % 10 == 0:
            print(f"Processed {i}/{total_queries}")

    df["contexts"] = all_contexts
    df["answer"] = all_answers

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Saved to: {output_csv_path}")

if __name__ == "__main__":
    INPUT_FILE = "./data/generation.csv"
    OUTPUT_FILE = "./results/agentic_generation_data.csv"
    
    run_generation_process(INPUT_FILE, OUTPUT_FILE)