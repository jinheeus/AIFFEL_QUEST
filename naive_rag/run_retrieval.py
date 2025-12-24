import pandas as pd
from graph import app

def run_retrieval_process(input_csv_path, output_csv_path):
    print(f"Loading from {input_csv_path}...")
    df = pd.read_csv(input_csv_path, encoding="utf-8")
    
    all_contexts = []
    all_contexts_idx = []

    total_queries = len(df)
    print(f"Starting Retrieval for {total_queries} queries...")

    for i, query in enumerate(df["question"], start=1):
        try:
            state = app.invoke({"question": query})
            results = state.get("documents", [])
            
            retrieved_contexts = [doc.page_content for doc in results]
            all_contexts.append(retrieved_contexts)

            retrieved_idx = [doc.metadata.get("idx") for doc in results if doc.metadata.get("idx")]
            all_contexts_idx.append(retrieved_idx)

        except Exception as e:
            print(f"Query {i} failed: {e}")
            all_contexts.append([])
            all_contexts_idx.append([])

        if i % 10 == 0:
            print(f"Processed {i}/{total_queries}")

    df["contexts"] = all_contexts
    df["contexts_idx"] = all_contexts_idx

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Saved to: {output_csv_path}")

if __name__ == "__main__":
    INPUT_FILE = "./data/retrieval.csv"
    OUTPUT_FILE = "./results/naive_retrieval_data.csv"
    
    run_retrieval_process(INPUT_FILE, OUTPUT_FILE)