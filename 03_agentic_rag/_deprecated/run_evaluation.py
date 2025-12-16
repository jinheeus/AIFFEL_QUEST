import sys
import os
import argparse

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from graph import app
from common.evaluate.evaluate_ragas import evaluate_metrics
from common.evaluate.evaluate_agentic import evaluate_agentic_metrics
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # Define paths
    data_dir = os.path.join(root_dir, "00_data")
    input_csv = os.path.join(data_dir, "ragas_v2.csv")
    output_csv = os.path.join(data_dir, "ragas_results_v3_agentic.csv")
    final_output_csv = os.path.join(data_dir, "ragas_results_v3_agentic_evaluated.csv")

    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at {input_csv}")
        sys.exit(1)

    # Wrapper class to adapt LangGraph app to the evaluation interface
    class AgenticRAGWrapper:
        def __init__(self, app):
            self.app = app
            self.last_query = None
            self.last_result = None

        def search_and_merge(self, query: str):
            """
            Runs the full agentic flow to get both documents and answer.
            Caches the result so .run() doesn't need to re-execute.
            """
            print(f"Processing query: {query}")
            self.last_query = query
            self.last_result = self.app.invoke({"query": query})

            # Return documents + graph_context as full context for evaluation
            docs = self.last_result.get("documents", [])
            graph_ctx = self.last_result.get("graph_context", [])

            # Ensure both are lists
            if not isinstance(docs, list):
                docs = [str(docs)]
            if not isinstance(graph_ctx, list):
                graph_ctx = [str(graph_ctx)]

            return docs + graph_ctx

        def run(self, query: str):
            """
            Returns the answer from the cached result.
            """
            if query == self.last_query and self.last_result:
                return self.last_result.get("answer", "No answer generated.")

            # If for some reason search_and_merge wasn't called first
            result = self.app.invoke({"query": query})
            return result.get("answer", "No answer generated.")

    # 1. Initialize Pipeline
    print("Initializing AgenticRAGPipeline...")
    pipeline = AgenticRAGWrapper(app)

    # 2. Run Pipeline on Dataset (Custom Loop for Agentic Metadata)
    print("Running pipeline on dataset...")
    df = pd.read_csv(input_csv)

    answers = []
    contexts_list = []
    categories = []
    personas = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        if pd.isna(question) or str(question).strip() == "":
            answers.append("")
            contexts_list.append([])
            categories.append("")
            personas.append("")
            continue

        try:
            # Run Agentic Flow
            # AgenticRAGWrapper.search_and_merge calls app.invoke and caches result
            # We can access the full state from the wrapper's last_result
            contexts = pipeline.search_and_merge(question)  # This triggers the flow
            answer = pipeline.run(question)  # This gets the cached answer

            # Extract Metadata from cached result
            last_result = pipeline.last_result
            category = last_result.get("category", "unknown")
            persona = last_result.get("persona", "unknown")

            answers.append(answer)
            contexts_list.append(contexts)
            categories.append(category)
            personas.append(persona)

        except Exception as e:
            print(f"Error: {e}")
            answers.append("Error")
            contexts_list.append([])
            categories.append("error")
            personas.append("error")

    df["answer"] = answers
    df["contexts"] = contexts_list
    df["category"] = categories
    df["persona"] = personas

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved intermediate results with metadata to {output_csv}")

    # 3. Evaluate with Ragas (Standard)
    print("Running Ragas evaluation...")
    evaluate_metrics(output_csv, final_output_csv)

    # 3-1. Bypass Ragas: Copy results to evaluated file for Custom Eval
    # import shutil
    # shutil.copy(output_csv, final_output_csv)

    # 4. Evaluate with LLM-as-a-Judge (Custom Agentic)
    print("Running Custom Agentic Evaluation (LLM-as-a-Judge)...")

    evaluate_agentic_metrics(
        final_output_csv, final_output_csv
    )  # Overwrite or new file? Overwrite for now.
