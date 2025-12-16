import sys
import os
import pandas as pd
from tqdm import tqdm


def run_pipeline_on_dataset(pipeline, input_path, output_path):
    """
    Runs the given pipeline on the dataset found at input_path and saves results to output_path.
    The pipeline object must have a .run(query) method and a .retriever.invoke(query) method (or similar).

    For HighContextRAGPipeline, we might need to adjust how contexts are retrieved if .retriever isn't standard.
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Starting evaluation on {len(df)} questions...")

    # Lists to store results
    answers = []
    contexts_list = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]

        if pd.isna(question) or str(question).strip() == "":
            answers.append("")
            contexts_list.append([])
            continue

        try:
            # 1. Retrieve Contexts
            # We need to handle different pipeline structures.
            # If pipeline has 'search_and_merge', use that to get contexts.
            if hasattr(pipeline, "search_and_merge"):
                # HighContextRAGPipeline
                contexts = pipeline.search_and_merge(question)
                # search_and_merge returns list of strings (full docs)
            elif hasattr(pipeline, "retriever"):
                # Standard LangChain Pipeline
                retrieved_docs = pipeline.retriever.invoke(question)
                contexts = [doc.page_content for doc in retrieved_docs]
            else:
                contexts = []
                print("Warning: Pipeline has no known retrieval method.")

            # 2. Generate Answer
            answer = pipeline.run(question)

            answers.append(answer)
            contexts_list.append(contexts)

        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            answers.append("Error")
            contexts_list.append([])

    # Update DataFrame
    df["answer"] = answers
    df["contexts"] = contexts_list

    # Save results
    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Done!")
