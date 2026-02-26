import sys
import os
import pandas as pd
import ast
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Add root directory to sys.path to import config
current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(common_dir)
sys.path.append(root_dir)

from common.config import Config
from common.model_factory import ModelFactory


def evaluate_metrics(input_path, output_path):
    print(f"Loading results from {input_path}...")
    df = pd.read_csv(input_path)

    # Convert string representation of list to actual list for 'contexts'
    if "contexts" in df.columns:
        df["contexts"] = df["contexts"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # Prepare Dataset for Ragas
    ragas_data = {
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].tolist(),
    }

    dataset = Dataset.from_dict(ragas_data)

    # Initialize LLM and Embeddings for Evaluation
    print(f"Initializing Evaluation Model (Provider: {Config.EVAL_PROVIDER})...")
    # Use Factory (Light) for evaluation
    llm = ModelFactory.get_eval_model(level="light", temperature=0.0)

    # We stick to OpenAI Embeddings for now as Ragas prefers it, or make it configurable if needed
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Define metrics
    metrics = [faithfulness, answer_relevancy]

    print("Starting Ragas evaluation...")
    results = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    print("Evaluation complete!")
    print(results)

    # Add scores back to DataFrame
    df["faithfulness"] = results["faithfulness"]
    df["answer_relevancy"] = results["answer_relevancy"]

    # Save updated results
    print(f"Saving evaluated results to {output_path}...")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Done!")
