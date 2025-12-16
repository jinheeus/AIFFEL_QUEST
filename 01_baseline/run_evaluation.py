import sys
import os
import argparse

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from pipeline import RAGPipelineV0
from common.evaluate.run_evaluation import run_pipeline_on_dataset
from common.evaluate.evaluate_ragas import evaluate_metrics

if __name__ == "__main__":
    # Define paths
    input_csv = os.path.join(root_dir, "00_data", "ragas.csv")
    output_results_csv = os.path.join(
        root_dir, "00_data", "ragas_results_v0_baseline.csv"
    )
    output_evaluated_csv = os.path.join(
        root_dir, "00_data", "ragas_results_v0_baseline_evaluated.csv"
    )

    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at {input_csv}")
        sys.exit(1)

    # 1. Initialize Pipeline
    print("Initializing RAGPipelineV0...")
    pipeline = RAGPipelineV0()

    # 2. Run Pipeline on Dataset
    print("Running pipeline on dataset...")
    run_pipeline_on_dataset(pipeline, input_csv, output_results_csv)

    # 3. Evaluate with Ragas
    print("Evaluating results with Ragas...")
    evaluate_metrics(output_results_csv, output_evaluated_csv)
