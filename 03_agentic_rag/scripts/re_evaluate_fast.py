import os
import sys

# Force OpenAI Provider for Evaluation (Bypass Gemini Rate Limit)
os.environ["EVAL_PROVIDER"] = "openai"

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(project_root)

from common.evaluate.evaluate_ragas import evaluate_metrics
from common.evaluate.evaluate_agentic import evaluate_agentic_metrics

if __name__ == "__main__":
    data_dir = os.path.join(project_root, "00_data")
    input_csv = os.path.join(data_dir, "ragas_results_v3_agentic.csv")
    final_output_csv = os.path.join(
        data_dir, "ragas_results_v3_agentic_evaluated_fast.csv"
    )

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        sys.exit(1)

    print(f"Starting Fast Evaluation with OpenAI (gpt-4o-mini)...")

    # 1. Ragas Evaluation
    # 1. Ragas Evaluation (SKIPPED for Time Efficiency - We just want KPR now)
    print("Skipping Ragas evaluation (using existing results)...")
    # evaluate_metrics(input_csv, final_output_csv)

    # Use the ALREADY EVALUATED file (with Ragas scores) as input
    evaluated_csv = os.path.join(data_dir, "ragas_results_v3_agentic_evaluated.csv")

    # 2. Agentic Evaluation
    print("Running Agentic evaluation (with KPR)...")
    # evaluate_agentic_metrics overwrites the file with new columns
    evaluate_agentic_metrics(evaluated_csv, evaluated_csv)

    print("Fast Evaluation Complete!")
