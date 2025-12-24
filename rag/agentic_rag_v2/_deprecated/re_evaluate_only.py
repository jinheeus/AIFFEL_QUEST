import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from common.evaluate.evaluate_agentic import evaluate_agentic_metrics
from common.evaluate.evaluate_ragas import evaluate_metrics

# The file that already contains the answers from the last run
target_csv = os.path.join(root_dir, "00_data", "ragas_results_v2_agentic_evaluated.csv")

if __name__ == "__main__":
    print(f"Re-evaluating {target_csv} with Ragas + Custom Judge...")

    # 1. Ragas Evaluation (Faithfulness, Relevancy)
    print("Step 1: Running Ragas Evaluation...")
    evaluate_metrics(target_csv, target_csv)

    # 2. Custom Agentic Evaluation (Routing, Persona)
    # print("Step 2: Running Custom Agentic Evaluation...")
    # evaluate_agentic_metrics(target_csv, target_csv)
    # print("Re-evaluation complete.")
