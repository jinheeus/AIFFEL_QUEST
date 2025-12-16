import argparse
import pandas as pd
import numpy as np
import os
from evaluate_hybrid_local import run_local_evaluation


def run_repeated_stats(csv_path, base_output_name="evaluate_hybrid_results", runs=3):
    print(f"🚀 Starting Repeated Evaluation ({runs} runs)...")

    scores = []

    for i in range(runs):
        print(f"\n▶️ [Run {i + 1}/{runs}]")
        output_file = f"{base_output_name}_run_{i + 1}.csv"

        # Run Evaluation
        # We need to capture the returned dataframe or read the saved file to get the score.
        # run_local_evaluation saves to file, let's modify it or just read back.
        # It doesn't return the score directly in current implementation, prints it.
        # I'll rely on reading the saved csv.

        run_local_evaluation(csv_path, output_file, limit=None)  # Full Run

        # Read Result to get mean score
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            mean_score = df["mean_score"].mean()
            scores.append(mean_score)
            print(f"   Run {i + 1} Score: {mean_score:.4f}")
        else:
            print(f"❌ Run {i + 1} output file not found!")
            scores.append(0.0)

    # Calculate Stats
    print("\n🏆 Final Statistics")
    print(f"Scores: {scores}")

    avg = np.mean(scores)
    std = np.std(scores)

    print(f"Mean Score: {avg:.4f}")
    print(f"Standard Deviation: {std:.4f}")

    # Save Report
    with open("final_stats_report.txt", "w") as f:
        f.write(f"Runs: {runs}\n")
        f.write(f"Scores: {scores}\n")
        f.write(f"Mean: {avg:.4f}\n")
        f.write(f"Std Dev: {std:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    # Needs to match evaluate_hybrid_local default path
    CSV_PATH = "./99_archive/experiments/retrieval.csv"

    run_repeated_stats(CSV_PATH, runs=args.runs)
