import pandas as pd
import numpy as np
import os
import ast
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from utils.llm import llm_openai

def run_evaluation_logic(df, run_id):
    print(f"\n[INFO] Starting Evaluation Run #{run_id}")
    
    data_dict = {
        "question": df["question"].tolist(),
        "contexts": df["contexts"].tolist(),
        "answer": df["answer"].tolist()
    }
    ragas_dataset = Dataset.from_dict(data_dict)

    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness],
        raise_exceptions=False,
        llm=llm_openai
    )
    
    scores = result.to_pandas()["faithfulness"].tolist()
    scores = [s if not np.isnan(s) else 0.0 for s in scores]
    
    df[f"faithfulness_run_{run_id}"] = scores
    
    dataset_avg = np.mean(scores)
    
    return df, dataset_avg

if __name__ == "__main__":
    DATA_FILE = "./results/agentic_generation_data.csv"
    OUTPUT_FILE = "./results/agentic_generation_data_score.csv"
    NUM_TRIALS = 3
    
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        exit()

    df = pd.read_csv(DATA_FILE, encoding="utf-8")
    
    try:
        df["contexts"] = df["contexts"].apply(ast.literal_eval)
    except:
        pass
    
    df["answer"] = df["answer"].fillna("")

    run_scores = []
    print("=" * 50)
    print("Starting Faithfulness Evaluation Process (Total 3 Runs)")
    print("=" * 50)

    for n in range(1, NUM_TRIALS + 1):
        df, score = run_evaluation_logic(df, n)
        run_scores.append(score)
        print(f"▶ Run {n} Completed | Mean Score: {score:.4f}")

    score_columns = [f"faithfulness_run_{n}" for n in range(1, NUM_TRIALS + 1)]
    df["mean_faithfulness"] = df[score_columns].mean(axis=1)

    os.makedirs("./results", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    total_mean = np.mean(run_scores)
    total_std = np.std(run_scores)

    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT (Faithfulness)")
    print("="*50)
    for i, s in enumerate(run_scores, 1):
        print(f"Run {i} : {s:.4f}")
    print("-" * 50)
    print(f"Final Result: {total_mean:.4f} ± {total_std:.4f}")
    print("="*50)
    print(f"Result saved to: {OUTPUT_FILE}")