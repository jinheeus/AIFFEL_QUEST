import pandas as pd
import ast
import os
import sys

# Paths (v2)
input_csv = "00_data/ragas_results_v2_agentic_evaluated.csv"
output_csv = "00_data/human_evaluation_sheet_v2.csv"


# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


def prepare_sheet():
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Loading results from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Prepare list for new dataframe
    rows = []

    for idx, row in df.iterrows():
        question = row.get("question", "")
        contexts_str = row.get("contexts", "[]")

        try:
            contexts = ast.literal_eval(contexts_str)
        except:
            contexts = []

        # Create a row dictionary
        new_row = {
            "ID": idx + 1,
            "Question": question,
        }

        # Add up to 5 documents
        for i, doc in enumerate(contexts[:5]):
            new_row[f"Doc_{i + 1}"] = doc
            new_row[f"Score_{i + 1}"] = ""  # Empty for human input

        rows.append(new_row)

    human_df = pd.DataFrame(rows)

    # Reorder columns for better readability
    cols = ["ID", "Question"]
    for i in range(1, 6):
        cols.append(f"Doc_{i}")
        cols.append(f"Score_{i}")

    human_df = human_df.reindex(columns=cols)

    print(f"Saving human evaluation sheet to {output_csv}...")
    human_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(
        "Done! You can now open this CSV and fill in the 'Score_X' columns (0, 1, 2)."
    )


if __name__ == "__main__":
    prepare_sheet()
