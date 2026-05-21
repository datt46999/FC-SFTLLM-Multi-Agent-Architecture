import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
import pandas as pd

import json
# # Downloads to ~/.cache/huggingface/hub/ by default
# data_dir = snapshot_download(
#     repo_id="gaia-benchmark/GAIA",
#     repo_type="dataset",
#     local_dir="./gaia_dataset"      # Optional: specify custom local folder
# )

# print(f"Dataset downloaded to: {data_dir}")

# # Load a specific config and split
# dataset = load_dataset(data_dir, "2023_all", split="test")

# for example in dataset:
#     question = example["Question"]
#     file_name = example["file_path"]  # may be empty string if no file
    
#     if file_name:
#         file_path = os.path.join(data_dir, file_name)
#         print(f"Q: {question}\nFile: {file_path}\n")
#     else:
#         print(f"Q: {question}\n")


data_dir = "./gaia_dataset"
combine_path = os.path.join(data_dir, "2023", "test", "metadata.parquet")
if os.path.exists(combine_path):
    df = pd.read_parquet(combine_path)
    output_path = os.path.join(data_dir, "metadata.jsonl")
    with open(output_path, "w",encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    print(f"Saved combined: {output_path} ({len(df)} rows)")
else:
    print("don't see dataset")
