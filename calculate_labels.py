import os
import label_wrapper
import pandas as pd

metadata_df = pd.read_csv("data/lexibench/character_matrices/stats.tsv", sep = "\t")
datasets = [row["Name"] for _,row in metadata_df.iterrows()]

label_dir = os.path.join("results", "difficulty_labels")

for dataset in datasets:
    msa_path = os.path.join("data/lexibench/character_matrices/", dataset, "bin.phy")
    label_wrapper.calculate_label(msa_path, os.path.join(label_dir, dataset, "label"))

