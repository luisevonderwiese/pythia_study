import os
from tabulate import tabulate
import pandas as pd
import label_wrapper 
import util



def get_difficulty(prefix):
    if not os.path.isfile(prefix + ".pythia.log"):
        raise ValueError("Error during difficulty prediction")
    with open(prefix + ".pythia.log", "r", encoding="utf-8") as outfile:
        lines = outfile.readlines()
    for line in lines:
        if line.startswith("The predicted difficulty"):
            return float(line.split(" ")[-1])
    raise ValueError("Error during difficulty prediction")

def run_pythia(msa_path, prefix):
    command = "pythia -m " + msa_path + " -p " + prefix + " -r bin/raxml-ng"
    os.system(command)

def run_pythia_safe(dataset):
    msa_path = os.path.join("data/lexibench/character_matrices/", dataset, "bin.phy")
    if not os.path.isfile(msa_path):
        print("MSA " + dataset + " does not exist")
        return
    prefix = os.path.join("data/pythia", dataset)
    prefix_dir = os.path.dirname(prefix)
    if not os.path.isdir(prefix_dir):
        os.makedirs(prefix_dir)
    run_pythia(msa_path, prefix)
    try:
        get_difficulty(prefix)
    except ValueError:
        util.write_padded_msa(msa_path, "temp.phy")
        run_pythia("temp.phy", prefix)
        os.remove("temp.phy")
    

metadata_df = pd.read_csv("data/lexibench/character_matrices/stats.tsv", sep = "\t")
datasets = [row["Name"] for _,row in metadata_df.iterrows()]

label_dir = os.path.join("data", "difficulty_labels")
ground_truths = {dataset: float("nan") for dataset in datasets}
absolute_errors = []
for dataset in datasets:
    ground_truths[dataset] = label_wrapper.get_label(os.path.join(label_dir, dataset, "label"))

res = []
if not os.path.isdir("data/pythia"):
    os.makedirs("data/pythia")

for dataset in datasets:
    row = [dataset, ground_truths[dataset]]
    run_pythia_safe(dataset)
    prefix = os.path.join("data/pythia", dataset)
    d = get_difficulty(prefix)
    row.append(d)
    absolute_error = abs(ground_truths[dataset] - d)
    row.append(absolute_error)
    absolute_errors.append(absolute_error)
    res.append(row)
headers = ["MSA", "ground truth", "prediction", "error"]
print(tabulate(res, tablefmt="pipe", floatfmt=".2f", headers=headers))

mae = sum(absolute_errors) / len(absolute_errors)
print(f"""
Overall Performance with Latest Predictor:
- Mean Absolute Error: {round(mae, 2)}
""")
