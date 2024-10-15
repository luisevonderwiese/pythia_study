import os
from tabulate import tabulate
from Bio import AlignIO
import pandas as pd

def get_difficulty(msa_name, predictor_name):
    prefix = os.path.join("data/pythia", predictor_name, msa_name + ".pythia")
    if not os.path.isfile(prefix):
        return float("nan")
    with open(prefix, "r") as outfile:
        lines = outfile.readlines()
        if len(lines) == 0:
            return float("nan")
        return float(lines[0])

def run_pythia(msa_name, predictor_name, redo):
    msa_path = os.path.join("data/msa", msa_name + ".phy")
    if not os.path.isfile(msa_path):
        print("MSA " + msa_name + " does not exist")
        return
    AlignIO.read(msa_path, "phylip-relaxed")
    predictor_path = os.path.join("predictors", predictor_name + ".pckl")
    if not os.path.isfile(predictor_path):
        print("Predictor " + predictor_name + " does not exist")
        return
    prefix = os.path.join("data/pythia", predictor_name, msa_name + ".pythia")
    if os.path.isfile(prefix) and not redo:
        print("Files with prefix " + prefix + " already exist")
        return
    prefix_dir = os.path.dirname(prefix)
    if not os.path.isdir(prefix_dir):
        os.makedirs(prefix_dir)
    command = "pythia -m " + msa_path + " -o " + prefix + " -r bin/raxml-ng -p " + predictor_path + " --removeDuplicates -v"
    print(command)
    os.system(command)


msa_names = [name.split(".")[0] for name in os.listdir("data/msa")]
predictor_names = ["latest"]
ground_truths = {msa_name: 0 for msa_name in msa_names}
predictions = {predictor_name: {msa_name: 0 for msa_name in msa_names} for predictor_name in predictor_names}
absolute_errors = {predictor_name: [] for predictor_name in predictor_names}
df = pd.read_parquet('all_data.parquet')
for i, row in df.iterrows():
    ground_truths[row["verbose_name"].split(".")[0]] = row["difficult"]
res = []
for msa_name in msa_names:
    row = [msa_name, ground_truths[msa_name]]
    for predictor_name in predictor_names:
        #run_pythia(msa_name, predictor_name, redo = True)
        d = get_difficulty(msa_name, predictor_name)
        row.append(d)
        predictions[predictor_name][msa_name] = d
        absolute_error = abs(ground_truths[msa_name]-d)
        row.append(absolute_error)
        absolute_errors[predictor_name].append(absolute_error)
    res.append(row)
headers = ["MSA", "ground truth"] + [predictor_name + suffix for predictor_name in predictor_names for suffix in [ " pred.", " error"]]
print(tabulate(res, tablefmt="pipe", floatfmt=".2f", headers = headers))

for predictor_name in predictor_names:
    error_list = absolute_errors[predictor_name]
    mae = sum(error_list) / len(error_list)
    print(predictor_name, str(mae))
