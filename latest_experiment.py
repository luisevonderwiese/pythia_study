import os
from tabulate import tabulate
import pandas as pd

def get_difficulty(msa_name):
    prefix = os.path.join("data/pythia", msa_name + ".pythia")
    if not os.path.isfile(prefix):
        return float("nan")
    with open(prefix, "r") as outfile:
        lines = outfile.readlines()
        if len(lines) == 0:
            return float("nan")
        return float(lines[0])

def run_pythia(msa_name, redo):
    msa_path = os.path.join("data/msa", msa_name + ".phy")
    if not os.path.isfile(msa_path):
        print("MSA " + msa_name + " does not exist")
        return
    predictor_path = os.path.join("predictors/latest.pckl")
    prefix = os.path.join("data/pythia", msa_name + ".pythia")
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
ground_truths = {msa_name: 0 for msa_name in msa_names}
df = pd.read_parquet('all_data.parquet')
absolute_errors = []
for i, row in df.iterrows():
    ground_truths[row["verbose_name"].split(".")[0]] = row["difficult"]
res = []
for msa_name in msa_names:
    row = [msa_name, ground_truths[msa_name]]
    #run_pythia(msa_name, redo = True)
    d = get_difficulty(msa_name)
    row.append(d)
    absolute_error = abs(ground_truths[msa_name]-d)
    row.append(absolute_error)
    absolute_errors.append(absolute_error)
    res.append(row)
headers = ["MSA", "ground truth", "prediction", "error"]
#print(tabulate(res, tablefmt="pipe", floatfmt=".2f", headers = headers))

mae = sum(absolute_errors) / len(absolute_errors)
print(f"""
Overall Performance with Latest Predictor:
- Mean Absolute Error: {round(mae, 2)}
""")
