import os
import util
import pandas as pd

def get_features(prefix):
    return pd.read_csv(prefix + ".csv")
    
def label_command(msa_path, prefix):
    command = "label"
    command += " -m " + msa_path
    command += " -i  bin/iqtree2"
    command += " -p " + prefix
    os.system(command)

def get_label(prefix):
    with open(prefix + ".labelGen.log", "r", encoding="utf-8") as out_file:
        lines = out_file.readlines()
    for line in lines:
        if line.startswith("Ground Truth Difficulty"):
            return float(line.split(" ")[-1])
    raise ValueError("Error during label calculation")

def calculate_label(msa_path, prefix):
    try:
        get_label(prefix)
    except (FileNotFoundError, ValueError):
        print("calculating difficulty label")
        label_command(msa_path, prefix)
        try:
            get_label(prefix)
        except ValueError:
            print("trying with padded msa")
            util.write_padded_msa(msa_path, "temp.phy")
            label_command("temp.phy", prefix)
            os.remove("temp.phy")
