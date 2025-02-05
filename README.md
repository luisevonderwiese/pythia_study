# Pythia Study

A study to examine, whether machine learning based tools from phylogenetics can be applied to language data using [Pythia](https://tschuelia.github.io/PyPythia/) as an example.
We use the predictor `latest.pckl` from [here](https://github.com/tschuelia/PyPythia/tree/50bf34ae2361696a7dff9cbedaf2bdb46441e8a2/pypythia/predictors)

## Preparing the data
* Follow the instructions [here](https://github.com/luisevonderwiese/difficulty-prediction-training-data/tree/tree_characterization) for computing ground truth difficulties and predictions features for data from [lexibench](https://github.com/lexibank/lexibench)
* Copy `lexibench_bin_msas` to `data/msa`
* Copy `results_lexibench/all_data.parquet` to this directory

## Setting up the environment
```
conda env create -f environment.yml
conda activate pythia-study
```

## Running the experiments
```
python latest_experiment.py
python loo_experiment.py
```
