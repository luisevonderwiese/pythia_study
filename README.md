# Pythia Study

A study to examine, whether machine learning based tools from phylogenetics can be applied to language data using [Pythia 2.0](https://tschuelia.github.io/PyPythia/) as an example.

## Setting up the environment
```
conda env create -f environment.yml
conda activate pythia-study
```

## Preparing the data
Clone the [glottolog repo](https://github.com/glottolog/glottolog) to a directory of your choice, then run:
```
lexibench --repos data/lexibench download --upgrade
lexibench --repos data/lexibench lingpy_wordlists
lexibench --repos data/lexibench character_matrices --formats bin.phy --glottolog <your_glottolog_path>
```

## Calculate Ground Truth
```
python calculate_labels.py
```


## Running the experiments
```
python latest_experiment.py
python loo_experiment.py
```
