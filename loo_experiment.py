import os
import math
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import label_wrapper

def train_loo(tdf, features, label):
    """
    Leave-one-out cross-validation
    """
    predictions = []
    prediction_mae = []
    prediction_mape = []
    training_mae = []
    training_mape = []

    for i in range(tdf.shape[0]):
        # Train a regressor using all FEATURES and the LABEL as target using all data except the i-th row
        train_df = tdf.drop(i)
        test_df = tdf.iloc[[i]]

        params = {
            "max_depth": 5,
            "num_leaves": 30
        }

        model = lgb.LGBMRegressor(**params, verbose=-1)
        model.fit(train_df[features], train_df[label])

        # Predict the i-th row
        prediction = model.predict(test_df[features])[0]
        predictions.append(prediction)

        mae = mean_absolute_error(test_df[label], [prediction])
        mape = mean_absolute_percentage_error(test_df[label], [prediction])
        prediction_mae.append(mae)
        prediction_mape.append(mape)

        # Evaluate the model on the training data
        train_predictions = model.predict(train_df[features])
        mae = mean_absolute_error(train_df[label], train_predictions)
        mape = mean_absolute_percentage_error(train_df[label], train_predictions)
        training_mae.append(mae)
        training_mape.append(mape)

    return predictions, prediction_mae, prediction_mape, training_mae, training_mape



# ordered by Pythia importance from most to least important
FEATURES = [
    'avg_rfdist_parsimony',
    'num_patterns/num_taxa',
    'bollback',
    'proportion_gaps',
    'proportion_invariant',
    'entropy',
    "num_patterns/num_sites",
    "pattern_entropy",
    'num_sites/num_taxa',
    "proportion_unique_topos_parsimony",
]

LABEL = "difficult"

label_dir = os.path.join("data", "difficulty_labels")

metadata_df = pd.read_csv("data/lexibench/character_matrices/stats.tsv", sep = "\t")
datasets = [row["Name"] for _,row in metadata_df.iterrows()]

ground_truths = {dataset: float("nan") for dataset in datasets}
tdf =  pd.DataFrame(columns = [LABEL] + FEATURES)
for i, dataset in enumerate(datasets):
    prefix = os.path.join(label_dir, dataset, "label")
    tdf.at[i, "difficult"]  = label_wrapper.get_label(prefix)
    fdf = label_wrapper.get_features(prefix)
    tdf.at[i, "avg_rfdist_parsimony"] = fdf["avg_rfdist_parsimony"]
    tdf.at[i, "num_patterns/num_taxa"] = fdf["num_patterns/num_taxa"]
    tdf.at[i, "bollback"] = fdf["bollback"]
    tdf.at[i, "proportion_gaps"] = fdf["proportion_gaps"]
    tdf.at[i, "proportion_invariant"] = fdf["proportion_invariant"]
    tdf.at[i, "entropy"] = fdf["entropy"]
    tdf.at[i, "num_patterns/num_sites"] = fdf["num_patterns/num_sites"]
    tdf.at[i, "pattern_entropy"] = fdf["pattern_entropy"]
    tdf.at[i, "num_sites/num_taxa"] = fdf["num_sites/num_taxa"]
    tdf.at[i, "proportion_unique_topos_parsimony"] = fdf["proportion_unique_topos_parsimony"]

tdf = tdf.astype(float)
predictions, prediction_mae, prediction_mape, training_mae, training_mape = train_loo(tdf, FEATURES, LABEL)
mae = mean_absolute_error(tdf[LABEL], predictions)
mape = mean_absolute_percentage_error(tdf[LABEL], predictions)


print(f"""
Overall LOO Performance:
- Mean Absolute Error: {round(mae, 2)}
- Mean Absolute Percentage Error: {round(mape, 2)}%
""")
