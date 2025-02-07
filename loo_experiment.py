import math
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

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


tdf = pd.read_parquet("all_data.parquet").reset_index(drop=True)
tdf = tdf.rename(columns={"num_topos_parsimony/num_trees_parsimony": "proportion_unique_topos_parsimony"})
tdf["num_patterns/num_sites"] = tdf["num_patterns"] / tdf["num_sites"]
tdf["pattern_entropy"] = tdf["bollback"] + tdf["num_sites"] * tdf["num_sites"].apply(lambda x: math.log(x))

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

predictions, prediction_mae, prediction_mape, training_mae, training_mape = train_loo(tdf, FEATURES, LABEL)
mae = mean_absolute_error(tdf[LABEL], predictions)
mape = mean_absolute_percentage_error(tdf[LABEL], predictions)


print(f"""
Overall LOO Performance:
- Mean Absolute Error: {round(mae, 2)}
- Mean Absolute Percentage Error: {round(mape, 2)}%
""")
