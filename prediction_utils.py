import math
import os
from enum import Enum
from functools import partial

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import treelite
from optuna.samplers import TPESampler
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split


class PredictionError(Enum):
    PLAIN = "plain error"
    ABS = "absolute error"
    PERC = "percentage error"


def evaluate_regressor(regressor, X, y):
    y_pred = regressor.predict(X).clip(min=0.0, max=1.0)
    df = pd.DataFrame(
        data={
            "regressor": [regressor],
            "R2 Score": [regressor.score(X, y)],
            "mse": [metrics.mean_squared_error(y, y_pred)],
            "mae": [metrics.mean_absolute_error(y, y_pred)],
            "mape": [metrics.mean_absolute_percentage_error(y, y_pred)],
            "max_error": [metrics.max_error(y, y_pred)],
            "explained_variance_score": [metrics.explained_variance_score(y, y_pred)],
            "mse_log": [metrics.mean_squared_log_error(y, y_pred)],
            "median_ae": [metrics.median_absolute_error(y, y_pred)],
        }
    )
    return df


def get_feature_importances(regressor, features):
    imp = regressor.feature_importances_

    return list(sorted(list(zip(features, imp)), key=lambda x: x[1]))


def plain_error(y_true, y_pred):
    return y_pred - y_true


def absolute_error(y_true, y_pred):
    return abs(plain_error(y_true, y_pred))


def percentage_error(y_true, y_pred):
    return abs((y_true - y_pred) / y_true)


def collect_error_data(regressor, X, y):
    y_pred = regressor.predict(X).clip(min=0.0, max=1.0)
    data = {
        "true": y,
        "predicted": y_pred,
        PredictionError.PLAIN.value: plain_error(y, y_pred),
        PredictionError.ABS.value: absolute_error(y, y_pred),
        PredictionError.PERC.value: percentage_error(y, y_pred),
    }
    return pd.DataFrame(data=data)


def plot_prediction_error(
    regressor,
    X_train,
    X_test,
    y_train,
    y_test,
    error_type: PredictionError = PredictionError.PLAIN,
    xbins=None,
):
    train_error_data = collect_error_data(regressor, X_train, y_train)
    test_error_data = collect_error_data(regressor, X_test, y_test)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=train_error_data[error_type.value],
            name="Prediction error on training batch",
            histnorm="percent",
            marker_color="LightSeaGreen",
            xbins=xbins,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=test_error_data[error_type.value],
            name="Prediction error on test batch",
            histnorm="percent",
            marker_color="MidnightBlue",
            xbins=xbins,
        )
    )

    fig.update_xaxes(title=f"Prediciton error ({error_type.value})")
    fig.update_yaxes(title="Proportion", ticksuffix="%")
    fig.update_layout(template="plotly_white", width=1000)
    return fig


def plot_plain_prediction_error_binned_by_difficulty(
    regressor,
    X_train,
    X_test,
    y_train,
    y_test,
    error_type: PredictionError = PredictionError.PLAIN,
):
    train_error_data = collect_error_data(regressor, X_train, y_train)
    train_error_data = pd.concat([train_error_data, X_train], axis=1)

    test_error_data = collect_error_data(regressor, X_test, y_test)
    test_error_data = pd.concat([test_error_data, X_test], axis=1)

    fig = go.Figure()

    gp = train_error_data.groupby(
        pd.cut(train_error_data.true, bins=np.arange(0, 1, 0.1), include_lowest=True)
    )[error_type.value]
    gp_test = test_error_data.groupby(
        pd.cut(test_error_data.true, bins=np.arange(0, 1, 0.1), include_lowest=True)
    )[error_type.value]

    for i, (diff_bin, els) in enumerate(gp):
        fig.add_trace(
            go.Bar(
                x=[str(diff_bin)],
                y=[100 * np.mean(els)],  # / diff_bin.right],
                marker_color="LightSeaGreen",
                width=0.2,
                name="Error on training batch",
                showlegend=i == 0,
            )
        )

    for i, (diff_bin, els) in enumerate(gp_test):
        fig.add_trace(
            go.Bar(
                x=[str(diff_bin)],
                y=[100 * np.mean(els)],  # / diff_bin.right],
                marker_color="MidnightBlue",
                width=0.2,
                name="Error on test batch",
                showlegend=i == 0,
            )
        )

    fig.update_xaxes(title="Difficulty")
    fig.update_yaxes(title=f"Prediciton error ({error_type.value})", range=[-15, 6])
    fig.update_layout(template="plotly_white")
    return fig


def export_c_library(
    model: lgb.LGBMRegressor,
    library_name: str = "test_prediction_library",
    target_platform: str = "unix",
    toolchain: str = "gcc",
    num_parallel_cores: int = 16,
    verbose: bool = False,
) -> None:
    if num_parallel_cores > 1:
        params = {"parallel_comp": num_parallel_cores}
    else:
        params = {}

    model.booster_.save_model("lgb_regressor.txt")
    bst = lgb.Booster(model_file="lgb_regressor.txt")
    treelite_model = treelite.Model.from_lightgbm(bst)

    treelite_model.export_srcpkg(
        platform=target_platform,
        toolchain=toolchain,
        pkgpath=os.path.join(os.getcwd(), f"{library_name}.zip"),
        libname=library_name,
        verbose=verbose,
        params=params,
    )


def lgbm_objective(trial, lgb_dataset, n_folds):  # df
    params_tune = {
        "objective": "regression",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 500),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "verbose": -1,
        "feature_pre_filter": False,
    }
    # num_iterations=trial.suggest_int("num_iterations", 100, 400, 50)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    return lgb.cv(params_tune, lgb_dataset, folds=kfold,)["l2-mean"][-1]


def optuna_optimization(X, y, n_trials):
    # params = {
    #     "learning_rate": 0.21186682461296144,
    #     "max_depth": 10,
    #     "lambda_l1": 0.0006052784011539973,
    #     "lambda_l2": 0.5438355839877388,
    #     "num_leaves": 100,
    #     "bagging_fraction": 0.8785347321084259,
    #     "bagging_freq": 1,
    #     "min_child_samples": 43,
    #     "num_iterations": 100,
    #     "objective": "regression",
    # }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    lgb_dataset = lgb.Dataset(X_train, label=y_train)

    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    # study.enqueue_trial(params)
    partial_objective = partial(lgbm_objective, lgb_dataset=lgb_dataset, n_folds=5)
    study.optimize(partial_objective, n_trials=n_trials)

    return study


def get_best_params(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
    study = optuna_optimization(X, y, n_trials=n_trials)
    return study.best_trial.params


def append_new_features(df):
    df["num_patterns/num_sites"] = df.num_patterns / df.num_sites
    df["pattern_entropy"] = df.bollback + df.num_sites * np.log(df.num_sites)


def train_with_params(df, params, features, label):
    X = df[features]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def plot_features(
    df, plot_cols, fig=None, marker_color="darkseagreen", name="all data", nbinsx=20
):
    if fig is None:
        fig = make_subplots(rows=len(plot_cols), cols=1, subplot_titles=plot_cols)

    for i, c in enumerate(plot_cols):
        lower = df[c].quantile(0.1)
        upper = df[c].quantile(0.9)
        data = df.loc[df[c].between(lower, upper)][c]

        fig.append_trace(
            go.Histogram(
                x=data,
                showlegend=i == 0,
                histnorm="percent",
                marker_color=marker_color,
                name=name,
                nbinsx=nbinsx,
            ),
            row=i + 1,
            col=1,
        )

        fig.update_xaxes(title=c, row=i + 1, col=1)

    fig.update_yaxes(title="proportion", ticksuffix="%")
    fig.update_layout(height=300 * len(plot_cols), width=1000, template="plotly_white")
    return fig
