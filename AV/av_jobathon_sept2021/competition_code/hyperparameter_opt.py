import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from utils import load_config

config = load_config("baseline_model")


def lgb_eval(
    num_leaves,
    feature_fraction,
    bagging_fraction,
    max_depth,
    sample_freq,
    min_split_gain,
    min_child_weight,
):
    opt_params = {
        "application": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "l1",
        "num_iterations": 1000,
        "boost_from_average": False,
        "verbose": -1,
        "num_leaves": int(round(num_leaves)),
        "subsample_freq": int(round(sample_freq)),
        "feature_fraction": max(min(feature_fraction, 1), 0),
        "bagging_fraction": max(min(bagging_fraction, 1), 0),
        "max_depth": int(round(max_depth)),
        "min_split_gain": min_split_gain,
        "min_child_weight": min_child_weight,
    }
    cv_result = lgb.cv(
        opt_params,
        train_set,
        nfold=5,
        seed=107,
        stratified=False,
        verbose_eval=None,
    )
    return -np.min(cv_result["l2-mean"])


# Hyperparameters optimization
param_opt = config["param_opt"]
if param_opt is True:
    train1 = pd.read_csv("data/processed/train.csv")
    train_data = train1.drop(["ID", "Sales", "Date"], axis=1)

    sales = train1["Sales"]

    train_set = lgb.Dataset(data=train_data, label=sales, free_raw_data=False)

    pbs = {
        "num_leaves": (60, 150),
        "feature_fraction": (0.1, 0.9),
        "bagging_fraction": (0.6, 1),
        "max_depth": (7, 14),
        "sample_freq": (1, 10),
        "min_split_gain": (0.001, 0.1),
        "min_child_weight": (1, 15),
    }
    bounds_transformer = SequentialDomainReductionTransformer()
    optimizer = BayesianOptimization(
        lgb_eval, pbs, random_state=207, bounds_transformer=bounds_transformer
    )
    optimizer.maximize(init_points=5, n_iter=15)

    print("Optimized Parameters: \n", optimizer.max)

    p = optimizer.max["params"]

    print(p)

    opt_params = {
        "application": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "l1",
        "num_iterations": 10000,
        "learning_rate": 0.02,
        "early_stopping_round": 50,
        "verbose": -1,
        "num_leaves": int(round(p["num_leaves"])),
        "subsample_freq": int(round(p["sample_freq"])),
        "feature_fraction": round(p["feature_fraction"], 2),
        "bagging_fraction": round(p["bagging_fraction"], 2),
        "max_depth": int(round(p["max_depth"])),
        "min_split_gain": p["min_split_gain"],
        "min_child_weight": p["min_child_weight"],
    }

    print(opt_params)

    with open("models/model_params.json", "w") as j:
        json.dump(opt_params, j, indent=2)

    print("Hyperparameters optimization finished!!!")
