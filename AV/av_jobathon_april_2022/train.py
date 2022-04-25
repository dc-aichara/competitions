from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

# Load data
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")


# Date features
def date_processor(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Create date features
    Args:
        df (pd.DataFrame): Input data as pandas dataframe
        date_col (str): Date column name in input dataframe

    Returns:
        pd.DataFrame: Input dataframe with added date features.

    """
    df[date_col] = pd.to_datetime(df[date_col])
    df["dow"] = df[date_col].dt.day_of_week
    df["dom"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["wom"] = np.ceil(df["dom"] / 7)
    df["wom"] = df["wom"].astype(int)

    return df


train = train.sort_values(by=["date", "hour"])
train.reset_index(drop=True, inplace=True)
train = date_processor(df=train, date_col="date")

test = test.sort_values(by=["date", "hour"])
test.reset_index(drop=True, inplace=True)
test = date_processor(df=test, date_col="date")

# LightGBM training parameters
params = {
    "application": "tweedie",
    "tweedie_variance_power": 1.1,
    "num_iterations": 10000,
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "subsample_freq": 1,
    "metric": "rmse",
    "verbose": -1,
    "min_data_in_leaf": 50,
    "device_type": "cpu",
}

# Keep last 4 months data for validation
date_split_data = "2020-11-01"
train_data = train[train["date"] < date_split_data]
valid_data = train[train["date"] >= date_split_data]
print(len(train_data), len(valid_data))

X_train = train_data.drop(["date", "demand"], axis=1)
y_train = train_data["demand"]

X_valid = valid_data.drop(["date", "demand"], axis=1)
y_valid = valid_data["demand"]

# keep columns order to use in inference
col_order = X_train.columns

lgb_train = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
lgb_valid = lgb.Dataset(data=X_valid, label=y_valid, free_raw_data=False)

early_stopping_callback = lgb.early_stopping(20)
eval_callback = lgb.log_evaluation(period=100)
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[early_stopping_callback, eval_callback],
)

# Feature Importance
feat_imp = model.feature_importance(importance_type="gain")
feat_importance = pd.DataFrame(
    {"features": X_train.columns, "importance": feat_imp}
).sort_values(by="importance", ascending=False)

model.save_model("models/demand_model.txt", num_iteration=model.best_iteration)
feat_importance.to_csv("models/feat_importance.csv", index=False)

# Predictions
test["demand"] = model.predict(test[col_order])
submission = test[["date", "hour", "demand"]]

date = datetime.today().strftime("%Y-%m-%d")
submission["demand"] = np.round(submission["demand"], 0).astype(int)
submission.to_csv(f"data/submissions/submission_{str(date)}.csv", index=False)

print(submission.tail())
print(submission.describe())
