import pandas as pd
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from utils import load_config

pd.options.mode.chained_assignment = None

config = load_config("baseline_model")

rolls = config["rolls"]
lags = config["lags"]
roll_feats = config["roll_feats"]
roll_fx = {fx for fx in config["roll_fx"]}
seed = config["random_seed"]
lookback_days = config["lookback_days"]
pcts = config["pcts"]
pct_feats = config["pct_feats"]
param_type = config["param_type"]
params = config[param_type]
param_opt = config["param_opt"]
if param_opt is True:
    params = json.load(open("models/model_params.json"))

train = pd.read_csv("data/processed/train.csv")
train["Date"] = pd.to_datetime(train["Date"])

train_data = train.drop(["ID", "Sales", "Date"], axis=1)
sales = train["Sales"]
col_order = pd.DataFrame({"col_order": train_data.columns})
col_order.to_csv("models/columns_order.csv", index=False)

X_train, X_valid, y_train, y_valid = train_test_split(
    train_data, sales, test_size=0.2, random_state=seed
)

lgb_train = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
lgb_valid = lgb.Dataset(data=X_valid, label=y_valid, free_raw_data=False)

model = lgb.train(
    params, lgb_train, valid_sets=[lgb_train, lgb_valid], verbose_eval=400
)

# Feature Importance
feat_imp = model.feature_importance(importance_type="gain")
feat_importance = pd.DataFrame(
    {"features": X_train.columns, "importance": feat_imp}
).sort_values(by="importance", ascending=False)

model.save_model("models/sales_model.txt", num_iteration=model.best_iteration)
feat_importance.to_csv("models/feat_importance.csv", index=False)

print("Model training finished!!!")
