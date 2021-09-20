import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb

from datetime import datetime, timedelta

from utils import load_config, date_processor, create_features

pd.options.mode.chained_assignment = None

config = load_config("baseline_model")

cat_cols = config["cat_cols"]
rolls = config["rolls"]
lags = config["lags"]
roll_feats = config["roll_feats"]
roll_fx = {fx for fx in config["roll_fx"]}
lookback_days = config["lookback_days"]
pcts = config["pcts"]
pct_feats = config["pct_feats"]

# Load models
model = lgb.Booster(model_file="models/sales_model.txt")

with open("models/ordinal_encoder.pkl", "rb") as pkl:
    ordnl = pickle.load(pkl)

with open("models/minmax_scaler.pkl", "rb") as pkl:
    scaler = pickle.load(pkl)
columns_order = pd.read_csv("models/columns_order.csv")
col_order = columns_order["col_order"]

train = pd.read_csv("data/interim/train.csv")
train["Date"] = pd.to_datetime(train["Date"])

test = pd.read_csv("data/raw/test.csv")
test[cat_cols] = ordnl.transform(test[cat_cols])
test = date_processor(test)


# Streaming predictions
test["Sales"] = 0
train_tail = train[
    train["Date"] >= test["Date"].min() - timedelta(days=lookback_days)
]
test = pd.concat([train_tail, test], axis=0)
test.reset_index(drop=True, inplace=True)
out, _ = create_features(
    test,
    rolls=rolls,
    lags=lags,
    pcts=pcts,
    roll_fx=roll_fx,
    roll_feat=roll_feats,
    pct_feat=pct_feats,
)
out = pd.merge(test, out, on="ID", how="left")
test = out

preds = []
for date in test["Date"].unique()[lookback_days:]:
    print(str(date))
    df_date = test[
        (test["Date"] <= date)
        & (test["Date"] >= pd.to_datetime(date) - timedelta(days=lookback_days))
    ]
    feats, num_cols = create_features(
        df_date,
        rolls=rolls,
        lags=lags,
        pcts=pcts,
        roll_fx=roll_fx,
        roll_feat=roll_feats,
        pct_feat=pct_feats,
    )
    test_date = pd.merge(
        df_date.drop(num_cols, axis=1), feats, on="ID", how="left"
    )
    test_date[num_cols] = scaler.transform(test_date[num_cols])
    test_date = test_date[col_order]
    df_date["Sales"] = model.predict(test_date)
    test.update(df_date["Sales"][-365:])

sub = pd.read_csv("data/submissions/SAMPLE.csv")
submission = pd.merge(sub["ID"], test[["ID", "Sales"]], on="ID", how="left")

date = datetime.today().strftime("%Y-%m-%d")
submission["Sales"] = np.round(submission["Sales"], 2)
submission.to_csv(f"data/submissions/submission_{str(date)}.csv", index=False)

print(submission.head())

print("Model Serving finished!!!")
