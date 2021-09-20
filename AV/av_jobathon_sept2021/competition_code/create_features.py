import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta

from utils import load_config, create_features

pd.options.mode.chained_assignment = None

config = load_config("baseline_model")

rolls = config["rolls"]
lags = config["lags"]
roll_feats = config["roll_feats"]
roll_fx = {fx for fx in config["roll_fx"]}
lookback_days = config["lookback_days"]
pcts = config["pcts"]
pct_feats = config["pct_feats"]


train = pd.read_csv("data/interim/train.csv")
train["Date"] = pd.to_datetime(train["Date"])

out, num_cols = create_features(
    train,
    rolls=rolls,
    lags=lags,
    pcts=pcts,
    roll_fx=roll_fx,
    roll_feat=roll_feats,
    pct_feat=pct_feats,
)

train = pd.merge(train, out, on="ID", how="left")

# MinMax Scaler
scaler = MinMaxScaler()
scaler.fit(train[num_cols])
with open("models/minmax_scaler.pkl", "wb") as pkl:
    pickle.dump(scaler, pkl)

train[num_cols] = scaler.transform(train[num_cols])

# Drop first few days data because most of the features are 0s.
train = train[
    pd.to_datetime(train["Date"])
    >= pd.to_datetime(train["Date"].min()) + timedelta(lookback_days)
]
train.reset_index(drop=True, inplace=True)
train.to_csv("data/processed/train.csv", index=False)

print("Create Features finished!!!")
