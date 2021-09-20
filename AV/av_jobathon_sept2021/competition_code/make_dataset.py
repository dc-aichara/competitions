import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

from utils import load_config, date_processor

pd.options.mode.chained_assignment = None

config = load_config("baseline_model")
cat_cols = config["cat_cols"]
# Load_data
train = pd.read_csv("data/raw/train.csv")

del train["#Order"]

# OrdinalEncoder for categorical features
ordnl = OrdinalEncoder()

ordnl.fit(train[cat_cols])
with open("models/ordinal_encoder.pkl", "wb") as pkl:
    pickle.dump(ordnl, pkl)

train[cat_cols] = ordnl.transform(train[cat_cols])

# Create date features
train = date_processor(train)
# Save to interim data folder
train.to_csv("data/interim/train.csv", index=False)

print("Make Dataset finished!!!")
