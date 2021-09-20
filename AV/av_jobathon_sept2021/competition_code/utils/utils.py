import yaml
import pandas as pd
import numpy as np
from typing import Tuple


def load_config(config_id: str) -> dict:
    """
    Reads Configuration from yaml file.

    Args:
        config_id (str): specific configuration ID to use in yaml file

    Returns:
        dict: Returns dictionary of configuration parameters
    """
    with open("config.yml", "r") as f:
        doc = yaml.load(f, yaml.Loader)

    config = doc[config_id]
    return config


def date_processor(df, date_col="Date"):
    df[date_col] = pd.to_datetime(df[date_col])
    df["dow"] = df[date_col].dt.day_of_week
    df["dom"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["wom"] = np.ceil(df["dom"] / 7)

    return df


def lag_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """
    Create lag features based
    Args:
        df (pd.DataFrame): Data table as pandas DataFrame
        lags (list): List of lag days

    Returns:
        pd.DataFrame: Returns lag features

    """
    out = pd.DataFrame()
    for lag in lags:
        out[f"lag_{lag}"] = (
            df["Sales"].shift(lag).replace([np.inf, -np.inf], np.nan).fillna(0)
        )

    return out


def pct_features(df: pd.DataFrame, pcts: list) -> pd.DataFrame:
    """
    Create percentage change features based
     Args:
         df (pd.DataFrame): Data table as pandas DataFrame
         pcts (list): List of percentage change days

     Returns:
         pd.DataFrame: Returns percentage change

    """
    out = pd.DataFrame()
    for pct in pcts:
        out[f"pct_{pct}"] = (
            df["Sales"]
            .shift(1)
            .pct_change(pct)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    return out


def rolling_features(
    df: pd.DataFrame, rolls: list, roll_fx: set
) -> pd.DataFrame:
    """
    Create rolling features based
    Args:
        df (pd.DataFrame): Data table as pandas DataFrame
        rolls (list): List of days for rolling features
        roll_fx (set): Function set for rolling feature.

    Returns:
        pd.DataFrame: Returns rolling features
    """
    out = pd.DataFrame()
    for roll in rolls:
        roll_df = (
            df["Sales"]
            .shift(1)
            .rolling(roll, min_periods=1)
            .agg(roll_fx)
            .add_prefix(f"roll_{roll}_")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        out = pd.concat([out, roll_df], axis=1)

    return out


def create_features(
    df: pd.DataFrame,
    rolls: list,
    lags: list,
    pcts: list,
    roll_fx: set,
    roll_feat: bool = True,
    pct_feat: bool = False,
) -> Tuple[pd.DataFrame, list]:
    """
    Create lags, rolling, and percentage change features
    Args:
        df (pd.DataFrame): Whole data as pandas DataFrame.
        rolls (list):  List of days for rolling features.
        lags (list):  List of days for lag features.
        pcts (list): List of days for percentage change features.
        roll_fx (set):  Rolling features functions set.
        roll_feat (bool): If create rolling features, Default is True.
        pct_feat (bool): If create percentage change features, Default is False.

    Returns:
        pd.DataFrame: New features DataFrame.

    """
    dfs = pd.DataFrame()
    for store_id in df["Store_id"].unique():
        store_df = df[df["Store_id"] == store_id]
        store_df = store_df.sort_values("Date")
        store_dfs = []
        # Roll features
        if roll_feat is True:
            rolls_feats = rolling_features(
                store_df, rolls=rolls, roll_fx=roll_fx
            )
            store_dfs.append(rolls_feats)
        # Lag features
        lag_feat = lag_features(store_df, lags=lags)
        store_dfs.append(lag_feat)

        # Percentage change features
        if pct_feat is True:
            pct_feat = pct_features(store_df, pcts=pcts)
            store_dfs.append(pct_feat)
        store_dfs = pd.concat(store_dfs, axis=1)

        store_dfs["ID"] = store_df["ID"]

        dfs = dfs.append(store_dfs)
    dfs.reset_index(drop=True, inplace=True)

    return dfs, list(dfs.columns[:-1])
