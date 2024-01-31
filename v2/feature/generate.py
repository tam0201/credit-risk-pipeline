import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool


def one_hot_encoding(df, cols, is_drop=True):
    for col in cols:
        dummies = pd.get_dummies(pd.Series(df[col]), prefix="onehot_%s"%col)
        df = pd.concat([df, dummies], axis=1)
    if is_drop:
        df = df.drop(cols, axis=1, inplace=False)
    return df

def get_difference(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame:
    """
    Create diff feature
    Args:
        df: dataframe
        num_features: list of numerical features
    Returns:
        dataframe
    """

    df_diff = (
        df.loc[:, num_features + ["customer_ID"]]
        .groupby(["customer_ID"])
        .progress_apply(lambda x: np.diff(x.values[-2:, :], axis=0).squeeze().astype(np.float32))
    )
    cols = [col + "_diff1" for col in df[num_features].columns]
    df_diff = pd.DataFrame(df_diff.values.tolist(), columns=cols, index=df_diff.index)
    df_diff.reset_index(inplace=True)

    return df_diff

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features
    Args:
        df: dataframe
    Returns:
        dataframe
    """
    # FEATURE ENGINEERING FROM

    all_cols = [c for c in list(df.columns) if c not in ["customer_ID", "S_2"]]
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features + ["preds"]]

    # Get the difference
    df_diff_agg = get_difference(df, num_features)

    num_features = [col for col in all_cols if col not in cat_features]
    df_num_agg = df.groupby("customer_ID")[num_features].agg(["first", "mean", "std", "min", "max", "last"])
    df_num_agg.columns = ["_".join(x) for x in df_num_agg.columns]
    df_num_agg.reset_index(inplace=True)

    df_cat_agg = df.groupby("customer_ID")[cat_features].agg(["count", "first", "last", "nunique"])
    df_cat_agg.columns = ["_".join(x) for x in df_cat_agg.columns]
    df_cat_agg.reset_index(inplace=True)

    # Transform int64 columns to int32
    cols = list(df_num_agg.dtypes[df_num_agg.dtypes == "float64"].index)
    df_num_agg.loc[:, cols] = df_num_agg.loc[:, cols].progress_apply(lambda x: x.astype(np.float32))

    # Transform int64 columns to int32
    cols = list(df_cat_agg.dtypes[df_cat_agg.dtypes == "int64"].index)
    df_cat_agg.loc[:, cols] = df_cat_agg.loc[:, cols].progress_apply(lambda x: x.astype(np.int32))

    df = df_num_agg.merge(df_cat_agg, how="inner", on="customer_ID").merge(df_diff_agg, how="inner", on="customer_ID")

    del df_num_agg, df_cat_agg, df_diff_agg
    gc.collect()

    return df
