import pandas as pd, numpy as np
from .utils import CLASS_NAMES

def load_prices(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def label_patterns(df, lookback=60):
    # Heuristic labels for demo:
    # Breakout: today's close near highest close in lookback (slightly relaxed)
    # Consolidation: recent std of returns is small
    # Reversal: sign flip after a local trend (slightly relaxed)
    ret = df["close"].pct_change().fillna(0)
    rolling_max = df["close"].rolling(lookback).max()

    # relaxed thresholds (were 0.999 and -0.001)
    breakout = (df["close"] >= rolling_max * 0.995)

    vola = ret.rolling(lookback//4).std().fillna(ret.std())
    consolidation = vola < vola.quantile(0.35)

    trend = ret.rolling(lookback//3).mean().fillna(0)
    reversal = (trend.shift(1) * ret) < -0.0005

    y = np.where(breakout, 0, np.where(consolidation, 1, np.where(reversal, 2, 1)))
    df["label"] = y
    return df

def train_test_split_time(df, test_ratio=0.2, val_ratio=0.1):
    n = len(df)
    n_test = int(n*test_ratio)
    n_val = int(n*val_ratio)
    train = df.iloc[: n - n_test - n_val]
    val   = df.iloc[n - n_test - n_val : n - n_test]
    test  = df.iloc[n - n_test : ]
    return train, val, test
