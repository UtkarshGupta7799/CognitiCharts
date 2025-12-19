import pandas as pd, numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from .utils import standardize

def build_features(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index.copy())
    out["ret"] = df["close"].pct_change().fillna(0)
    out["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
    out["oc_spread"] = (df["open"] - df["close"]) / (df["close"] + 1e-8)

    rsi = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"]).macd()
    sma20 = SMAIndicator(df["close"], window=20).sma_indicator()
    sma60 = SMAIndicator(df["close"], window=60).sma_indicator()
    out["rsi"] = rsi.fillna(50)
    out["macd"] = macd.fillna(0)
    out["sma_gap"] = (sma20 - sma60) / (df["close"] + 1e-8)

    out["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean())/(df["volume"].rolling(20).std()+1e-8)
    out = out.fillna(0.0)
    return standardize(out)
