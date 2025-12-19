import numpy as np, pandas as pd, yaml, os, joblib

CLASS_NAMES = ["Breakout", "Consolidation", "Reversal"]

def load_config(path="app/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def standardize(df: pd.DataFrame):
    x = (df - df.mean()) / (df.std() + 1e-8)
    return x.fillna(0.0)

def window_stack(arr, width):
    # arr: [N, F] -> [N-width+1, width, F]
    out = []
    for i in range(len(arr)-width+1):
        out.append(arr[i:i+width])
    return np.stack(out)

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
