
import numpy as np
import pandas as pd
from pathlib import Path

def load_times(csv_path):
    df = pd.read_csv(csv_path)
    if 't' not in df.columns:
        raise ValueError("CSV must contain column 't' (cumulative failure times).")
    t = df['t'].values.astype(float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("'t' must be strictly increasing.")
    return t

def cum_to_intervals(t):
    t = np.asarray(t, dtype=float)
    d = np.diff(np.concatenate([[0.0], t]))
    return d

def train_valid_split(t, train_n=28):
    if train_n >= len(t):
        raise ValueError("train_n must be less than total sample size.")
    return t[:train_n], t[train_n:]

def ensure_output_dir(path='output'):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
