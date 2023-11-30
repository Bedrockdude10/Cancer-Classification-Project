from collections import defaultdict
import pandas as pd
import numpy as np

def read_data(path):
    dtype = defaultdict(lambda: np.float32)
    dtype["label"] = str
    return pd.read_csv(path, dtype=dtype)

def int_enc(data):
    unique = data["label"].unique().tolist()
    data["label"] = [ unique.index(x) for x in data["label"] ]
    return data

def shuffle_Xy(X, y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]