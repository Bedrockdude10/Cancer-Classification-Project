from sklearn.metrics import confusion_matrix, roc_curve
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_confusion_matrix(y, yhat, title, path=None):
    cmat = confusion_matrix(y, yhat)
    cpct = cmat/np.sum(cmat, axis=0)

    sns.heatmap(cpct, cmap="Blues", annot=True)
    plt.title(title)

    if path:
        plt.savefig(path)

    plt.show()

def plot_roc_curves(y, yhat_proba, title, path=None):

    for i in range(len(np.unique(y))):
        fpr, tpr, _ = roc_curve(
            (y == i).astype(int),
            yhat_proba[:, i]
        )
        plt.plot(fpr, tpr)

    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True  Positive Rate")

    if path:
        plt.savefig(path)

    plt.show()
