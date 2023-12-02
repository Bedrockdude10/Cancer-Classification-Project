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
    return data, unique

def shuffle_Xy(X, y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]

def plot_confusion_matrix(y, yhat, labels, title, path=None, show=False):
    cmat = confusion_matrix(y, yhat)
    cpct = cmat/np.sum(cmat, axis=0)

    sns.heatmap(
        cpct, 
        cmap="Blues", 
        annot=True, 
        xticklabels=labels, 
        yticklabels=labels
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(title)

    if path:
        plt.savefig(path)

    if show:
        plt.show()
    
    else:
        plt.close()

def plot_roc_curves(y, yhat_proba, labels, title, path=None, show=False):

    for i in np.unique(y):
        fpr, tpr, _ = roc_curve(
            (y == i).astype(int),
            yhat_proba[:, i],
        )
        plt.plot(fpr, tpr, label=labels[i])

    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    if path:
        plt.savefig(path)

    if show:
        plt.show()
    
    else:
        plt.close()