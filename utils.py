from sklearn.metrics import confusion_matrix, roc_curve
from collections import defaultdict
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

def validate_files(GPL):
    files = glob(f'{GPL}/*.csv')

    ncols = None
    columns = []

    for f in files:

        # just read the columns
        df = pd.read_csv(f, nrows=1)

        # assert that all files have same n cols
        if ncols is None: ncols = df.shape[1]
        else: assert ncols == df.shape[1]

        columns += df.columns.tolist()
        columns = list(set(columns))

    # assert that all columns are the same
    assert len(columns) == ncols

def read_and_cast(path):

    df = pd.read_csv(path)
    # drop the primary key
    df = df.drop(["samples"], axis=1)

    cols = df.columns.tolist()
    cols.remove("type")

    # there are some data entry erros which cause the columns
    # coerce = send these values to nan
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float16)

    # we want tag the rows with the type of cancer
    cancer = path.split("/")[-1].split("_")[0].lower()

    # suppress pandas fragmentation warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["cancer"] = cancer

    # drop the missing values
    return df.dropna(axis=0)

def merge_files(GPL):
    files = glob(f'{GPL}/*.csv')
    fname = f"{GPL}/{GPL}.csv"

    # start with first one to get columns
    df = read_and_cast(files[0])
    df.to_csv(fname, index=False)

    # for the rest, skip columns (iloc)
    for f in tqdm(files[1:]):
        df = read_and_cast(f)
        df.to_csv(fname, mode="a", index=False, header=False)

def read_data(path):
    dtype = defaultdict(lambda: np.float16)
    dtype["cancer"] = str
    dtype["label"] = str
    dtype["type"] = str
    return pd.read_csv(path, dtype=dtype)

def label_data(df):
    df["label"] = [ x if x == "normal" else df.cancer[i] for i, x in enumerate(df.type) ]
    return df.drop(["cancer", "type"], axis=1)

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