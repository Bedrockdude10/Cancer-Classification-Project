from sklearn.metrics import confusion_matrix, roc_curve
from collections import defaultdict
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

######################### Data Prepocessing Functions #########################

def validate_files(GPL):
    """
    Validate files by making sure all the csv's in the given directory
    have the same columns (genes).

    GPL: a directory containing all files of a GPL type.
    """
    # get a list of all csv in the dir
    files = glob(f'{GPL}/*.csv')

    ncols = None
    columns = set()
    for f in files:

        # just read the columns
        df = pd.read_csv(f, nrows=1)

        # assert that all files have same n cols
        if ncols is None: ncols = df.shape[1]
        else: assert ncols == df.shape[1]

        # union these columns
        columns.union(set(df.columns.tolist()))

    # assert that all columns are the same
    assert len(columns) == ncols

def read_and_cast(path):
    """
    Read a csv file, drop the primary key, convert features
    to a numeric data typeand drop nan rows.
    Additionally, label each row with the type of cancer in
    question.

    path: a path to a csv file.
    """

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

    # suppress pandas df fragmentation warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["cancer"] = cancer

    # drop the missing values
    return df.dropna(axis=0)

def merge_files(GPL):
    """
    Merge all files with the same GPL number into one large datasets.

    GPL: a directory containing all files of a GPL type.
    """
    files = glob(f'{GPL}/*.csv')
    fname = f"{GPL}/{GPL}.csv"

    # start with the first file to get columns
    df = read_and_cast(files[0])
    df.to_csv(fname, index=False)

    # for the rest, skip columns (iloc)
    for f in tqdm(files[1:]):
        df = read_and_cast(f)
        # append the dataframe (not column names)
        df.to_csv(fname, mode="a", index=False, header=False)

def read_data(path):
    """
    Read the merged dataset with correct data types specified
    to avoid loading the data as 'object' type.

    path: path to a merged dataset.
    """
    dtype = defaultdict(lambda: np.float16)
    dtype["cancer"] = str
    dtype["label"] = str
    dtype["type"] = str
    return pd.read_csv(path, dtype=dtype)

def label_data(data):
    """
    Label the dataset based on the following scheme:
        if healthy of any type -> normal,
        if tumoral of any type -> type of cancer.
    Additionally, drop the now extraneous label and cancer cols.

    data: a pandas dataframe with the data containing cancer and type fields.
          clarification: type is healthy, tumoral, etc., cancer is the type of 
                         cancer as determined by what file the row comes from.
    """
    data["label"] = [ 
        x if x == "normal" else data.cancer[i] for i, x in enumerate(data.type) 
    ]
    return data.drop(["cancer", "type"], axis=1)

def int_enc(data):
    """
    Encode the categorical target variable as an integer.

    data: pandas dataframe with 'label' already constructed.
    """
    unique = data["label"].unique().tolist()
    data["label"] = [ unique.index(x) for x in data["label"] ]
    return data, unique

def shuffle_Xy(X, y):
    """
    Randomly shuffle the X and y arrays while preserving
    the correspondence between the two.

    X: numpy array with feature data (rows are observations)
    y: numpy array with target variable (int encoded)
    """
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]

######################### Data Visualization Functions ########################

def plot_confusion_matrix(y, yhat, labels, title, path=None, show=False):
    """
    Given the predicted and true labels, plot the confusion matrix along
    with class labels.
    Optionally save the output to a png.
    Optionally display the plot in the nb.

    y: numpy array of true labels (int encoded).
    yhat: numpy array of predicted labels (also ints).
    labels: string labels corresponding to the encoded classes.
    title: a title for the plot.
    path: location to save the plot (optional).
    show: boolean, whether or not to display the plot.
    """

    # calulate the confusion matrix as a pct of total
    cmat = confusion_matrix(y, yhat)
    cpct = cmat/np.sum(cmat, axis=0)

    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cpct, cmap="Blues")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    plt.setp(ax.get_yticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, round(cpct[i, j], 2),
                        ha="center", va="center", 
                        color=("w" if cpct[i, j] > .25 else "black"))

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(title)
    plt.tight_layout()

    # optionally save figure
    if path:
        plt.savefig(path)

    # optionally show figure
    if show:
        plt.show()
    
    # otherwise, close the plot, we are done.
    else:
        plt.close()

def plot_roc_curves(y, yhat_proba, labels, title, path=None, show=False):
    """
    Given a set of true classes and the predicted probabilities, plot
    the reciever operator characteristic curve for each class.
    Optionally save the output to a png.
    Optionally display the plot in the nb.

    y: numpy array of true labels (int encoded).
    yhat_proba: numpy array of predicted probabilities.
    labels: string labels corresponding to the encoded classes.
    title: a title for the plot.
    path: location to save the plot (optional).
    show: boolean, whether or not to display the plot.
    """

    fig, ax = plt.subplots()

    # for each class
    for i in np.unique(y):
        # find the false positive rate and true positive rate
        # of the binary classification problem associated to 
        # this class (ie. 1 for this class 0 for all others).
        fpr, tpr, _ = roc_curve(
            (y == i).astype(int), # 1 for this class 0 for others
            yhat_proba[:, i],     # probabilities of this class
        )
        # plot the fpr vs. tpr with corresponding label
        ax.plot(fpr, tpr, label=labels[i])

    # give a title, label axes, provide legend
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.legend()

    # optionally save figure
    if path:
        plt.savefig(path)

    # optionally show figure
    if show:
        plt.show()
    
    # otherwise, close the plot, we are done.
    else:
        plt.close()

def plot_feature_analysis(total_contribution, features, title, n=30, path=None, show=False):
    """
    Visualize the contribution of features to the final output by plotting the distribution
    of contribution (in a boxplot) and the top n features (in a bar plot).

    total_contribution: array, net (absolute) contribution of each feature (sum over class)
    features: a list of feature labels (should have len = len(total_contribution))
    title: title for the plot
    n: the number of significant features to display in the box plot
    path: location to save image
    show: boolean, to display or not to display, that is the question.
    """
    sorted_idx = np.argsort(total_contribution)[::-1]
    top_n = sorted_idx[:n]

    label = list(np.array(features)[top_n])

    fig, axe = plt.subplots(figsize=(8, 6), ncols=2, width_ratios=[1, .25], sharey=True)

    axe[0].bar(range(n), total_contribution[top_n])
    axe[0].set_xticks(range(n), labels=label, rotation=90)
    axe[0].set_xlabel("Gene")
    axe[0].set_ylabel("Contribution")
    axe[0].set_title("Most Important Features")

    axe[1].boxplot([ total_contribution ])
    axe[1].set_xticks([])
    axe[1].set_title("Distribution")
    axe[1].set_ylabel("Contribution")

    plt.suptitle(title)
    # pls stop cutting off the xlabels...
    plt.tight_layout()
    
    # optionally save figure
    if path:
        plt.savefig(path)

    # optionally show figure
    if show:
        plt.show()
    
    # otherwise, close the plot, we are done.
    else:
        plt.close()