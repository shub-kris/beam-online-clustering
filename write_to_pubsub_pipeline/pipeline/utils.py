import numpy as np
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups


def get_dataset(categories: list, split: str = "train"):
    """
    It fetches the 20 newsgroups dataset, removes headers, footers, and quotes, and returns the data and
    targets as lists

    Args:
      categories (list): list of categories to load.
      subset (str): train or test. Defaults to train

    Returns:
      A list of data and a list of targets.
    """
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    label_map = {class_name: class_id for class_id, class_name in enumerate(labels)}
    labels_subset = np.array([label_map[class_name] for class_name in categories])
    emotion_dataset = load_dataset("emotion")
    X, y = np.array(emotion_dataset[split]["text"]), np.array(
        emotion_dataset[split]["label"]
    )
    subclass_idxs = [idx for idx, label in enumerate(y) if label in labels_subset]
    X_subset, y_subset = X[subclass_idxs], y[subclass_idxs]
    return X_subset.tolist(), y_subset.tolist()
