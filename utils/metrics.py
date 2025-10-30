import numpy as np

def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()

def sensitivity(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))
    return tp / (tp + fn + 1e-8)

def specificity(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    return tn / (tn + fp + 1e-8)