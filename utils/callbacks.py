from sklearn.metrics import roc_auc_score
import numpy as np


def auc_callback(batch, model, pred):
    target = batch[-1].data.numpy()
    pred = pred.data.numpy()
    idx = np.isfinite(pred)
    pred[~idx] = 0.0
    return roc_auc_score(target, pred)
