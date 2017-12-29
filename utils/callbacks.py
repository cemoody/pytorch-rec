from sklearn.metrics import roc_auc_score
# from sklearn.metrics import mean_squared_error
import numpy as np


def auc_callback(batch, model, pred):
    target = batch[-1].data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    idx = np.isfinite(pred)
    pred[~idx] = 0.0
    return roc_auc_score(target, pred)


def rms_callback(batch, model, pred):
    target = batch[-1].data.cpu().numpy()
    pred = pred.data.cpu().numpy()
    idx = np.isfinite(pred)
    pred[~idx] = 0.0
    return np.sqrt(((target - pred)**2.0).mean())
