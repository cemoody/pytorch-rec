from sklearn.metrics import roc_auc_score


def auc_callback(batch, model, pred):
    target = batch[-1].data.numpy()
    pred = pred.data.numpy()
    return roc_auc_score(target, pred)
