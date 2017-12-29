import skorch
import os.path
import numpy as np
import torch.optim as optim
from skorch.net import NeuralNetRegressor
from sklearn.utils import shuffle
from skorch.dataset import CVSplit
import sklearn.metrics
import torch.nn as nn

from models.mf import MF
from models.fm import FM
from models.mfpoly2 import MFPoly2
from models.vmf import VMF
from utils.rangeloader import RangeLoader


dim = 32
window = 500
n_epochs = 40
batch_size = 2048
model_type = 'MFPoly2'
fn = model_type + '_checkpoint'
train_split = CVSplit(10)
parallel = False


n_item = np.load('data/full.npz')['n_item'].tolist()
n_user = np.load('data/full.npz')['n_user'].tolist()
n_genr = np.load('data/full.npz')['n_genr'].tolist()
n_obs = np.load('data/full.npz')['n_obs'].tolist()
seed = np.load('data/full.npz')['seed'].tolist()

train = np.load('data/loocv_train.npz')
test = np.load('data/loocv_test.npz')
train_feat = train['train_feat'].astype('int64')
train_scor = train['train_scor'][:, None].astype('float32')
test_feat = test['test_feat'].astype('int64')
test_scor = test['test_scor'][:, None].astype('float32')


def features(feat, scor, include_frame=False):
    user, item = feat[:, 0], feat[:, 1] - n_user
    frame = feat[:, -1].astype('float')
    x = [user.astype('int64'), item.astype('int64')]
    if include_frame:
        x.append(frame.astype('float32'))
    y = scor.astype('float32')
    return x, y


if model_type == 'MF':
    model = MF(n_user, n_item, dim, n_obs, luv=1,
               lub=1, liv=1, lib=1)
    train_x, train_y = features(train_feat, train_scor)
    test_x, test_y = features(test_feat, test_scor)
elif model_type == 'MFPoly2':
    model = MFPoly2(n_user, n_item, dim, n_obs,
                    luv=1e2, lub=1e2, liv=1e2, lib=1e2)
    train_x, train_y = features(train_feat, train_scor, include_frame=True)
    test_x, test_y = features(test_feat, test_scor, include_frame=True)
elif model_type == 'FM':
    n_feat = n_user + n_item + n_genr + 1
    model = FM(n_feat, dim, n_obs, lb=1e-3, lv=1e-3)
    train_x, train_y = features(train_feat, train_scor)
    test_x, test_y = features(test_feat, test_scor)
elif model_type == 'VMF':
    model = VMF(n_user, n_item, dim, n_obs, luv=1e-3,
                lub=1e-3, liv=1e-3, lib=1e-3)
    train_x, train_y = features(train_feat, train_scor)
    test_x, test_y = features(test_feat, test_scor)


if parallel:
    model = nn.DataParallel(model)


def criterion(**kwargs):
    def wrapper(prediction, target):
        if parallel:
            return model.module.loss(prediction, target)
        else:
            return model.loss(prediction, target)
    return wrapper

if False:
    # If train_split is defined, then the model will handle cross validation
    # and we'll merge train & test now and split it again later
    train_x = [np.concatenate((tx, vx)) for (tx, vx) in zip(train_x, test_x)]
    train_y = np.concatenate((train_y, test_y))


score = 'neg_mean_squared_error'
callbacks = [skorch.callbacks.EpochScoring(score, name='mse_valid')]

net = NeuralNetRegressor(model, max_epochs=200, batch_size=batch_size,
                         criterion=criterion, optimizer=optim.Adam,
                         optimizer__lr=1e-3, callbacks=callbacks,
                         verbose=1, use_cuda=True, train_split=train_split,
                         iterator_train=RangeLoader,
                         iterator_valid=RangeLoader)

net.initialize()
# net.load_params('model_final.pt')
net.fit(train_x, train_y)

# Done w/ training, sabe * eval
net.save_params('model_final.pt')
pred_y = net.predict(test_x)
valid_err = sklearn.metrics.mean_squared_error(test_y, pred_y)
print("Final test error", valid_err)


# Output vectors
model = model.cpu()
np.savez("model", user_bas=model.embed_user.bias.weight.data.numpy(),
         user_vec=model.embed_user.vect.weight.data.numpy(),
         item_bas=model.embed_item.bias.weight.data.numpy(),
         item_vec=model.embed_item.vect.weight.data.numpy())
