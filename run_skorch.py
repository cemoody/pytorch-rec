import torch
import pickle
import skorch
import os.path
import numpy as np
import torch.optim as optim
from skorch.net import NeuralNetRegressor
from sklearn.utils import shuffle
from skorch.dataset import CVSplit
import sklearn.metrics
import torch.nn as nn
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

from models.mf import MF
from models.fm import FM
from models.mfpoly2 import MFPoly2
from models.vmf import VMF
from utils.rangeloader import RangeDataLoader


window = 500
n_epochs = 41
max_loops = 40
batch_size = 2048 * 4
model_type = 'MFPoly2'
fn = model_type + '_checkpoint'
train_split = CVSplit(10)
parallel = False
torch.cuda.set_device(int(os.getenv('GPU')))


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


def make_model(model_type, luv, lub, liv, lib, dim):
    dim = 8 * int(dim)
    if model_type == 'MF':
        model = MF(n_user, n_item, dim, n_obs,
                   luv=luv, lub=lub, liv=liv, lib=lib)
        train_x, train_y = features(train_feat, train_scor)
        test_x, test_y = features(test_feat, test_scor)
    elif model_type == 'MFPoly2':
        model = MFPoly2(n_user, n_item, dim, n_obs,
                        luv=luv, lub=lub, liv=liv, lib=lib)
        train_x, train_y = features(train_feat, train_scor, include_frame=True)
        test_x, test_y = features(test_feat, test_scor, include_frame=True)
    elif model_type == 'FM':
        n_feat = n_user + n_item + n_genr + 1
        model = FM(n_feat, dim, n_obs, lb=lub, lv=luv)
        train_x, train_y = features(train_feat, train_scor)
        test_x, test_y = features(test_feat, test_scor)
    elif model_type == 'VMF':
        model = VMF(n_user, n_item, dim, n_obs,
                    luv=luv, lub=lub, liv=liv, lib=lib)
        train_x, train_y = features(train_feat, train_scor)
        test_x, test_y = features(test_feat, test_scor)
    if parallel:
        model = nn.DataParallel(model)
    return model, train_x, train_y, test_x, test_y


def save(input, err):
    if os.path.exists('log'):
        log = pickle.load(open('log', 'rb'))
    else:
        log = dict(x0=[], y0=[])
    log['x0'].append(input)
    log['y0'].append(err)
    pickle.dump(log, open('log', 'wb'))


def func(input):
    lr = 1e-3
    luv, lub, liv, lib = 1.0, 1.0, 1.0, 1.0
    model_type, wd, dim = input
    print(input)
    score = 'neg_mean_squared_error'
    callbacks = [skorch.callbacks.EpochScoring(score, name='mse_valid')]
    model, tx, ty, vx, vy = make_model(model_type, luv, lub, liv, lib, dim)

    net = NeuralNetRegressor(model, max_epochs=5, batch_size=batch_size,
                             criterion=torch.nn.MSELoss, optimizer=optim.Adam,
                             optimizer__weight_decay=wd,
                             optimizer__lr=lr, callbacks=callbacks,
                             verbose=1, use_cuda=True, train_split=train_split,
                             warm_start=True,
                             iterator_train=RangeDataLoader,
                             iterator_valid=RangeDataLoader,
                             iterator_train__shuffle=True,
                             iterator_valid__shuffle=True)

    net.initialize()
    last = np.inf
    for _ in range(max_loops):
        net.fit(tx, ty)
        curr = np.mean(net.history[-1, 'batches', :, 'train_loss'])
        frac = abs(curr - last) / curr
        last = curr
        if frac < 1e-3:
            print("Threshold met, converged")
            break

    # Done w/ training, sabe * eval
    # net.save_params('model_final.pt')
    py = net.predict(vx)
    valid_err = sklearn.metrics.mean_squared_error(vy, py)
    print("Final test error", valid_err)
    save(input, valid_err)
    return valid_err


space = [Categorical(['MFPoly2']),  #, 'MFPoly2', 'FM', 'VMF']),  # model_type
         Real(10**-6, 10**-1, 'log-uniform'),  # wd
         Integer(1, 12),                    # dim
        ]

x0, y0 = None, None
# if os.path.exists('log'):
#     log = pickle.load(open('log', 'rb'))
#     x0, y0 = log['x0'], log['y0']
#     print(x0, y0)
res = gp_minimize(func, space, verbose=True, x0=x0, y0=y0)
