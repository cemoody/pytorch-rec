import os
import os.path
import torch
import numpy as np
import torch.optim as optim

from utils.trainer import Trainer
from utils.callbacks import rms_callback

from models.mf import MF
from models.fm import FM
from models.vfm import VFM
from models.mfpoly2 import MFPoly2
from models.gumbel_mf import GumbelMF
from models.poincare_mf import PoincareMF


dim = 64
window = 50
n_epochs = 400
batchsize = 4096
model_type = 'PoincareMF'
learning_rate = 1e-3
fn = model_type + '_checkpoint'
torch.cuda.set_device(int(os.getenv('GPU')))


def scor_flag(scor, median=None):
    flag = np.zeros_like(scor)
    if median is None:
        median = np.median(scor)
    flag[scor > median] = 1.0
    flag[scor < median] = -1.0
    return flag, median


n_item = np.load('data/full.npz')['n_item'].tolist()
n_user = np.load('data/full.npz')['n_user'].tolist()
n_obs = np.load('data/full.npz')['n_obs'].tolist()
seed = np.load('data/full.npz')['seed'].tolist()

train = np.load('data/loocv_train.npz')
test = np.load('data/loocv_test.npz')
train_feat = train['train_feat'].astype('int64')
train_scor = train['train_scor'].astype('float32')
train_scor_tri, median = scor_flag(train_scor)
test_feat = test['test_feat'].astype('int64')
test_scor = test['test_scor'].astype('float32')
test_scor_tri, _ = scor_flag(test_scor, median)
n_feat = int(max(train_feat.max(), test_feat.max()) + 1)

# Create feature grouping for VFM
feat_group = np.zeros(n_feat).astype('int64')
# All user are group 0, items group 1, genre group 2
feat_group[:n_user] = 0
feat_group[n_user: n_user + n_item] = 1
feat_group[n_user + n_item:] = 2

callbacks = {'rms': rms_callback}
optimizer = None

if model_type == 'MF':
    model = MF(n_user, n_item, dim, n_obs, luv=1e+4,
               lub=1e+4, liv=1e+4, lib=1e+4)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    train_args = (user, item, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    test_args = (user, item, test_scor)
elif model_type == 'FM':
    model = FM(n_feat, dim, n_obs, lv=1e+0, lb=1e+0)
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    frame = train_feat[:, -1].astype('int')
    train_args = (np.vstack((user, item, frame)).T, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    frame = test_feat[:, -1].astype('int')
    test_args = (np.vstack((user, item, frame)).T, test_scor)
elif model_type == 'VFM':
    model = VFM(n_feat, dim, n_obs, feat_group)
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    frame = train_feat[:, -1].astype('int')
    train_args = (np.vstack((user, item, frame)).T, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    frame = test_feat[:, -1].astype('int')
    test_args = (np.vstack((user, item, frame)).T, test_scor)
    batchsize = 4096 * 4
    learning_rate = 1e-2
elif model_type == 'MFPoly2':
    model = MFPoly2(n_user, n_item, dim, n_obs, luv=3e+3,
                    lub=3e+3, liv=3e+3, lib=3e+3)
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    frame = train_feat[:, -1].astype('float')
    train_args = (user, item, frame, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    frame = test_feat[:, -1].astype('float')
    test_args = (user, item, frame, test_scor)
if model_type == 'GumbelMF':
    dim = 256
    model = GumbelMF(n_user, n_item, dim, n_obs, luv=1e-2,
                     lub=1e-2, liv=1e-2, lib=1e-2, tau=0.8)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    train_args = (user, item, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    test_args = (user, item, test_scor)
if model_type == 'PoincareMF':
    dim = 2
    model = PoincareMF(n_user, n_item, dim, n_obs)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    train_args = (user, item, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    test_args = (user, item, test_scor)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    learning_rate = 1e-2


model = model.cuda()
if optimizer is None:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Reload model if desired
if os.path.exists(fn):
    print(f"Loading from {fn}")
    model.load_state_dict(torch.load(fn))
t = Trainer(model, optimizer, batchsize=batchsize,
            callbacks=callbacks, seed=seed, print_every=25,
            window=window, cuda=True)
for epoch in range(n_epochs):
    model.run_disc = True
    model.train(True)
    t.fit(*train_args)
    model.train(False)
    t.test(*test_args)
    t.print_summary()
    torch.save(model.state_dict(), fn)

# Output vectors
np.savez("model", user_bas=model.embed_user.bias.weight.data.numpy(),
         user_vec=model.embed_user.vect.weight.data.numpy(),
         item_bas=model.embed_item.bias.weight.data.numpy(),
         item_vec=model.embed_item.vect.weight.data.numpy())
