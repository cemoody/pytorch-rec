import numpy as np
import torch
import os.path

import torch.optim as optim

from trainer import Trainer
from callbacks import auc_callback
from mf import MF
from mfpoly2 import MFPoly2
from mfpoincare import MFPoincare


dim = 16
n_epochs = 2
batchsize = 4096
model_type = 'MFPoly2'
fn = model_type + '_checkpoint'

n_items = np.load('full.npz')['n_items'].tolist()
n_users = np.load('full.npz')['n_users'].tolist()
n_obs = np.load('full.npz')['n_obs'].tolist()
seed = np.load('full.npz')['seed'].tolist()

train_item = np.load('train.npz')['item'].astype('int64')
train_user = np.load('train.npz')['user'].astype('int64')
train_uage = np.load('train.npz')['uage'].astype('float32')
train_like = np.load('train.npz')['like'].astype('float32')

test_item = np.load('test.npz')['item'].astype('int64')
test_user = np.load('test.npz')['user'].astype('int64')
test_uage = np.load('test.npz')['uage'].astype('float32')
test_like = np.load('test.npz')['like'].astype('float32')

callbacks = {'auc': auc_callback}
if model_type == 'MF':
    model = MF(n_users, n_items, dim, n_obs, luv=1e-3,
               lub=1e-3, liv=1e-3, lib=1e-3)
    train_args = (train_user, train_item, train_like)
    test_args = (test_user, test_item, test_like)
elif model_type == 'MFPoly2':
    model = MFPoly2(n_users, n_items, dim, n_obs, luv=1e-3,
                    lub=1e-3, liv=1e-3, lib=1e-3)
    train_args = (train_user, train_item, train_uage, train_like)
    test_args = (test_user, test_item, test_uage, test_like)
elif model_type == 'MFPoincare':
    model = MFPoincare(n_users, n_items, dim, n_obs)
    train_args = (train_user, train_item, train_uage, train_like)
    test_args = (test_user, test_item, test_uage, test_like)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reload model if desired
if os.path.exists(fn):
    model.load_state_dict(torch.load(fn))
t = Trainer(model, optimizer, batchsize=batchsize,
            callbacks=callbacks, seed=seed)
for epoch in range(n_epochs):
    t.fit(*train_args)
    t.test(*test_args)
    t.print_summary()
    torch.save(model.state_dict(), fn)

# Output vectors
np.savez("model", user_bas=model.embed_user.bias.weight.data.numpy(),
         user_vec=model.embed_user.vect.weight.data.numpy(),
         item_bas=model.embed_item.bias.weight.data.numpy(),
         item_vec=model.embed_item.vect.weight.data.numpy())
