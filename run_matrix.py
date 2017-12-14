import numpy as np
import torch
import os.path

import torch.optim as optim

from utils.trainer import Trainer
from utils.callbacks import rms_callback

from models.fm import FM
from models.mf import MF
from models.mfdeep1 import MFDeep1


dim = 32
n_epochs = 40
batchsize = 4096 * 8
model_type = 'MFDeep1'
learning_rate = 1e-2
fn = model_type + '_checkpoint'


n_item = np.load('data/full.npz')['n_item'].tolist()
n_user = np.load('data/full.npz')['n_user'].tolist()
n_obs = np.load('data/full.npz')['n_obs'].tolist()
seed = np.load('data/full.npz')['seed'].tolist()

train = np.load('data/loocv_train.npz')
test = np.load('data/loocv_test.npz')
train_feat = train['train_feat'].astype('int64')
train_scor = train['train_scor'].astype('float32')
test_feat = test['test_feat'].astype('int64')
test_scor = test['test_scor'].astype('float32')


callbacks = {'rms': rms_callback}

if model_type == 'MF':
    model = MF(n_user, n_item, dim, n_obs, luv=1e-3,
               lub=1e-3, liv=1e-3, lib=1e-3)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    train_args = (user, item, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    test_args = (user, item, test_scor)
elif model_type == 'MFDeep1':
    model = MFDeep1(n_user, n_item, dim, n_obs, luv=1e-6,
                    lub=1e-6, liv=1e-6, lib=1e-6)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    train_args = (user, item, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    test_args = (user, item, test_scor)
elif model_type == 'FM':
    n_feat = n_item + n_user
    model = FM(n_feat, dim, n_obs, lb=1e-6, lv=1e-6)
    train_args = (train_feat, train_scor)
    test_args = (test_feat, test_scor)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

# Reload model if desired
if os.path.exists(fn):
    print(f"Loading from {fn}")
    model.load_state_dict(torch.load(fn))
t = Trainer(model, optimizer, batchsize=batchsize,
            callbacks=callbacks, seed=seed)
for epoch in range(n_epochs):
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
