import numpy as np
import torch
import os.path

import torch.optim as optim

from utils.trainer import Trainer
from utils.callbacks import rms_callback

from models.mf import MF
from models.mfdeep1 import MFDeep1
from models.mfgan import MFGAN
from models.mfpoly2 import MFPoly2


dim = 64
window = 50
n_epochs = 400
batchsize = 4096
model_type = 'MFPoly2'
learning_rate = 1e-3
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
elif model_type == 'MFPoly2':
    model = MFPoly2(n_user, n_item, dim, n_obs, luv=1e-0,
                    lub=1e-0, liv=1e-0, lib=1e-0)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    frame = train_feat[:, -1].astype('float')
    train_args = (user, item, frame, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    frame = test_feat[:, -1].astype('float')
    test_args = (user, item, frame, test_scor)
elif model_type == 'MFDeep1':
    model = MFDeep1(n_user, n_item, dim, n_obs, luv=1e-3,
                    lub=1e-3, liv=1e-3, lib=1e-3, lmat=1.0)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    train_args = (user, item, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    test_args = (user, item, test_scor)
elif model_type == 'MFGAN':
    model = MFGAN(n_user, n_item, dim, n_obs, luv=1.0,
                  lub=1, liv=1, lib=1, lmat=1.0)
    # The first two columns give user and item indices
    user, item = train_feat[:, 0], train_feat[:, 1] - n_user
    train_args = (user, item, train_scor)
    user, item = test_feat[:, 0], test_feat[:, 1] - n_user
    test_args = (user, item, test_scor)
    window = 50


model = model.cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

# Reload model if desired
if os.path.exists(fn):
    print(f"Loading from {fn}")
    model.load_state_dict(torch.load(fn))
t = Trainer(model, optimizer, batchsize=batchsize, clip=1,
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
