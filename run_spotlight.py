import torch
import os.path
import numpy as np

from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel


train = np.load('data/loocv_train.npz')
test = np.load('data/loocv_test.npz')
train_feat = train['train_feat'].astype('int64')
train_scor = train['train_scor'][:, None].astype('float32')
test_feat = test['test_feat'].astype('int64')
test_scor = test['test_scor'][:, None].astype('float32')


model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=64,  # latent dimensionality
                                   n_iter=20,  # number of epochs of training
                                   batch_size=1024 * 4,  # minibatch size
                                   l2=1e-9,  # strength of L2 regularization
                                   learning_rate=1e-3,
                                   use_cuda=torch.cuda.is_available())

def features(feat, scor):
    user = feat[:, 0].astype('int64')
    item = feat[:, 1].astype('int64')
    y = scor[:, 0].astype('float32')
    return user, item, y


train = Interactions(*features(train_feat, train_scor))
test_user, test_item, test_y = features(test_feat, test_scor)
model.fit(train, verbose=True)
pred_y = model.predict(test_user, test_item)

rmse = np.sqrt(((pred_y - test_y)**2.0).mean())
print(rmse)
