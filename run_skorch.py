import skorch
import os.path
import numpy as np
import torch.optim as optim
from skorch.net import NeuralNetRegressor
from sklearn.utils import shuffle
import sklearn.metrics

from models.mf import MF
from models.fm import FM
from models.mfpoly2 import MFPoly2
from models.vmf import VMF
from utils.rangeloader import RangeLoader


dim = 32
window = 500
n_epochs = 40
batchsize = 4096 * 8
model_type = 'MFPoly2'
learning_rate = 1e-3
fn = model_type + '_checkpoint'


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
    model = MF(n_user, n_item, dim, n_obs, luv=1e-3,
               lub=1e-3, liv=1e-3, lib=1e-3)
    train_x, train_y = features(train_feat, train_scor)
    test_x, test_y = features(test_feat, test_scor)
elif model_type == 'MFPoly2':
    model = MFPoly2(n_user, n_item, dim, n_obs)
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


def criterion(**kwargs):
    def wrapper(prediction, target):
        return model.loss(prediction, target)
    return wrapper


net = NeuralNetRegressor(model, max_epochs=100, batch_size=4096 * 4,
                         criterion=criterion, optimizer=optim.Adagrad,
                         optimizer__lr=1e-0,
                         verbose=1, use_cuda=True,
                         iterator_train=RangeLoader, train_split=None,
                         iterator_valid=RangeLoader)
net.initialize()
if os.path.exists('model_final.pt'):
    print("Reloading from last checkpoint")
    net.load_params('model_final.pt')
net.fit(train_x, train_y)
import pdb; pdb.set_trace()
pred_y = net.predict(test_x)
valid_err = sklearn.metrics.mean_squared_error(test_y, pred_y)
print("Final test error", valid_err)
net.save_params('model_final.pt')


# Output vectors
model = model.cpu()
np.savez("model", user_bas=model.embed_user.bias.weight.data.numpy(),
         user_vec=model.embed_user.vect.weight.data.numpy(),
         item_bas=model.embed_item.bias.weight.data.numpy(),
         item_vec=model.embed_item.vect.weight.data.numpy())