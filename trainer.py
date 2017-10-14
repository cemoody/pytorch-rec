import time
import pandas as pd

import torch
from torch.autograd import Variable
from sklearn.utils import shuffle


def chunks(batchsize, *arrs):
    n = batchsize
    lens = [arr.shape[0] for arr in arrs]
    err = "Not all arrays are of same shape"
    length = lens[0]
    assert all(length == l for l in lens), err
    for i in range(0, length, n):
        yield [Variable(torch.from_numpy(arr[i:i + n])) for arr in arrs]


class Trainer(object):
    def __init__(self, model, optimizer, callbacks={}, seed=42,
                 print_every=25, batchsize=2048):
        self.model = model
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.previous_log = []
        self.log = []
        self._epoch = 0
        self._iteration = 0
        self.seed = seed
        self.print_every = print_every
        self.batchsize = batchsize

    def fit(self, *args):
        # args is X1, X2,...Xn, Yn
        self._iteration = 0
        args = shuffle(*args, random_state=self.seed)
        for batch in chunks(self.batchsize, *args):
            start = time.time()
            target = batch[-1]
            self.optimizer.zero_grad()
            pred = self.model.forward(*batch[:-1])
            loss = self.model.loss(pred, target)
            loss.backward()
            self.optimizer.step()
            stop = time.time()
            self.run_callbacks(batch, pred, loss=loss.data[0],
                               train=True, iter_time=stop-start)
            if self._iteration % self.print_every == 0:
                self.print_log(header=self._iteration == 0)
            self._iteration += 1
        self._epoch += 1
        self.previous_log.extend(self.log)
        self.log = []

    def test(self, *args):
        # args is X1, X2...Xn, Y
        # Where Xs are features, Y is the outcome
        for batch in chunks(self.batchsize, *args):
            pred = self.model.forward(*batch[:-1])
            self.run_callbacks(batch, pred, train=False)
            if self._iteration % self.print_every == 0:
                self.print_log(header=self._iteration == 0)

    def run_callbacks(self, batch, pred, **kwargs):
        vals = {name: cb(batch, self.model, pred)
                for (name, cb) in self.callbacks.items()}
        vals['timestamp'] = time.time()
        vals['epoch'] = self._epoch
        vals['iteration'] = self._iteration
        vals.update(kwargs)
        self.log.append(vals)

    def print_log(self, header=False):
        logs = pd.DataFrame(self.log).sort_values('timestamp')
        roll = logs.rolling(window=500).mean().reset_index()
        logs = logs.reset_index()
        concat = logs.merge(roll, how='left', on='index',
                            suffixes=('', '_rolling'))
        del concat['iter_time_rolling']
        del concat['iteration_rolling']
        del concat['timestamp_rolling']
        del concat['epoch_rolling']
        del concat['train_rolling']
        del concat['index']
        del concat['timestamp']
        line = (concat.tail(1)
                      .applymap("{0:1.4f}".format)
                      .to_string(header=header))
        print(line)

    def print_summary(self):
        logs = pd.DataFrame(self.previous_log).sort_values('timestamp')
        print('SUMMARY------')
        print(logs.groupby(('epoch', 'train')).mean())
        print('')
