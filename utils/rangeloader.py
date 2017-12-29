import random
from sklearn.utils import shuffle

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import pin_memory_batch


class RangeLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch_ranges = self.gen_batch_ranges()
        random.shuffle(batch_ranges)
        if kwargs.get('shuffle', True):
            dataset = args[0]
            if dataset.y is not None:
                *dataset.X, dataset.y = shuffle(*dataset.X, dataset.y)
            else:
                dataset.X = shuffle(*dataset.X)
        self.batch_ranges = batch_ranges

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.batch_ranges) == 0:
            raise StopIteration
        start, stop = self.batch_ranges.pop()
        return self.prep(start, stop)

    def gen_batch_ranges(self):
        imin = 0
        imax = len(self.dataset)
        ranges = []
        start = 0
        for stop in range(self.batch_size, imax, self.batch_size):
            stop = min(imax, stop)
            ranges.append((start, stop))
            start = stop
        if stop < imax:
            ranges.append((stop, imax))
        return ranges

    def prep(self, start, stop):
        batch = self.dataset[start: stop]
        # test = default_collate(batch)
        if self.pin_memory:
            batch = pin_memory_batch(batch)
        return batch
