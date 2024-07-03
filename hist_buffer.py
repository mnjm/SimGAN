import numpy as np

class HistoryBuffer:

    def __init__(self, shape, max_size, batch_size):
        self.max_size = max_size
        self.half_batch_size = batch_size // 2
        self.buffer = np.zeros(shape=(0, *shape))

    def add(self, imgs):
        if len(self.buffer) < self.max_size:
            self.buffer = np.append(self.buffer, imgs[:self.half_batch_size], axis=0)
        else:
            idxs = np.random.choice(np.arange(len(self.buffer)), self.half_batch_size, replace=False)
            self.buffer[idxs] = imgs[:self.half_batch_size]
        return

    def get(self):
        return self.buffer[:self.half_batch_size]
