import h5py
import numpy as np
import sys

class H5DatasetIter:

    def __init__(self, h5_path, batch_size):
        with h5py.File(h5_path, 'r') as root:
            h_imgs = root['image']
            print(type(h_imgs))
            assert isinstance(h_imgs, h5py.Group), "Invalid h5 file"
            self.data = np.zeros((len(h_imgs), 35, 55, 1), dtype=np.float32)
            for i, img in enumerate(h_imgs.values()):
                self.data[i] = np.expand_dims(img, -1)
        self.start_idx = 0
        self.batch_size = batch_size
        return

    def __iter__(self): return self

    def __next__(self):
        start_idx, end_idx = self.start_idx, self.start_idx + self.batch_size
        le = len(self.data)
        ret = self.data[start_idx:end_idx, ...]
        if end_idx > le:
            end_idx -= le
            ret = np.vstack((ret, self.data[:end_idx]))
        self.start_idx = end_idx
        ret = (ret - 128.0) / 128.0
        return ret

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

if __name__ == "__main__":
    h5_file = sys.argv[1]
    batch_size = 256

    ds = H5DatasetIter(h5_file, batch_size)
    # ds.data = ds.data[:300]
    for x in ds:
        print(x.shape)
