import h5py
import numpy as np
import sys

class DatasetIter:

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

if __name__ == "__main__":
    h5_file = sys.argv[1]
    batch_size = 256

    ds = DatasetIter(h5_file, batch_size)
    # ds.data = ds.data[:300]
    for x in ds:
        print(x.shape)
