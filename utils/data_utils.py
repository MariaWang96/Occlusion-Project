import torch
import pickle
import numpy as np
import cv2


def _get_suffix(filename):
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos+1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


_numpy_to_tensor = lambda x: torch.from_numpy(x)
_load_cpu = _load
_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))

if __name__ == '__main__':
    pass