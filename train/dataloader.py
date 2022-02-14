import os
import os.path as osp
from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np

from utils.data_utils import _numpy_to_tensor, _load_cpu, img_loader
from utils.base_params import *

def _parse_param_batch(param):
    N = param.shape[0]
    p_ = param[:, :12].view(N, 3, -1) # 3x4
    p = p_[:, :, :3]
    offset = p_[:, :, -1].view(N, 3, 1)
    alpha_shp = param[:, 12:52].view(N, -1, 1)
    alpha_exp = param[:, 52:].view(N, -1, 1)
    return p, offset, alpha_shp, alpha_exp

def _parse_param(param):
    p_ = param[:12].reshape(3, -1) # 3x4
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape( -1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp

def reconstruct_vertices(param, mat = True):

    param = param * param_std + param_mean
    p, offset, alpha_shp, alpha_exp = _parse_param(param)
    vertices = p @ (u +
                    w_shp @ alpha_shp +
                    w_exp @ alpha_exp).reshape(3, -1, order='F') \
               + offset
    if mat:
        vertices[1,:] = std_size + 1 - vertices[1,:]

    return vertices

def scale_to_450(vertex, roi_box):
    pts3d = vertex.copy()
    # to 450 for compare
    sx, sy, ex, ey = eval(roi_box)
    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    # pts3d[2, :] = -1 * pts3d[2, :]
    return np.array(pts3d, dtype=np.float32)

class ToTensorG(object):

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2,0,1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__+'()'

class NormalizeG(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self,tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

class TrainValDataset(data.Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.img_loader = img_loader

    def _target_loader(self, index):
        target = self.params[index]
        return target

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)

        target = self._target_loader(index)

        if self.transform is not None:
            img = self.transform(img)

        return img, target,

    def __len__(self):
        return len(self.lines)

class TestDataset(data.Dataset):
    def __init__(self, root, filelists, roilists, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.lines_filename = Path(filelists).read_text().strip().split('\n')
        self.lines_roi = Path(roilists).read_text().strip().split('\n')
        self.img_loader = img_loader

    def _roi_filename_loader(self, index):
        filename = self.lines_filename[index]
        roi = self.lines_roi[index]
        return roi, filename

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines_filename[index])
        img = self.img_loader(path)

        roi, fn = self._roi_filename_loader(index)

        if self.transform is not None:
            img = self.transform(img)

        return img, roi, fn

    def __len__(self):
        return len(self.lines_filename)

class TestDataset1(data.Dataset):
    def __init__(self, root, filelists, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.lines_filename = Path(filelists).read_text().strip().split('\n')
        self.img_loader = img_loader

    def _roi_filename_loader(self, index):
        filename = self.lines_filename[index]
        return filename

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines_filename[index])
        img = self.img_loader(path)

        fn = self._roi_filename_loader(index)

        if self.transform is not None:
            img = self.transform(img)

        return img, fn

    def __len__(self):
        return len(self.lines_filename)

# -------------------------------------------------------------------------- #
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Save checkpoint to {}'.format(filename))


class TrainValDataset_Normal(data.Dataset):
    def __init__(self, root, filelists, param_fp, wnormal_pth, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.normals_weight = _numpy_to_tensor(_load_cpu(wnormal_pth))
        self.img_loader = img_loader

    def _target_loader(self, index):
        params = self.params[index]
        normals_weight = self.normals_weight[index]
        return params, normals_weight

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)

        params, normals_weight = self._target_loader(index)

        if self.transform is not None:
            img = self.transform(img)

        return img, params, torch.diag(normals_weight)

    def __len__(self):
        return len(self.lines)