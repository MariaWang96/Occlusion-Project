import os.path as osp
import numpy as np
import scipy.io as sio
from .data_utils import _load, _to_ctype

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)


d = make_abs_path('../train_configs')

keypoints = _load(osp.join(d, 'keypoints_sim.npy'))

w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))

u_shp = _load(osp.join(d, 'u_shp.npy'))
u_exp = _load(osp.join(d, 'u_exp.npy'))
u = u_shp + u_exp

meta = _load(osp.join(d, 'param_whitening.pkl'))
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')

w = np.concatenate((w_shp, w_exp), axis=1)
w_base = w[keypoints]
w_norm = np.linalg.norm(w, axis=0)
w_base_norm = np.linalg.norm(w_base, axis=0)

dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1,1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]

std_size = 120

# load tri for render
tri = sio.loadmat(osp.join(d, 'tri.mat'))
tri = tri['tri'] - 1
tri = _to_ctype(tri.T).astype(np.int32)

keyindex = sio.loadmat(osp.join(d, 'keyindex.mat'))
keyindex = keyindex['keyindex'].astype(np.int32)
keyindex = (keyindex - 1).T