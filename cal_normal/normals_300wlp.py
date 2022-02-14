import torch
import numpy as np
import scipy.io as sio
import os.path as osp
import pickle
from pathlib import Path
from tqdm import tqdm

from utils.functions import _parse_param_batch, get_normal, vertex_normals
from utils.base_params import u, w_shp, w_exp, tri, keyindex
from utils.data_utils import _numpy_to_cuda, _numpy_to_tensor, _load_cpu

_to_tensor = _numpy_to_cuda

def main():
    # load data
    root = '300W-LP-Aug/train_aug_120x120'
    filelist_train = 'train_configs/train_aug_120x120.list.train'
    filelist_val = 'train_configs/train_aug_120x120.list.val'

    param_train = 'train_configs/param_all_norm.pkl'
    param_val = 'train_configs/param_all_norm_val.pkl'

    pklfile = open('results1/normals_68_train_aug.pkl','wb')

    params = _numpy_to_tensor(_load_cpu(param_train))
    filenames = Path(filelist_train).read_text().strip().split('\n')

    results = []

    with tqdm(total=len(filenames)) as tqm: # len(filenames)
        for index in range(len(params)): # len(params)
            target = params[index]
            p, offset, alpha_shp, alpha_exp = _parse_param_batch(target)
            vertices = p @ (_numpy_to_cuda(u) +
                            _numpy_to_cuda(w_shp) @ alpha_shp +
                            _numpy_to_cuda(w_exp) @ alpha_exp).view(1, -1, 3).permute(0,2,1) + offset
            # torch.Size([1, 3, 53215])
            vertices[:,1,:] = 120 + 1 - vertices[:,1,:]

            faces = torch.tensor(tri)
            faces = faces.unsqueeze(0)  # torch.Size([1, 105840, 3])
            normals = vertex_normals(vertices.permute(0,2,1), faces)
            indexes = _to_tensor(keyindex)
            indexes = indexes.squeeze(1)
            normals_68 = torch.index_select(normals, 1, indexes.long())
            normals_results = normals_68.squeeze(0).numpy().flatten()
            # txt.write(str(list(normals_results)))
            # txt.write('\n')
            results.append(normals_results)
            results.append(normals_results)

            tqm.set_description('Processing %i' % (index))
            tqm.update()

    tqm.close()
    results = np.array(results, dtype=np.float32).reshape((len(params)*2, 68 * 3))
    pickle.dump(results, pklfile)
    pklfile.close()

    return 0

def main_weight():
    # load data
    filelist_train = 'results1/train_aug_120x120_train_aug.txt'
    filelist_val = 'results1/train_aug_120x120_val_aug.txt'

    normal_train = 'results1/normals_68_train_aug.pkl'
    normal_val = 'results1/normals_68_val_aug.pkl'

    pklfile = open('results1/normals_weight_68_val_aug.pkl','wb')

    normals = _numpy_to_tensor(_load_cpu(normal_val))
    filenames = Path(filelist_val).read_text().strip().split('\n')

    results = []
    weight_sub = torch.zeros(68, dtype = torch.float32)

    with tqdm(total=len(filenames)) as tqm: # len(filenames)
        for index in range(len(normals)): # len(params)
            target = normals[index]
            target = target.view(68,3)
            for i in range(68):
                if target[i,2] < 0:
                    weight_sub[i] = 4
                else:
                    weight_sub[i] = 16

            results.append(weight_sub.numpy())

            tqm.set_description('Processing %i' % (index))
            tqm.update()

    tqm.close()
    print(len(results))
    results = np.array(results, dtype=np.float32).reshape((len(normals), 68))
    pickle.dump(results, pklfile)
    pklfile.close()

    return 0


def main_double():
    # load data
    root = '300W-LP-Aug/train_aug_120x120'
    filelist_train = 'train_configs/train_aug_120x120.list.train'
    filelist_val = 'train_configs/train_aug_120x120.list.val'

    normal68_train = 'results/normals_68_train.pkl'
    normal68_val = 'results/normals_68_val.pkl'

    pklfile = open('results1/normals_68_train_aug.pkl','wb')

    normals = _load_cpu(normal68_train)
    filenames = Path(filelist_train).read_text().strip().split('\n')

    results = []
    with tqdm(total=len(filenames)) as tqm: # len(filenames)
        for index in range(len(normals)): # len(params)
            target = normals[index]
            results.append(target)
            results.append(target)
            tqm.set_description('Processing %i' % (index))
            tqm.update()
    tqm.close()
    print(len(results))
    results = np.array(results, dtype=np.float32).reshape((len(normals)*2, 68 * 3))
    pickle.dump(results, pklfile)
    pklfile.close()
    return 0

if __name__ == '__main__':
    # main()
    main_weight()
    # main_double()
