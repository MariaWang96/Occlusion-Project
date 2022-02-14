import torch
import torch.nn.functional as F
import torch.utils.data as data
from pathlib import Path
import numpy as np

from utils.data_utils import _load_cpu, _numpy_to_tensor
from utils.base_params import param_std, param_mean, std_size, u, w_shp, w_exp

def _parse_param_batch(target):
    param = target * param_std + param_mean
    p_ = param[:12].view(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].view(3, 1)
    alpha_shp = param[12:52].view(-1, 1)
    alpha_exp = param[52:].view(-1, 1)
    return p, offset, alpha_shp, alpha_exp
def _parse_param(param):
    p_ = param[:12].reshape(3, -1) # 3x4
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape( -1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp

def vertex_normals(vertices, faces):
    """
    :param vertices: B X N X 3
    :param faces: B X F X 3
    :return: B X N X 3
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    normals = torch.zeros(bs * nv, 3)

    faces = faces + (torch.arange(bs, dtype=torch.int32) * nv)[:, None, None]
    vertices_face = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_face = vertices_face.view(-1, 3, 3) # torch.Size([105840, 3, 3])

    normals.index_add_(0, faces[:, 1].long(),
                      torch.cross(vertices_face[:, 2] - vertices_face[:, 1],
                                  vertices_face[:, 0] - vertices_face[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                      torch.cross(vertices_face[:, 0] - vertices_face[:, 2],
                                  vertices_face[:, 1] - vertices_face[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                      torch.cross(vertices_face[:, 1] - vertices_face[:, 0],
                                  vertices_face[:, 2] - vertices_face[:, 0]))
    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    return normals

class Dataset300WLP(data.Dataset):
    # only need path & params for cal and save
    def __init__(self, root, filelist, param_fp):
        self.root = root
        self.lines = Path(filelist).read_text().strip().split('\n')
        self.param = _numpy_to_tensor(_load_cpu(param_fp))

    def _target_loader(self, index):
        target = self.param[index]
        return target

    def __getitem__(self, index):
        img_name = self.lines[index]
        target = self._target_loader(index)
        return img_name, target

    def __len__(self):
        return len(self.lines)


def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex 法线方向
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''
    pt0 = vertices[triangles[:, 0], :]  # [ntri, 3]
    pt1 = vertices[triangles[:, 1], :]  # [ntri, 3]
    pt2 = vertices[triangles[:, 2], :]  # [ntri, 3]
    tri_normal = np.cross(pt0 - pt1, pt0 - pt2)  # [ntri, 3]. normal of each triangle

    normal = np.zeros_like(vertices)  # [nver, 3]
    for i in range(triangles.shape[0]):
        normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
        normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
        normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]

    # normalize to unit length
    mag = np.sum(normal ** 2, 1)  # [nver]
    zero_ind = (mag == 0)
    mag[zero_ind] = 1
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))

    normal = normal / np.sqrt(mag[:, np.newaxis])

    return normal

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

import pickle
def load_pkl(path):
    return pickle.load(open(path, 'rb'))