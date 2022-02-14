#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from utils.data_utils import _numpy_to_cuda
from utils.base_params import *

from train.dataloader import _parse_param_batch
from pytorch3d.loss import chamfer_distance

_to_tensor = _numpy_to_cuda  # gpu

def vertex_normals(vertices, faces):
    """
    :param vertices: B X N X 3
    :param faces: B X F X 3
    :return:  B X N X 3
    """

    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = 'cuda:0'
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expand
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))
    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    return normals

class Loss_all(nn.Module):
    """Input and target are all 62-d param"""

    def __init__(self, opt_style='resample', resample_num=132):
        super(Loss_all, self).__init__()
        self.opt_style = opt_style
        self.param_mean = _to_tensor(param_mean)
        self.param_std = _to_tensor(param_std)

        self.u = _to_tensor(u)
        self.w_shp = _to_tensor(w_shp)
        self.w_exp = _to_tensor(w_exp)
        self.w_norm = _to_tensor(w_norm)

        # self.u_base = u_base
        # self.w_shp_base = w_shp_base
        # self.w_exp_base = w_exp_base

        self.w_shp_length = self.w_shp.shape[0] // 3  #53215
        self.keypoints = _to_tensor(keypoints)    # 68 keypoints ==> 68*3=204
        self.resample_num = resample_num

        self.tri = _to_tensor(tri)
        self.std_size = std_size

        self.keyindex = _to_tensor(keyindex)
        self.keyindex = (self.keyindex).squeeze(1)

    def reconstruct_and_parse(self, input, target):
        # reconstruct
        param = input * self.param_std + self.param_mean
        param_gt = target * self.param_std + self.param_mean

        # parse param
        p, offset, alpha_shp, alpha_exp = _parse_param_batch(param)     #input
        pg, offsetg, alpha_shpg, alpha_expg = _parse_param_batch(param_gt)     #target

        return (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg)

    def _calc_weights_resample(self, input_, target_):
        # resample index
        if self.resample_num <= 0:
            keypoints_mix = self.keypoints
        else:
            index = torch.randperm(self.w_shp_length)[:self.resample_num].reshape(-1, 1)
            keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()
            keypoints_mix = torch.cat((self.keypoints, keypoints_resample))

        w_shp_base = self.w_shp[keypoints_mix]
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]

        input = input_.clone().detach().requires_grad_(False)
        target = target_.clone().detach().requires_grad_(False)

        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input, target)

        input = self.param_std * input + self.param_mean
        target = self.param_std * target + self.param_mean

        N = input.shape[0]

        offset[:, -1] = offsetg[:, -1]

        weights = torch.zeros_like(input, dtype=torch.float)

        tmpv = (u_base +
                w_shp_base @ alpha_shp +
                w_exp_base @ alpha_exp).view(N, -1, 3).permute(0, 2, 1)

        tmpv_norm = torch.norm(tmpv, dim=2)   # for dim=2 ,get 2 norm,   tmpv_norm: 100x3
        offset_norm = sqrt(w_shp_base.shape[0] // 3)  # //: 整数除法,返回不大于结果的一个最大的整数 600//3 = 200 keypoints

        # for pose
        param_diff_pose = torch.abs(input[:, :11] - target[:, :11])
        for ind in range(11):
            if ind in [0, 4, 8]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 0]
            elif ind in [1, 5, 9]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 1]
            elif ind in [2, 6, 10]:
                weights[:, ind] = param_diff_pose[:, ind] * tmpv_norm[:, 2]
            else:    # 3, 7, 11   indicates the x,y directino's translation vector ---- with z
                weights[:, ind] = param_diff_pose[:, ind] * offset_norm
                

        ## This is the optimizest version
        # for shape_exp
        magic_number = 0.00057339936  # scale
        param_diff_shape_exp = torch.abs(input[:, 12:] - target[:, 12:])
        w = torch.cat((w_shp_base, w_exp_base), dim=1)
        w_norm = torch.norm(w, dim=0)   
        # print('here')                     100x50              1x50 
        weights[:, 12:] = magic_number * param_diff_shape_exp * w_norm
        eps = 1e-6
        weights[:, :12] += eps
        weights[:, 12:] += eps

        # normalize the weights
        maxes, _ = weights.max(dim=1)
        maxes = maxes.view(-1, 1)
        weights /= maxes

        # zero the z
        weights[:, 11] = 0

        return weights

    def cal_vdc(self, input_, target_):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input_, target_)

        # resample index ---> 200 points
        index = torch.randperm(self.w_shp_length)[:self.resample_num].reshape(-1, 1)
        keypoints_resample = torch.cat((3 * index, 3 * index + 1, 3 * index + 2), dim=1).view(-1).cuda()
        keypoints_mix = torch.cat((self.keypoints, keypoints_resample))
        w_shp_base = self.w_shp[keypoints_mix]
        u_base = self.u[keypoints_mix]
        w_exp_base = self.w_exp[keypoints_mix]

        N = input_.shape[0]
        # offset[:, -1] = offsetg[:, -1]
        gt_vertex = pg @ (u_base + w_shp_base @ alpha_shpg + w_exp_base @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset

        loss, _ = chamfer_distance(gt_vertex.permute(0, 2, 1), vertex.permute(0, 2, 1))

        return loss * 1e-3

    def cal_nwlmdc(self, input_, target_, normals_weight):
        (p, offset, alpha_shp, alpha_exp), (pg, offsetg, alpha_shpg, alpha_expg) \
            = self.reconstruct_and_parse(input_, target_)

        N = input_.shape[0]
        vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp) \
            .view(N, -1, 3).permute(0, 2, 1) + offset # B X 3 X N
        vertex[:,1,:] = self.std_size + 1 - vertex[:,1,:]
        vertex_gt = pg @ (self.u + self.w_shp @ alpha_shpg + self.w_exp @ alpha_expg) \
            .view(N, -1, 3).permute(0, 2, 1) + offsetg  # B X 3 X N
        vertex_gt[:,1,:] = self.std_size + 1 - vertex_gt[:,1,:]

        indexes = _to_tensor(keyindex)
        indexes = indexes.squeeze(1)
        vertex_68 = torch.index_select(vertex.permute(0,2,1), 1, indexes.long()) # B X 68 X 3
        vertex_gt = torch.index_select(vertex_gt.permute(0,2,1), 1, indexes.long()) # B X 68 X 3

        loss = torch.mean(
            torch.bmm(normals_weight, (vertex_gt - vertex_68) ** 2)
        )
        return loss * 1e-3

        
    def forward(self, input, target, normals):
        weights = self._calc_weights_resample(input, target)
        loss_wpdc = weights * (input - target) ** 2

        loss_vdc = self.cal_vdc(input, target)

        loss_nwlmdc = self.cal_nwlmdc(input, target, normals)

        # print(loss_wpdc.mean(), loss_vdc, 3 * loss_nwlmdc)
        #
        # print(loss_wpdc.mean(), loss_nwlmdc)
        total_loss = loss_wpdc.mean() + 3 * loss_vdc + 3 * loss_nwlmdc

        return total_loss
