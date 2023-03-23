# Copyright (c) 2020 Graz University of Technology All rights reserved.

import torch
import torch.distributed as dist
from torch import Tensor
from typing import Optional, List
import numpy as np
from config import cfg
from scipy.optimize import linear_sum_assignment

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

def hungarian_match_2djoints(joint_loc_gt_np, joint_loc_pred_np, joint_valid_np, mask, joint_class_pred):
    mid_dist_th = 2
    # get associations based on hungarian algo.
    gt_inds_batch = []
    row_inds_batch = []
    for i in range(joint_loc_pred_np.shape[0]):
        # for invalid joints, make sure there locations are somewhere far off, so that any asso. with them will be invalid
        joint_loc_gt_np[i][np.logical_not(joint_valid_np[i])] = np.array(
            [cfg.input_img_shape[0], cfg.input_img_shape[0]]) * 2

        # create the cost matrix. any cost > mid_dist_th is clipped. This helps in scenarios where cost matrix is tall
        dist = np.linalg.norm(np.expand_dims(joint_loc_pred_np[i], 1) - np.expand_dims(joint_loc_gt_np[i], 0),
                              axis=2)  # max_num_peaks x 21
        dist1 = dist[mask[i]]  # remove the invalid guys
        dist1[dist1 > mid_dist_th] = mid_dist_th

        # invoke the hungry hungarian
        indices = linear_sum_assignment(dist1)
        row_ind = indices[0]
        asso_ind = indices[1]

        # 0 - invalid class, rest all classes follow the same order as in the joint_gt
        gt_inds = asso_ind + 1

        # if the any associations have a distance > mid_dist_th, then its not right
        for ii in range(row_ind.shape[0]):
            if dist1[row_ind[ii], asso_ind[ii]] >= mid_dist_th:
                gt_inds[ii] = 0

        # when cost matrix is tall, assign the false postives to 0 index
        if row_ind.shape[0] < np.sum(mask[i]):
            false_pos_row_inds = np.setdiff1d(np.arange(0, np.sum(mask[i])), row_ind)
            assert false_pos_row_inds.shape[0] == (np.sum(mask[i]) - row_ind.shape[0])
            row_ind = np.concatenate([row_ind, false_pos_row_inds], axis=0)
            gt_inds = np.concatenate([gt_inds, np.zeros((false_pos_row_inds.shape[0],))], axis=0)

        gt_inds_batch.append(gt_inds)
        row_inds_batch.append(row_ind)
    asso_inds_batch_list = gt_inds_batch
    row_inds_batch_list = row_inds_batch
    gt_inds_batch = np.concatenate(gt_inds_batch, axis=0).astype(np.int)

    gt_inds_batch = np.tile(np.expand_dims(gt_inds_batch, 0), [joint_class_pred.shape[0], 1])  # 6 x M
    gt_inds_batch = np.reshape(gt_inds_batch, [-1])  # 6*M

    return gt_inds_batch, row_inds_batch_list, asso_inds_batch_list


def nearest_match_2djoints(joint_loc_gt_np, joint_loc_pred_np, joint_valid_np, mask, joint_class_pred, joints_3d_gt_np,
                           peak_joints_map_batch_np, target_obj_kps_heatmap_np, obj_kps_coord_gt_np, obj_kps_3d_gt_np):
    mid_dist_th = 3
    gt_inds_batch = []
    row_inds_batch = []
    bs = joint_loc_gt_np.shape[0]

    for i in range(bs):
        # for invalid joints, make sure there locations are somewhere far off, so that any asso. with them will be invalid
        joint_loc_gt_np[i][np.logical_not(joint_valid_np[i])] = np.array(
            [cfg.input_img_shape[0], cfg.input_img_shape[0]]) * 2

        gt_inds = np.zeros((np.sum(mask[i])))
        for j in np.arange(0, cfg.max_num_peaks):
            if mask[i,j] == 0:
                continue

            curr_joint_loc_pred = joint_loc_pred_np[i, j]
            if cfg.has_object:
                if peak_joints_map_batch_np[i,j] == cfg.obj_cls_index:

                    if target_obj_kps_heatmap_np[i, int(curr_joint_loc_pred[1]), int(curr_joint_loc_pred[0])] > 100:
                        gt_inds[j] = cfg.obj_cls_index
                    else:
                        gt_inds[j] = 0
                    continue


            dist = np.linalg.norm(curr_joint_loc_pred - joint_loc_gt_np[i], axis=-1)
            closest_pts_mask = dist <= mid_dist_th
            if not np.any(closest_pts_mask):
                continue
            foreground_pt_ind = np.argmin(joints_3d_gt_np[i,:,2][closest_pts_mask])
            gt_inds[j] = np.where(closest_pts_mask)[0][foreground_pt_ind]+1

        gt_inds_batch.append(gt_inds)
        row_inds_batch.append(np.arange(0, np.sum(mask[i])))

    asso_inds_batch_list = gt_inds_batch
    row_inds_batch_list = row_inds_batch
    gt_inds_batch = np.concatenate(gt_inds_batch, axis=0).astype(np.int)

    gt_inds_batch = np.tile(np.expand_dims(gt_inds_batch, 0), [joint_class_pred.shape[0], 1])  # 6 x M
    gt_inds_batch = np.reshape(gt_inds_batch, [-1])  # 6*M

    return gt_inds_batch, row_inds_batch_list, asso_inds_batch_list

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.float32)

def get_tgt_mask():
    if cfg.predict_type == 'angles':
        tgt_mask = torch.zeros((cfg.num_queries, cfg.num_queries), dtype=torch.bool)
        # global rot
        tgt_mask[0, :] = True
        tgt_mask[0, 0] = False
        tgt_mask[cfg.num_joint_queries_per_hand, :] = True
        tgt_mask[cfg.num_joint_queries_per_hand, cfg.num_joint_queries_per_hand] = False

        # fingers
        for i in range(5):
            # right hand
            s = 3 * i + 1
            e = 3 * i + 4
            tgt_mask[s:e, :] = True
            tgt_mask[s:e, s:e] = False
            # left hand
            s = s + cfg.num_joint_queries_per_hand
            e = e + cfg.num_joint_queries_per_hand
            tgt_mask[s:e, :] = True
            tgt_mask[s:e, s:e] = False

        # trans and shape
        tgt_mask[cfg.shape_indx, :] = True
        tgt_mask[cfg.shape_indx, cfg.shape_indx] = False


        if cfg.has_object:
            # make hand queries depend on object
            tgt_mask[:2 * cfg.num_joint_queries_per_hand, cfg.obj_rot_indx] = False
            tgt_mask[:2 * cfg.num_joint_queries_per_hand, cfg.obj_trans_indx] = False
            tgt_mask[cfg.shape_indx, cfg.obj_rot_indx] = False
            tgt_mask[cfg.shape_indx, cfg.obj_trans_indx] = False

    elif cfg.predict_type == 'vectors':
        tgt_mask = torch.zeros((cfg.num_queries, cfg.num_queries), dtype=torch.bool)
        # fingers
        for i in range(5):
            # right hand
            s = 4 * i + 0
            e = 4 * i + 4
            tgt_mask[s:e, :] = True
            tgt_mask[s:e, s:e] = False
            # left hand
            s = s + cfg.num_joint_queries_per_hand
            e = e + cfg.num_joint_queries_per_hand
            tgt_mask[s:e, :] = True
            tgt_mask[s:e, s:e] = False
        # trans and shape
        tgt_mask[cfg.shape_indx, :] = True
        tgt_mask[cfg.shape_indx, cfg.shape_indx] = False


    else:
        raise NotImplementedError


    return tgt_mask

def get_src_memory_mask(peak_joints_map_batch):
    src_mask_list = []
    memory_mask_list = []
    for i in range(peak_joints_map_batch.shape[0]):
        mask = torch.zeros((cfg.max_num_peaks, cfg.max_num_peaks), dtype=torch.bool)
        joint_locs_mask = torch.logical_and(peak_joints_map_batch[i] != cfg.obj_cls_index, peak_joints_map_batch[i] != 0)
        mask[joint_locs_mask] = True
        mask[joint_locs_mask, joint_locs_mask] = False

        src_mask_list.append(mask)

        memory_mask = torch.zeros((cfg.num_queries, cfg.max_num_peaks), dtype=torch.bool)
        if np.sum((peak_joints_map_batch[i]==cfg.obj_cls_index).cpu().numpy()) > 0:
            memory_mask[cfg.obj_rot_indx] = peak_joints_map_batch[i]!=cfg.obj_cls_index
        memory_mask_list.append(memory_mask)

    src_mask = torch.stack(src_mask_list,dim=0).unsqueeze(1).repeat(1,cfg.nheads,1,1).view(-1, cfg.max_num_peaks, cfg.max_num_peaks)
    memory_mask_list = torch.stack(memory_mask_list, dim=0).unsqueeze(1).repeat(1,cfg.nheads,1,1).view(-1, cfg.num_queries, cfg.max_num_peaks)

    return src_mask, memory_mask_list

def get_root_rel_from_parent_rel_depths(dep):
    joint_recon_order = [3, 2, 1, 0,
                         7, 6, 5, 4,
                         11, 10, 9, 8,
                         15, 14, 13, 12,
                         19, 18, 17, 16]

    dep_root = []
    for j in range(5):
        for i in range(4):
            if i == 0:
                dep_root.append(dep[joint_recon_order[j*4+i]])
            else:
                new_dep = dep[joint_recon_order[j*4+i]] + dep_root[-1]
                dep_root.append(new_dep)

    dep_root_reorder = np.array([dep_root[i] for i in joint_recon_order]+[0])
    return dep_root_reorder

def my_print(string, f=None):
    print(string)
    if f is not None:
        f.write(string+'\n')

def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def get_valid_front_face_from_bary(bary, verts, faces_in):
    # given barycentric coordinates for each triangle in the mesh, find which frontmost triangle the point belongs to
    '''

    :param bary: N x M x F x 3
    :param verts: N x V x 3
    :param faces_in: F x 3
    :return:
    '''
    faces = np.tile(np.expand_dims(faces_in, 0), (verts.shape[0], 1, 1))  # N x F x 3
    for i in range(verts.shape[0]):
        faces[i] += verts.shape[1]*i
    tri_verts = verts.reshape(-1, 3)[faces.reshape(-1)].reshape(verts.shape[0], faces_in.shape[0], 3,
                                                                3)  # N x F x 3 x 3
    mean_tri_dep = torch.mean(tri_verts[:,:,:,2], dim=2) # N x F
    mean_tri_dep = mean_tri_dep.unsqueeze(1).repeat(1, bary.shape[1], 1) # N x M x F

    inside_pts = torch.logical_and(bary[:, :, :, 0] >= 0, bary[:, :, :, 0] <= 1)
    inside_pts = torch.logical_and(inside_pts, bary[:, :, :, 1] >= 0)
    inside_pts = torch.logical_and(inside_pts, bary[:, :, :, 1] <= 1)
    inside_pts = torch.logical_and(inside_pts, bary[:, :, :, 2] >= 0)
    inside_pts = torch.logical_and(inside_pts, bary[:, :, :, 2] <= 1) # N x M x F

    mean_tri_dep[torch.logical_not(inside_pts)] = float('inf') # N x M x F
    min_val, hit_tri_ind = torch.min(mean_tri_dep, dim=2) # N x M
    valid_hit_tri_ind = min_val != float('inf') # N x M

    hit_tri_verts = torch.gather(tri_verts, 1, hit_tri_ind[:,:,None,None].repeat(1,1,3,3)) # N x M x 3 x 3
    hit_tri_center = torch.mean(hit_tri_verts, dim=2) # N x M x 3

    return hit_tri_center, hit_tri_ind, valid_hit_tri_ind

def get_mesh_contacts(contact_pos_pred, vert, faces_in, cam_param):
    '''

    :param contact_pos_pred: N x M x 2
    :param vert: N x V x 3
    :param faces_in: F x 3
    :param cam_param: N x 3
    :return:
    '''

    contact_pos_plane = torch.cat([contact_pos_pred,
                                   torch.zeros((contact_pos_pred.shape[0], contact_pos_pred.shape[1], 1)).to(contact_pos_pred.device)], dim=2) # N x M x 3
    vert_plane = vert[:,:,:2]*cam_param.unsqueeze(1)[:,:,:1] + cam_param.unsqueeze(1)[:,:,1:] # N x V x 2
    vert_plane = torch.cat([vert_plane, torch.zeros((vert_plane.shape[0], vert_plane.shape[1], 1)).to(vert_plane.device)], dim=2) # N x V x 3

    bary_points = get_barycentric_points_from_contained_points(contact_pos_plane, vert_plane, faces_in) # N x M x F x 3
    hit_tri_center, hit_tri_ind, valid_hit_tri_ind  = get_valid_front_face_from_bary(bary_points, vert, faces_in)

    return hit_tri_center, hit_tri_ind, valid_hit_tri_ind




def get_barycentric_points_from_contained_points(points, verts, faces_in):
    # give a set of points on the surface of the mesh, get their barycentric coordinates for each triangle in face
    # http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    '''

    :param points: N x M x 3
    :param verts: N x V x 3
    :param faces_in: F x 3
    :return:
    '''
    faces = np.tile(np.expand_dims(faces_in,0), (verts.shape[0],1,1)) # N x F x 3
    for i in range(verts.shape[0]):
        faces[i] += verts.shape[1]*i
    tri_verts = verts.reshape(-1,3)[faces.reshape(-1)].reshape(verts.shape[0], faces_in.shape[0], 3, 3) # N x F x 3 x 3

    a = tri_verts[:, :, 0, :] # N x F x 3
    b = tri_verts[:, :, 1, :] # N x F x 3
    c = tri_verts[:, :, 2, :] # N x F x 3
    v0 = (b - a).unsqueeze(1).repeat(1,points.shape[1],1,1) # N x M x F x 3
    v1 = (c - a).unsqueeze(1).repeat(1,points.shape[1],1,1) # N x M x F x 3
    v2 = points.unsqueeze(2) - a.unsqueeze(1) # N x M x F x 3
    d00 = torch.sum(v0 * v0, dim=3) # N x M x F
    d01 = torch.sum(v0 * v1, dim=3)
    d11 = torch.sum(v1 * v1, dim=3)
    d20 = torch.sum(v2 * v0, dim=3)
    d21 = torch.sum(v2 * v1, dim=3)
    denom = d00 * d11 - d01 * d01 # N x M x F
    v = (d11 * d20 - d01 * d21) / denom # N x M x F
    w = (d00 * d21 - d01 * d20) / denom # N x M x F
    u = 1 - v - w # N x M x F

    bary = torch.stack([u,v,w], dim=3) # N x M x F x 3

    return bary
