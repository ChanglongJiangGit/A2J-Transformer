import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from nets.layer import make_linear_layers


## generate 3D coords of 3D anchors. num is 256*3
def generate_all_anchors_3d():
    if cfg.dataset == "InterHand2.6M":
        x_center = 7.5
        y_center = 7.5
        stride = 16
        step_h = 16
        step_w = 16
        
        d_center = 63.5
        stride_d = 64
        step_d = 3
        
    elif cfg.dataset == 'nyu':
        x_center = 7.5
        y_center = 7.5
        stride = 16
        step_h = 16
        step_w = 16
        
        d_center = -75
        stride_d = 75
        step_d = 3
        
    elif cfg.dataset == 'hands2017':
        x_center = 7.5
        y_center = 7.5
        stride = 16
        step_h = 11
        step_w = 11
        
        d_center = -75
        stride_d = 75
        step_d = 3

    anchors_h = np.arange(0,step_h) * stride + x_center
    anchors_w = np.arange(0,step_w) * stride + y_center
    anchors_d = np.arange(0,step_d) * stride_d + d_center
    anchors_x, anchors_y, anchors_z = np.meshgrid(anchors_h, anchors_w, anchors_d)
    all_anchors = np.vstack((anchors_x.ravel(), anchors_y.ravel(), anchors_z.ravel())).transpose()  #256*3
    return all_anchors


class generate_keypoints_coord_new(nn.Module): 
    def __init__(self, num_joints, is_3D=True):
        super(generate_keypoints_coord_new, self).__init__()
        self.is_3D = is_3D
        self.num_joints = num_joints

    def forward(self, total_coords, total_weights, total_references):
        lvl_num, batch_size, a, _ = total_coords.shape
        total_coords = total_coords.reshape(lvl_num, batch_size, a, self.num_joints, -1)  ## l,b,a,j,3 
        
        weights_softmax = F.softmax(total_weights, dim=2)
        weights = torch.unsqueeze(weights_softmax, dim=4).expand(-1,-1,-1,-1, 3) ## l,b,a,j,3 
        
        keypoints = torch.unsqueeze(total_references, dim = 3).expand(-1,-1,-1,cfg.keypoint_num,-1) + total_coords
        pred_keypoints = (keypoints * weights).sum(2)  ## l,b,a,3 
        anchors = (torch.unsqueeze(total_references, dim = 3) * weights).sum(2)
        return pred_keypoints, anchors
    
class a2jloss_new(nn.Module):
    def __init__(self, spatialFactor=0.5, is_3D=True, use_lvl_weights=False):
        super(a2jloss_new, self).__init__()
        self.is_3D = is_3D
        self.spatialFactor = spatialFactor
        self.use_lvl_weights = use_lvl_weights
        if self.use_lvl_weights is True:
            self.lvl_weights = cfg.lvl_weights

    def forward(self, total_keypoints_coord, total_anchor, annotations, joint_valid):
        lvl_num, batch_size, j, _ = total_keypoints_coord.shape
        # total_keypoints_coord  ## l,b,j,3 
        # total_anchor           ## l,b,j,3   
        # annotations            ##   b,j,3 
        # joint_valid            ##   b,j   
        
        total_annotations = torch.unsqueeze(annotations,0).expand(lvl_num,-1,-1,-1)  ## l,b,j,3 
        total_joint_valid = joint_valid[None,:,:,None].expand(lvl_num,-1,-1, 3)  ## l,b,j,3 
        
        ## Count the number of valid and set zero items to one
        per_batch_valid_num = torch.sum(joint_valid, dim = 1)
        per_batch_valid_num = torch.where(
                    torch.le(per_batch_valid_num, 0.5),
                    per_batch_valid_num+1,
                    per_batch_valid_num
        )
        per_batch_valid_num = torch.unsqueeze(per_batch_valid_num,0).expand(lvl_num,-1) * 3  ## l,b ,*3 indicates how many 3d coordinates are available in l and b

        anchor_diff = torch.abs(total_anchor - total_annotations)
        anchor_loss = torch.where(
                    torch.le(anchor_diff, 1),
                    0.5 * 1 * torch.pow(anchor_diff, 2),
                    anchor_diff - 0.5 / 1
                    )
        anchor_loss_valid = (anchor_loss * total_joint_valid).reshape(lvl_num, batch_size, -1)  ## l,b,j*3 
        anchor_loss_mean_batch = torch.mean(torch.sum(anchor_loss_valid, 2)/ per_batch_valid_num, 1)  ## l,b  

        regression_diff_xy = torch.abs(total_keypoints_coord[:,:,:,0:2] - total_annotations[:,:,:,0:2])
        regression_loss_xy = torch.where(
                    torch.le(regression_diff_xy, 1),
                    0.5 * 1 * torch.pow(regression_diff_xy, 2),
                    regression_diff_xy - 0.5 / 1
                    )
        regression_diff_z = torch.abs(total_keypoints_coord[:,:,:,2] - total_annotations[:,:,:,2])
        regression_loss_z = torch.where(
                    torch.le(regression_diff_z, 3),
                    0.5 * (1/3) * torch.pow(regression_diff_z, 2),
                    regression_diff_z - 0.5 / (1/3)
                    )
        regression_loss = torch.cat((regression_loss_xy * self.spatialFactor, regression_loss_z.unsqueeze(3)), dim = 3)
        regression_loss_valid = (regression_loss * total_joint_valid).reshape(lvl_num, batch_size, -1)  ## l,b,j*3
        regression_loss_mean_batch = torch.mean(torch.sum(regression_loss_valid, 2)/ per_batch_valid_num, 1)  ## l,b 
        
        if self.use_lvl_weights is True:
            anchor_loss_output = torch.mean(anchor_loss_mean_batch * torch.tensor(self.lvl_weights).to(anchor_loss_mean_batch.device))
            regression_loss_output = torch.mean(regression_loss_mean_batch * torch.tensor(self.lvl_weights).to(regression_loss_mean_batch.device))
        else:
            anchor_loss_output = torch.mean(anchor_loss_mean_batch)
            regression_loss_output = torch.mean(regression_loss_mean_batch)
        
        return anchor_loss_output, regression_loss_output