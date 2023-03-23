import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from nets.layer import make_linear_layers


## generate 3D coords of 3D anchors. num is 256*3
def generate_all_anchors_3d():
    x_center = 7.5
    y_center = 7.5
    stride = 16
    step_h = 16
    step_w = 16

    d_center = 63.5
    stride_d = 64
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
        
        keypoints = torch.unsqueeze(total_references, dim = 3).expand(-1,-1,-1,42,-1) + total_coords
        pred_keypoints = (keypoints * weights).sum(2)  ## l,b,a,3 
        anchors = (torch.unsqueeze(total_references, dim = 3) * weights).sum(2)
        return pred_keypoints, anchors