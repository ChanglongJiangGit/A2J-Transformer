# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from config import cfg
from anchor import *

from nets.layer  import MLP
from nets.position_encoding import build_position_encoding

# use dab-deformable-detr
from dab_deformable_detr.deformable_transformer import build_deforamble_transformer
from dab_deformable_detr.backbone import build_backbone
from utils.miscdetr import NestedTensor
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class A2J_model(nn.Module):
    def __init__(self, backbone_net, position_embedding, transformer, num_classes, num_queries, num_feature_levels,
                 with_box_refine=True, two_stage=False, use_dab=True, 
                 num_patterns=0, anchor_refpoints_xy=True, fix_anchor=False, is_3D=True, use_lvl_weights=False):
        """ Initializes the model.
        Parameters:
            backbone_net: torch module of the backbone to be used. 
            transformer: torch module of the transformer architecture. 
            num_classes: number of object classes, given 42
            num_queries: number of object queries, given 256*3
            num_feature_levels: number of feature layers used form backbone, default = 4
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR, default is False
            use_dab: using dab-deformable-detr
            num_patterns: number of pattern embeddings
            anchor_refpoints_xy: init the x,y of anchor boxes to A2J anchors and freeze them. 
            fix_anchor: if fix the reference points as the initial anchor points to stop renew them
            is_3D: if the model regresses 3D coords of the keypoints
        """
        super(A2J_model, self).__init__()
        self.backbone = backbone_net
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.num_classes = num_classes  # default = 42
        self.num_queries = num_queries  # default = 768
        self.num_feature_levels = num_feature_levels # default = 4
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.anchor_refpoints_xy = anchor_refpoints_xy
        self.fix_anchor = fix_anchor
        self.is_3D = is_3D 
        self.use_lvl_weights = use_lvl_weights
        self.kernel_size = cfg.kernel_size

        hidden_dim = transformer.d_model  # =cfg.hidden_dim, default = 256
        self.bbox_embed_anchor = MLP(hidden_dim, hidden_dim, 2, 3)
        if self.is_3D:
            self.bbox_embed_keypoints = MLP(hidden_dim, hidden_dim, self.num_classes *3, 3)   ## 3D coord
        else:
            assert self.is_3D is False
            self.bbox_embed_keypoints = MLP(hidden_dim, hidden_dim, self.num_classes *2, 3)   ## only xy-coord
        self.anchor_weights = MLP(hidden_dim, hidden_dim, self.num_classes *1, 3)

        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 3)

                if anchor_refpoints_xy: 
                    self.anchors = generate_all_anchors_3d()
                    self.anchors = torch.from_numpy(self.anchors).cuda().float() 
                    self.refpoint_embed.weight.data = self.anchors 
                    self.refpoint_embed.weight.data.requires_grad = False


        if num_feature_levels > 1: 
            num_backbone_outs = len(self.backbone.strides) #8,16,32
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_] # [512, 1024, 2048]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs): 
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.bbox_embed_anchor = _get_clones(self.bbox_embed_anchor, num_pred)
            self.transformer.decoder.bbox_embed = self.bbox_embed_anchor
        else:
            nn.init.constant_(self.bbox_embed_anchor.layers[-1].bias.data[2:], -2.0)
            self.bbox_embed_anchor = nn.ModuleList([self.bbox_embed_anchor for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            for box_embed in self.bbox_embed_anchor:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # for final output
        nn.init.constant_(self.bbox_embed_anchor[0].layers[-1].bias.data[2:], -2.0)
        self.bbox_embed_keypoints = _get_clones(self.bbox_embed_keypoints, num_pred)
        nn.init.constant_(self.bbox_embed_keypoints[0].layers[-1].bias.data[2:], -2.0)
        self.anchor_weights = _get_clones(self.anchor_weights, num_pred)
        nn.init.constant_(self.anchor_weights[0].layers[-1].bias.data[2:], -2.0)

        self.generate_keypoints_coord_new = generate_keypoints_coord_new(self.num_classes, is_3D = self.is_3D)
        self.a2jloss_new = a2jloss_new(is_3D = self.is_3D, use_lvl_weights= self.use_lvl_weights)


    
    # def forward(self, x): 
    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']
        input_mask = inputs['mask']
        batch_size = input_img.shape[0]
        if cfg.dataset == 'nyu' or cfg.dataset == 'hands2017':
            n, c, h, w = input_img.size()  # x: [B, 1, H ,W]
            input_img = input_img[:,0:1,:,:]  # depth
            input_img = input_img.expand(n,3,h,w) ## convert depth to rgb domain
        
        samples = NestedTensor(input_img,input_mask.squeeze(1))

        ## get pyramid features
        features, pos = self.backbone(samples)
        srcs = []
        masks = []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)   
                masks.append(mask)  
                pos.append(pos_l)   

        tgt_embed = self.tgt_embed.weight
        refanchor = self.refpoint_embed.weight
        
        ## Convert refanchor to [0,1] range
        refanchor = refanchor / cfg.output_hm_shape_all
        query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        
        ## Transformer module. Enhance features.
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_coords = []
        outputs_weights = []
        references = []

        ## Predict offset and weights for each layer. 
        ## Total 6 layers, which is the same as enc/dec layers.
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference.squeeze(0).expand(batch_size,-1 ,-1) 
            else:
                reference = inter_references[lvl - 1]
            outputs_weight = self.anchor_weights[lvl](hs[lvl])  
            tmp = self.bbox_embed_keypoints[lvl](hs[lvl])  
            assert reference.shape[-1] == 3 
            
            ## convert result to [0,256] range, same to size of output img 
            reference = reference  * cfg.output_hm_shape_all
            outputs_coord = tmp * cfg.output_hm_shape_all 
            
            outputs_coords.append(outputs_coord)
            outputs_weights.append(outputs_weight)
            references.append(reference)
            
        total_outputs_coord = torch.stack(outputs_coords) ## A2J-offsets
        total_outputs_weights = torch.stack(outputs_weights) ## A2J-weights
        total_references = torch.stack(references) ## A2J-anchors

        ## generate final coords
        keypoints_coord, anchor = self.generate_keypoints_coord_new(total_outputs_coord, total_outputs_weights, total_references)
        
        ## get loss
        anchor_loss, regression_loss = self.a2jloss_new(keypoints_coord, anchor, targets['joint_coord'], meta_info['joint_valid'])
        
        if mode == 'train':
            loss = {}
            loss['Cls_loss'] = anchor_loss
            loss['Reg_loss'] = regression_loss
            loss['A2Jloss'] = 1*anchor_loss + regression_loss* cfg.RegLossFactor 
            loss['total_loss'] =loss['A2Jloss']
            return loss

        elif mode == 'test':
            ## use the result of last layer as the final result
            pred_keypoints = keypoints_coord[-1]
            out = {}
            out['joint_coord'] =pred_keypoints
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            return out



def get_model(mode, joint_num):
    backbone_net = build_backbone(cfg)
    transformer = build_deforamble_transformer(cfg)
    position_embedding = build_position_encoding(cfg)
    model = A2J_model(backbone_net, 
                      position_embedding,
                      transformer,
                      num_classes = joint_num * 2, 
                      num_queries = cfg.num_queries, 
                      num_feature_levels = cfg.num_feature_levels, 
                      two_stage=cfg.two_stage,
                      use_dab=True,
                      num_patterns=cfg.num_patterns,
                      anchor_refpoints_xy=cfg.anchor_refpoints_xy,
                      fix_anchor = cfg.fix_anchor, 
                      is_3D=cfg.is_3D,
                      use_lvl_weights=cfg.use_lvl_weights)
    
    ## Statistical Model Size
    print('BackboneNet No. of Params = %d M'%(sum(p.numel() for p in backbone_net.parameters() if p.requires_grad)/1e6))
    print('Transformer No. of Params = %d M'%(sum(p.numel() for p in transformer.parameters() if p.requires_grad)/1e6))
    print('Total No. of Params = %d M' % (sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))
    return model
