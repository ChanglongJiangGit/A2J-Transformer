# Copyright (c) 2020 Graz University of Technology All rights reserved.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
from utils.misc import NestedTensor
from nets.layer import MLP
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, img, mask):
        assert mask is not None
        mask = (mask.squeeze(1)<0)
        
        not_mask = ~ mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=img.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # pos_x = x_embed[:, :, :, None] / dim_t
        # pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.div(x_embed[:, :, :, None], dim_t)
        pos_y = torch.div(y_embed[:, :, :, None], dim_t)

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        pos = pos.permute(0, 3, 1, 2) # N x hidden_dim x H x W

        return pos ,mask


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, img, mask):
        x = img
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class PositionEmbeddingConvLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super(PositionEmbeddingConvLearned, self).__init__()
        self.input_size = 8
        self.num_emb_layers = [32, 64, 128, 128, num_pos_feats] # 8, 16, 32, 64, 128
        self.embed = nn.Embedding(self.input_size*self.input_size, self.num_emb_layers[0])

        self.deconv_layers = []
        for i in range(len(self.num_emb_layers)-1):
            if i == len(self.num_emb_layers)-1:
                self.deconv_layers.append(make_deconv_layers([self.num_emb_layers[i], self.num_emb_layers[i + 1]], bnrelu_final=False).to('cuda'))
            else:
                self.deconv_layers.append(make_deconv_layers([self.num_emb_layers[i], self.num_emb_layers[i+1]]).to('cuda'))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, img, mask):
        input = self.embed.weight.view(self.input_size, self.input_size, self.num_emb_layers[0]).permute(2,0,1).unsqueeze(0).cuda()
        for i in range(len(self.deconv_layers)):
            input = self.deconv_layers[i](input)
        input = input.repeat([img.shape[0],1,1,1])
        return input


class PositionEmbeddingLinearLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super(PositionEmbeddingLinearLearned, self).__init__()
        self.linear = MLP(input_dim=2, hidden_dim=[16, 32, 64, 128], output_dim=num_pos_feats, num_layers=5)

    def forward(self, img, mask):
        xx, yy = torch.meshgrid(torch.arange(img.shape[3]), torch.arange(img.shape[2]))
        pixel_locs = torch.stack([yy, xx], dim=2).to(torch.float).to(img.device) # 128 x 128 x 2
        pos = self.linear(pixel_locs.view(-1, 2)) # 128*128 x 256
        pos = pos.view(img.shape[2], img.shape[3], pos.shape[-1]).permute(2,0,1)
        pos = pos.unsqueeze(0).repeat([img.shape[0],1,1,1])
        return pos


class PositionEmbeddingSimpleCat(nn.Module):
    def __init__(self, num_pos_feats=256):
        super(PositionEmbeddingSimpleCat, self).__init__()

    def forward(self, img, mask):
        xx, yy = torch.meshgrid(torch.arange(img.shape[3]), torch.arange(img.shape[2]))
        pos = torch.stack([yy, xx], dim=2).to(torch.float).to(img.device) # 128 x 128 x 2
        pos = pos.permute(2,0,1).unsqueeze(0).repeat([img.shape[0],1,1,1])
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding in ('v4', 'convLearned'):
        position_embedding = PositionEmbeddingConvLearned(args.hidden_dim)
    elif args.position_embedding in ('v5', 'linearLearned'):
        position_embedding = PositionEmbeddingLinearLearned(args.hidden_dim)
    elif args.position_embedding in ('v6', 'simpleCat'):
        position_embedding = PositionEmbeddingSine(16, normalize=True)
        # position_embedding = PositionEmbeddingSimpleCat(N_steps)

    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
