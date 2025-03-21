import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from models.galerkin import simple_attn
from models import register
from utils import make_coord


class RBF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        lengthscale = torch.tensor(3.0)
        self.lengthscale = nn.parameter.Parameter(lengthscale)

    def forward(self, scale):
        r2 = scale / (self.lengthscale) ** 2 * -0.5
        return torch.exp(r2)


@register('AnyTSR')
class AnyTSR(nn.Module):

    def __init__(self, encoder_spec, width=256, blocks=16):
        super().__init__()
        self.width = width
        self.encoder = models.make(encoder_spec)

        self.q_proj = nn.Sequential(nn.Conv2d(2, 64, 1), nn.ReLU())
        self.k_proj = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())
        self.v_proj = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())

        self.conv00 = nn.Conv2d((64+64) * 4 + 2 + 1, self.width, 1)

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)

        self.rbf = RBF()

        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 1, 1)

    def gen_feat(self, inp, scale):
        self.inp = inp
        self.feat = self.encoder(inp, scale)
        return self.feat

    def query_v(self, coord, cell, scale):
        feat = (self.feat)
        grid = 0

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0],
                                                                                                        2, *feat.shape[
                                                                                                            -2:])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        feat_offsets = []
        feat_s = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] = coord_[:, :, :, 0] + vx * rx + eps_shift
                coord_[:, :, :, 1] = coord_[:, :, :, 1] + vy * ry + eps_shift
                coord_ = coord_.clamp(-1 + 1e-6, 1 - 1e-6)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord

                rel_coord_ = rel_coord.clone()
                rel_coord_ = self.q_proj(rel_coord_)  
                q = rel_coord_.reshape(feat.shape[0], 64, coord.shape[1] * coord.shape[2])  

                rel_coord[:, 0, :, :] = rel_coord[:, 0, :, :] * feat.shape[-2]
                rel_coord[:, 1, :, :] = rel_coord[:, 1, :, :] * feat.shape[-1]
                rel_dis = rel_coord ** 2
                rel_dis = rel_dis.sum(dim=1, keepdim=True)
                weight = self.rbf(rel_dis).reshape(feat.shape[0], -1, coord.shape[1], coord.shape[2])

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                feat_ = feat_ * weight 
                feat_s.append(feat_)

                k = self.k_proj(feat_).reshape(feat.shape[0], feat.shape[1],
                                               coord.shape[1] * coord.shape[2])  
                v = self.v_proj(feat_).reshape(feat.shape[0], feat.shape[1],
                                               coord.shape[1] * coord.shape[2])  

                attn = torch.matmul(q, k.transpose(1, 2))  
                attn = torch.softmax(attn, dim=-1)
                feat_offset = torch.matmul(attn, v).reshape(feat.shape[0], 64, coord.shape[1],
                                                            coord.shape[2])  

                feat_offsets.append(feat_offset)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        scale = scale.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])

        grid = torch.cat([*feat_offsets, *feat_s, rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2]), scale],dim=1)

        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)

        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))

        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell, scale):
        self.gen_feat(inp, scale)
        return self.query_v(coord, cell, scale)
