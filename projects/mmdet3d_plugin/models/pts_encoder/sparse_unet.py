# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from torch import nn as nn
from functools import partial
from collections import OrderedDict
from mmdet3d.models.builder import MIDDLE_ENCODERS
from timm.models.layers import trunc_normal_
from ..utils import sparse_utils


import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=False, norm_fn=None, indice_key=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None

        if inplanes == planes:
            self.proj = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(inplanes, planes, kernel_size=1, bias=False),
                norm_fn(planes)
            )

        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        out = replace_feature(out, out.features + self.proj(residual).features)
        out = replace_feature(out, self.relu(out.features))

        return out


@MIDDLE_ENCODERS.register_module()
class SpUNetBase(nn.Module):
    def __init__(self,
                 in_channels,
                 base_channels=32,
                 channels=(32, 64, 128, 256, 256, 128, 96, 96),
                 layers=(2, 3, 4, 6, 2, 2, 2, 2),
                 mae_cfg=None,
                 fp16_enabled=False):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.mae_cfg = mae_cfg
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled
        # Spconv init all weight on its own

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        p = nn.Parameter(torch.zeros(1, in_channels))
        trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
        self.register_parameter(f'mtoken', p)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, self.base_channels, 5, padding=1, bias=False, indice_key='stem'),
            norm_fn(self.base_channels),
            nn.ReLU()
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(enc_channels, channels[s], kernel_size=2, stride=2, bias=False, indice_key=f"spconv{s + 1}"),
                norm_fn(channels[s]),
                nn.ReLU()
            ))
            self.enc.append(spconv.SparseSequential(OrderedDict([
                (f"block{i}", block(channels[s], channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                for i in range(layers[s])
            ])))
            # decode num_stages
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(channels[len(channels) - s - 2], dec_channels, kernel_size=2, bias=False, indice_key=f"spconv{s + 1}"),
                norm_fn(dec_channels),
                nn.ReLU()
            ))
            self.dec.append(spconv.SparseSequential(OrderedDict([
                (f"block{i}", block(dec_channels + enc_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                if i == 0 else (f"block{i}", block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                for i in range(layers[len(channels) - s - 1])
            ])))

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        sparse_shape = torch.add(torch.max(coors[:, 1:], dim=0).values, 96).tolist()

        if self.mae_cfg is not None:
            voxel_active = []
            for bs_idx in range(batch_size):
                voxel_active.append(sparse_utils.random_masking(1, (coors[:, 0] == bs_idx).sum().item(), 1, self.mae_cfg.mask_ratio, coors.device).squeeze())
            voxel_active = torch.cat(voxel_active, dim=0)
            voxel_features = voxel_features[voxel_active]
            coors = coors[voxel_active]

        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features,
            coors,
            sparse_shape,
            batch_size
        )
        x = self.conv_input(input_sp_tensor)

        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)

        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop(-1)
            x = replace_feature(x, torch.cat([x.features, skip.features], dim=1))
            x = self.dec[s](x)

        batch_points = torch.cat([coors[:, 0:1].float(), voxel_features[:, :3]], dim=-1)
        return batch_points, x.features
