# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
from mmdet3d.models.builder import VOXEL_ENCODERS
import torch_scatter
import numpy as np


@VOXEL_ENCODERS.register_module()
class CustomDynamicSimpleVFE(nn.Module):
    def __init__(
        self, voxel_size=(0.2, 0.2, 4), point_cloud_range=(0, -40, -3, 70.4, 40, 1)
    ):
        super(CustomDynamicSimpleVFE, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.fp16_enabled = False

    @torch.no_grad()
    @force_fp32(out_fp16=True)
    def forward(self, pts):
        voxel_size = torch.tensor(self.voxel_size).to(pts[0])
        pc_range = torch.tensor(self.point_cloud_range).to(pts[0])
        voxel_shape = torch.round((pc_range[3:] - pc_range[:3]) / voxel_size).long()

        points, coords = [], []
        for bs_idx, i_pts in enumerate(pts):
            p_coords = ((i_pts[:, :3] - pc_range[:3]) / voxel_size).long()
            points.append(i_pts)
            coords.append(
                F.pad(p_coords[:, [2, 1, 0]], (1, 0), mode="constant", value=bs_idx)
            )  # (N, 4), [bs_idx, z, y, x]
        points, coords = torch.cat(points, dim=0), torch.cat(coords, dim=0)  # (N, 4)

        kept = torch.all(
            (coords[..., [3, 2, 1]] >= torch.zeros_like(voxel_shape))
            & (coords[..., [3, 2, 1]] < voxel_shape),
            dim=-1,
        )
        coords, points = coords[kept], points[kept]

        _, inverse = coords.unique(sorted=True, return_inverse=True, dim=0)
        _, indices = torch_scatter.scatter_max(
            torch.arange(len(coords)).to(coords.device), inverse, dim=0
        )
        coords, points = coords[indices], points[indices]

        return points, coords


@VOXEL_ENCODERS.register_module()
class GridSample(nn.Module):
    def __init__(self, voxel_size=(0.05, 0.05, 0.05), flip_coords=False):
        super(GridSample, self).__init__()
        self.voxel_size = voxel_size
        self.flip_coords = flip_coords
        self.fp16_enabled = False

    def ravel_hash_vec(self, coords):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        coords_max = coords.max(dim=0, keepdim=False)[0] + 1

        keys = torch.zeros_like(coords[:, 0])
        for j in range(coords.shape[1] - 1):
            keys += coords[:, j]
            keys *= coords_max[j + 1]
        keys += coords[:, -1]
        return keys

    @torch.no_grad()
    @force_fp32(out_fp16=True)
    def forward(self, pts):
        voxel_size = torch.tensor(self.voxel_size).to(pts[0])

        points, coords = [], []
        for bs_idx, i_pts in enumerate(pts):
            p_coords = (i_pts[:, :3] / voxel_size).long()
            p_coords -= p_coords.min(dim=0, keepdim=True)[0]
            points.append(i_pts)
            coords.append(
                F.pad(
                    p_coords
                    if not self.flip_coords
                    else torch.flip(p_coords, dims=[-1]),
                    (1, 0),
                    mode="constant",
                    value=bs_idx,
                )
            )  # (N, 4), [bs_idx, x, y, z] or [bs_idx, z, y, x]
        points, coords = torch.cat(points, dim=0), torch.cat(coords, dim=0)  # (N, 4)

        key = self.ravel_hash_vec(coords)
        key_sort, idx_sort = torch.sort(key, dim=0)
        _, inverse, count = key_sort.unique_consecutive(
            return_inverse=True, return_counts=True, dim=0
        )

        idx_select = (
            torch.cumsum(torch.cat([count.new_zeros(1), count[:-1]], dim=0), dim=0)
            + torch.randint(
                0, count.max() if self.training else 1, count.shape, device=count.device
            )
            % count
        )
        idx_unique = idx_sort[idx_select]

        points, coords = points[idx_unique], coords[idx_unique]
        return points, coords
