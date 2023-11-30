import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmcv.cnn import xavier_init
from .render_utils import models
from .render_utils.rays import RayBundle
from ..utils import Uni3DViewTrans, sparse_utils
import pickle
from .. import utils


@HEADS.register_module()
class RenderHead(BaseModule):
    def __init__(
        self,
        in_channels,
        unified_voxel_size,
        unified_voxel_shape,
        pc_range,
        render_conv_cfg,
        view_cfg,
        ray_sampler_cfg,
        render_ssl_cfg,
        **kwargs
    ):
        super().__init__()
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = True
        self.in_channels = in_channels
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.unified_voxel_shape = np.array(unified_voxel_shape, dtype=np.int32)
        self.unified_voxel_size = np.array(unified_voxel_size, dtype=np.float32)

        if view_cfg is not None:
            vtrans_type = view_cfg.pop("type", "Uni3DViewTrans")
            self.view_trans = getattr(utils, vtrans_type)(
                pc_range=self.pc_range,
                voxel_size=self.unified_voxel_size,
                voxel_shape=self.unified_voxel_shape,
                **view_cfg
            )  # max pooling, deformable detr, bilinear

        self.render_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                render_conv_cfg["out_channels"],
                kernel_size=render_conv_cfg["kernel_size"],
                padding=render_conv_cfg["padding"],
                stride=1,
            ),
            nn.BatchNorm3d(render_conv_cfg["out_channels"]),
            nn.ReLU(inplace=True),
        )

        model_type = render_ssl_cfg.pop("type")
        self.render_model = getattr(models, model_type)(
            pc_range=self.pc_range,
            voxel_size=self.unified_voxel_size,
            voxel_shape=self.unified_voxel_shape,
            **render_ssl_cfg
        )

        self.ray_sampler_cfg = ray_sampler_cfg
        self.part = 8192  # avoid out of GPU memory

    def loss(self, preds_dict, targets):
        batch_size = len(targets)
        loss_dict = {}
        for bs_idx in range(batch_size):
            i_loss_dict = self.render_model.loss(preds_dict[bs_idx], targets[bs_idx])
            for k, v in i_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = []
                loss_dict[k].append(v)
        for k, v in loss_dict.items():
            loss_dict[k] = torch.stack(v, dim=0).mean()
        return loss_dict

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, pts_feats, img_feats, rays, img_metas, img_depth):
        """
        Args:
            Currently only support single-frame, no 3D data augmentation, no 2D data augmentation
            ray_o: [(N*C*K, 3), ...]
            ray_d: [(N*C*K, 3), ...]
            img_feats: [(B, N*C, C', H, W), ...]
            img_depth: [(B*N*C, 64, H, W), ...]
        Returns:

        """
        uni_feats = []
        if img_feats is not None:
            uni_feats.append(
                self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            )
        if pts_feats is not None:
            uni_feats.append(pts_feats)

        uni_feats = sum(uni_feats)
        uni_feats = self.render_conv(uni_feats)

        batch_ret = []
        for bs_idx in range(len(img_metas)):
            i_ray_o, i_ray_d, i_ray_depth, scaled_points = (
                rays[bs_idx]["ray_o"],
                rays[bs_idx]["ray_d"],
                rays[bs_idx].get("depth", None),
                rays[bs_idx]["scaled_points"],
            )
            if self.training:
                ray_bundle = RayBundle(
                    origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth
                )
                preds_dict = self.render_model(
                    ray_bundle, uni_feats[bs_idx], points=scaled_points
                )
            else:
                assert i_ray_o.shape[0] == i_ray_d.shape[0]
                num_parts = (i_ray_o.shape[0] - 1) // self.part
                part_ret = []
                for num_part in range(num_parts + 1):
                    ray_bundle = RayBundle(
                        origins=i_ray_o[
                            num_part * self.part : (num_part + 1) * self.part
                        ],
                        directions=i_ray_d[
                            num_part * self.part : (num_part + 1) * self.part
                        ],
                    )
                    part_preds_dict = self.render_model(
                        ray_bundle, uni_feats[bs_idx], points=scaled_points
                    )
                    part_ret.append(part_preds_dict)
                preds_dict = {}
                for p_ret in part_ret:
                    for k, v in p_ret.items():
                        if k not in preds_dict:
                            preds_dict[k] = []
                        preds_dict[k].append(v)
                for k, v in preds_dict.items():
                    preds_dict[k] = torch.cat(v, dim=0)
                    assert (
                        preds_dict[k].shape[0] == i_ray_o.shape[0] == i_ray_d.shape[0]
                    )
            batch_ret.append(preds_dict)
        return batch_ret

    def sample_rays(self, pts, imgs, img_metas):
        lidar2img, lidar2cam = [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = np.asarray(lidar2img)
        lidar2cam = np.asarray(lidar2cam)

        new_pts = []
        for bs_idx, i_pts in enumerate(pts):
            dis = i_pts[:, :2].norm(dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            )
            new_pts.append(i_pts[dis_mask])
        pts = new_pts

        if (
            sparse_utils._cur_active_voxel is not None
            and self.ray_sampler_cfg.only_point_mask
        ):
            pc_range = torch.from_numpy(self.pc_range).to(pts[0])
            mask_voxel_size = (
                torch.from_numpy(self.unified_voxel_size).to(pts[0])
                / sparse_utils._cur_voxel_scale
            )
            mask_voxel_shape = (
                torch.from_numpy(self.unified_voxel_shape).to(pts[0].device)
                * sparse_utils._cur_voxel_scale
            )
            nonactive_voxel_mask = torch.zeros(
                (len(pts), *mask_voxel_shape.flip(dims=[0])),
                dtype=torch.bool,
                device=pts[0].device,
            )
            nonactive_voxel_mask[
                sparse_utils._cur_voxel_coords[~sparse_utils._cur_active_voxel]
                .long()
                .unbind(dim=1)
            ] = True
            new_pts = []
            for bs_idx in range(len(pts)):
                p_pts = pts[bs_idx]
                p_coords = (p_pts[:, :3] - pc_range[:3]) / mask_voxel_size
                kept = torch.all(
                    (p_coords >= torch.zeros_like(mask_voxel_shape))
                    & (p_coords < mask_voxel_shape),
                    dim=-1,
                )
                p_coords = F.pad(
                    p_coords[:, [2, 1, 0]].long(), (1, 0), mode="constant", value=bs_idx
                )
                p_coords, p_pts = p_coords[kept], p_pts[kept]
                p_nonactive_pts_mask = nonactive_voxel_mask[p_coords.unbind(dim=1)]
                new_pts.append(p_pts[p_nonactive_pts_mask])
            pts = new_pts

        if sparse_utils._cur_active is not None and self.ray_sampler_cfg.only_img_mask:
            active_mask = sparse_utils._get_active_ex_or_ii(imgs.shape[-2])
            assert (
                active_mask.shape[-2] == imgs.shape[-2]
                and active_mask.shape[-1] == imgs.shape[-1]
            )
            active_mask = active_mask.view(
                imgs.shape[0], -1, imgs.shape[-2], imgs.shape[-1]
            )

        batch_ret = []
        for bs_idx in range(len(pts)):
            i_imgs = imgs[bs_idx]
            i_pts = pts[bs_idx]
            i_lidar2img = i_pts.new_tensor(lidar2img[bs_idx]).flatten(0, 1)
            i_img2lidar = torch.inverse(
                i_lidar2img
            )  # TODO: Are img2lidar and img2cam consistent after image data augmentation?
            i_cam2lidar = torch.inverse(
                i_pts.new_tensor(lidar2cam[bs_idx]).flatten(0, 1)
            )
            i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
            i_pts_cam = torch.matmul(
                i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
            ).squeeze(-1)

            eps = 1e-5
            i_pts_mask = i_pts_cam[..., 2] > eps
            i_pts_cam[..., :2] = i_pts_cam[..., :2] / torch.maximum(
                i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
            )

            # (N*C, 3) [(H, W, 3), ...]
            pad_before_shape = torch.tensor(
                img_metas[bs_idx]["pad_before_shape"], device=i_pts_cam.device
            )
            Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

            # (N*C, M)
            i_pts_mask = (
                i_pts_mask
                & (i_pts_cam[..., 0] > 0)
                & (i_pts_cam[..., 0] < Ws - 1)
                & (i_pts_cam[..., 1] > 0)
                & (i_pts_cam[..., 1] < Hs - 1)
            )

            i_imgs = i_imgs.permute(0, 2, 3, 1)
            i_imgs = i_imgs * i_imgs.new_tensor(
                img_metas[0]["img_norm_cfg"]["std"]
            ) + i_imgs.new_tensor(img_metas[0]["img_norm_cfg"]["mean"])
            if not img_metas[0]["img_norm_cfg"]["to_rgb"]:
                i_imgs[..., [0, 1, 2]] = i_imgs[..., [2, 1, 0]]  # bgr->rgb

            i_sampled_ray_o, i_sampled_ray_d, i_sampled_rgb, i_sampled_depth = (
                [],
                [],
                [],
                [],
            )
            for c_idx in range(len(i_pts_mask)):
                j_sampled_all_pts, j_sampled_all_pts_cam, j_sampled_all_depth_mask = (
                    [],
                    [],
                    [],
                )

                """ sample points """
                j_sampled_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                j_sampled_pts_cam = i_pts_cam[c_idx][j_sampled_pts_idx]

                if self.ray_sampler_cfg.only_img_mask:
                    j_sampled_pts_mask = ~active_mask[
                        bs_idx,
                        c_idx,
                        j_sampled_pts_cam[:, 1].long(),
                        j_sampled_pts_cam[:, 0].long(),
                    ]
                    j_sampled_pts_idx = j_sampled_pts_mask.nonzero(as_tuple=True)[0]
                else:
                    j_sampled_pts_idx = torch.arange(
                        len(j_sampled_pts_cam),
                        dtype=torch.long,
                        device=j_sampled_pts_cam.device,
                    )

                point_nsample = min(
                    len(j_sampled_pts_idx),
                    int(len(j_sampled_pts_idx) * self.ray_sampler_cfg.point_ratio)
                    if self.ray_sampler_cfg.point_nsample == -1
                    else self.ray_sampler_cfg.point_nsample,
                )
                if point_nsample > 0:
                    replace_sample = (
                        True
                        if point_nsample > len(j_sampled_pts_idx)
                        else self.ray_sampler_cfg.replace_sample
                    )
                    j_sampled_pts_idx = j_sampled_pts_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pts_idx),
                                point_nsample,
                                replace=replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pts_idx.device)
                    ]
                    j_sampled_pts_cam = j_sampled_pts_cam[j_sampled_pts_idx]
                    j_sampled_pts = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pts_cam[..., :2]
                                * j_sampled_pts_cam[..., 2:3],
                                j_sampled_pts_cam[..., 2:],
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]
                    j_sampled_all_pts.append(j_sampled_pts)
                    j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :2])
                    j_sampled_all_depth_mask.append(
                        torch.ones_like(j_sampled_pts_cam[:, 0])
                    )

                """ sample pixels """
                if self.ray_sampler_cfg.merged_nsample - point_nsample > 0:
                    pixel_interval = self.ray_sampler_cfg.pixel_interval
                    sky_region = self.ray_sampler_cfg.sky_region
                    tx = torch.arange(
                        0,
                        Ws[c_idx, 0],
                        pixel_interval,
                        device=i_imgs.device,
                        dtype=i_imgs.dtype,
                    )
                    ty = torch.arange(
                        int(sky_region * Hs[c_idx, 0]),
                        Hs[c_idx, 0],
                        pixel_interval,
                        device=i_imgs.device,
                        dtype=i_imgs.dtype,
                    )
                    pixels_y, pixels_x = torch.meshgrid(ty, tx)
                    i_pixels_cam = torch.stack([pixels_x, pixels_y], dim=-1)

                    j_sampled_pixels_cam = i_pixels_cam.flatten(0, 1)
                    if self.ray_sampler_cfg.only_img_mask:
                        j_sampled_pixels_mask = ~active_mask[
                            bs_idx,
                            c_idx,
                            j_sampled_pixels_cam[:, 1].long(),
                            j_sampled_pixels_cam[:, 0].long(),
                        ]  # (Q,)
                        j_sampled_pixels_idx = j_sampled_pixels_mask.nonzero(
                            as_tuple=True
                        )[0]
                    else:
                        j_sampled_pixels_idx = torch.arange(
                            len(j_sampled_pixels_cam),
                            dtype=torch.long,
                            device=j_sampled_pixels_cam.device,
                        )

                    pixel_nsample = min(
                        len(j_sampled_pixels_idx),
                        self.ray_sampler_cfg.merged_nsample - point_nsample,
                    )
                    j_sampled_pixels_idx = j_sampled_pixels_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pixels_idx),
                                pixel_nsample,
                                replace=self.ray_sampler_cfg.replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pixels_idx.device)
                    ]
                    j_sampled_pixels_cam = j_sampled_pixels_cam[j_sampled_pixels_idx]
                    j_sampled_pixels = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pixels_cam,
                                torch.ones_like(j_sampled_pixels_cam),
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]
                    j_sampled_all_pts.append(j_sampled_pixels)
                    j_sampled_all_pts_cam.append(j_sampled_pixels_cam)
                    j_sampled_all_depth_mask.append(
                        torch.zeros_like(j_sampled_pixels_cam[:, 0])
                    )

                if len(j_sampled_all_pts) > 0:
                    """merge"""
                    j_sampled_all_pts = torch.cat(j_sampled_all_pts, dim=0)
                    j_sampled_all_pts_cam = torch.cat(j_sampled_all_pts_cam, dim=0)
                    j_sampled_all_depth_mask = torch.cat(
                        j_sampled_all_depth_mask, dim=0
                    )

                    unscaled_ray_o = i_cam2lidar[c_idx : c_idx + 1, :3, 3].repeat(
                        j_sampled_all_pts.shape[0], 1
                    )
                    i_sampled_ray_o.append(
                        unscaled_ray_o * self.render_model.scale_factor
                    )
                    i_sampled_ray_d.append(
                        F.normalize(j_sampled_all_pts - unscaled_ray_o, dim=-1)
                    )
                    sampled_depth = (
                        torch.norm(
                            j_sampled_all_pts - unscaled_ray_o, dim=-1, keepdim=True
                        )
                        * self.render_model.scale_factor
                    )
                    sampled_depth[j_sampled_all_depth_mask == 0] = -1.0
                    i_sampled_depth.append(sampled_depth)
                    i_sampled_rgb.append(
                        i_imgs[
                            c_idx,
                            j_sampled_all_pts_cam[:, 1].long(),
                            j_sampled_all_pts_cam[:, 0].long(),
                        ]
                        / 255.0
                    )

                    # with open('outputs/{}_{}.pkl'.format(img_metas[bs_idx]['sample_idx'], c_idx), 'wb') as f:
                    #     pickle.dump({
                    #         'pts': i_pts[:, :3].cpu().numpy(),
                    #         'img':  i_imgs[c_idx].cpu().numpy(),
                    #         'pts_cam': i_pts_cam[c_idx][i_pts_mask[c_idx].nonzero(as_tuple=True)[0]][:, :3].cpu().numpy(),
                    #         'sampled_pts': j_sampled_all_pts.cpu().numpy(),
                    #         'ray_o': unscaled_ray_o.cpu().numpy()
                    #     }, f)
                    #     print('save to outputs/{}_{}.pkl'.format(img_metas[bs_idx]['sample_idx'], c_idx))

            batch_ret.append(
                {
                    "ray_o": torch.cat(i_sampled_ray_o, dim=0),
                    "ray_d": torch.cat(i_sampled_ray_d, dim=0),
                    "rgb": torch.cat(i_sampled_rgb, dim=0),
                    "depth": torch.cat(i_sampled_depth, dim=0),
                    "scaled_points": pts[bs_idx][:, :3]
                    * self.render_model.scale_factor,
                }
            )

        return batch_ret

    def sample_rays_test(self, pts, imgs, img_metas):
        lidar2img, lidar2cam = [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = np.asarray(lidar2img)  # (B, N, C, 4, 4)
        lidar2cam = np.asarray(lidar2cam)

        batch_ret = []
        for bs_idx in range(len(img_metas)):
            i_imgs = imgs[bs_idx]  # (N*C, 3, H, W)
            l = 2
            H, W = img_metas[0]["img_shape"][0][0], img_metas[0]["img_shape"][0][1]
            assert H == i_imgs.shape[-2] and W == i_imgs.shape[-1]
            tx = torch.linspace(0, W - 1, W // l, device=i_imgs.device)
            ty = torch.linspace(0, H - 1, H // l, device=i_imgs.device)
            pixels_y, pixels_x = torch.meshgrid(ty, tx)
            i_pts_cam = torch.stack(
                [
                    pixels_x,
                    pixels_y,
                    torch.ones_like(pixels_y),
                    torch.ones_like(pixels_y),
                ],
                dim=-1,
            ).view(
                1, -1, 4, 1
            )  # (H, W, 4) -> (1, H*W, 4, 1), [x, y, 1, 1]
            i_img2lidar = torch.inverse(
                i_imgs.new_tensor(lidar2img[bs_idx]).flatten(0, 1)
            ).unsqueeze(
                1
            )  # (N*C, 1, 4, 4)
            i_pts = torch.matmul(i_img2lidar, i_pts_cam).squeeze(-1)  # (N*C, H*W, 4)
            i_cam2lidar = torch.inverse(
                i_imgs.new_tensor(lidar2cam[bs_idx]).flatten(0, 1)
            )  # (N*C, 4, 4)

            i_sampled_ray_o = (
                i_cam2lidar[..., :3, 3].unsqueeze(1).repeat(1, i_pts.shape[1], 1)
            )  # (N*C, H*W, 3)  # camera position, in LiDAR coordinate
            i_sampled_ray_d = F.normalize(
                i_pts[..., :3] - i_sampled_ray_o, dim=-1
            )  # (N*C, H*W, 3), normalized direction vector, in LiDAR coordinate
            i_sampled_ray_d_norm = torch.norm(
                i_pts[..., :3] - i_sampled_ray_o, dim=-1, keepdim=True
            )
            i_sampled_ray_o = i_sampled_ray_o * self.render_model.scale_factor

            vis_img = i_imgs.permute(0, 2, 3, 1)  # (N*C, H, W, 3)
            vis_img = vis_img * i_imgs.new_tensor(
                img_metas[0]["img_norm_cfg"]["std"]
            ) + i_imgs.new_tensor(img_metas[0]["img_norm_cfg"]["mean"])
            if not img_metas[0]["img_norm_cfg"]["to_rgb"]:
                vis_img[..., [0, 1, 2]] = vis_img[..., [2, 1, 0]]  # bgr->rgb

            batch_ret.append(
                {
                    "ray_o": i_sampled_ray_o.flatten(0, 1),  # (N*C*H*W, 3)
                    "ray_d": i_sampled_ray_d.flatten(0, 1),  # (N*C*H*W, 3)
                    # 'ray_d_norm': i_sampled_ray_d_norm.flatten(0, 1),  # (N*C*H*W, 1)
                    "rgb": vis_img,
                }
            )

        return batch_ret
