from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from pdb import set_trace
import pickle
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet


@DETECTORS.register_module()
class UVTRSSL(MVXTwoStageDetector):
    """UVTR."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        depth_head=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(UVTRSSL, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        if self.with_img_backbone:
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
            if depth_head is not None:
                depth_dim = self.pts_bbox_head.view_trans.depth_dim
                dhead_type = depth_head.pop("type", "SimpleDepth")
                if dhead_type == "SimpleDepth":
                    self.depth_net = Conv2d(out_channels, depth_dim, kernel_size=1)
                else:
                    self.depth_net = DepthNet(
                        out_channels, out_channels, depth_dim, **depth_head
                    )
            self.depth_head = depth_head

        if pts_middle_encoder:
            self.pts_fp16 = (
                True if hasattr(self.pts_middle_encoder, "fp16_enabled") else False
            )

    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, "depth_head") and self.depth_head is not None

    @force_fp32()
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.pts_voxel_encoder or pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        if not self.pts_fp16:
            voxel_features = voxel_features.float()

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = self.input_proj(img_feat)
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"))
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        if hasattr(self, "img_backbone"):
            img_feats = self.extract_img_feat(img, img_metas)
            img_depth = self.pred_depth(
                img=img, img_metas=img_metas, img_feats=img_feats
            )
        else:
            img_feats, img_depth = None, None

        if hasattr(self, "pts_voxel_encoder"):
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = None

        return pts_feats, img_feats, img_depth

    @auto_fp16(apply_to=("img"))
    def pred_depth(self, img, img_metas, img_feats=None):
        if img_feats is None or not self.with_depth_head:
            return None
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        depth = []
        for _feat in img_feats:
            _depth = self.depth_net(_feat.view(-1, *_feat.shape[-3:]))
            _depth = _depth.softmax(dim=1)
            depth.append(_depth)
        return depth

    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
        self, pts_feats, img_feats, points, img, img_metas, img_depth
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        batch_rays = self.pts_bbox_head.sample_rays(points, img, img_metas)
        out_dict = self.pts_bbox_head(
            pts_feats, img_feats, batch_rays, img_metas, img_depth
        )
        losses = self.pts_bbox_head.loss(out_dict, batch_rays)
        if self.with_depth_head and hasattr(self.pts_bbox_head.view_trans, "loss"):
            losses.update(
                self.pts_bbox_head.view_trans.loss(img_depth, points, img, img_metas)
            )
        return losses

    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, points=None, img_metas=None, img=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats, img_feats, points, img, img_metas, img_depth
        )
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        num_augs = len(img_metas)
        if points is not None:
            if num_augs != len(points):
                raise ValueError(
                    "num of augmentations ({}) != num of image meta ({})".format(
                        len(points), len(img_metas)
                    )
                )

        assert num_augs == 1
        if not isinstance(img_metas[0], list):
            img_metas = [img_metas]
        if not isinstance(img, list):
            img = [img]
        results = self.simple_test(img_metas[0], points, img[0])

        return results

    def simple_test(self, img_metas, points=None, img=None):
        """Test function without augmentaiton."""
        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        batch_rays = self.pts_bbox_head.sample_rays_test(points, img, img_metas)
        results = self.pts_bbox_head(
            pts_feats, img_feats, batch_rays, img_metas, img_depth
        )
        with open("outputs/{}.pkl".format(img_metas[0]["sample_idx"]), "wb") as f:
            H, W = img_metas[0]["img_shape"][0][0], img_metas[0]["img_shape"][0][1]
            num_cam = len(img_metas[0]["img_shape"])
            l = 2
            # init_weights = results[0]["vis_weights"]
            # init_weights = init_weights.reshape(num_cam, -1, *init_weights.shape[1:])
            # init_sampled_points = results[0]["vis_sampled_points"]
            # init_sampled_points = init_sampled_points.reshape(
            #     num_cam, -1, *init_sampled_points.shape[1:]
            # )
            # pts_idx = np.random.randint(
            #     0, high=init_sampled_points.shape[1], size=(256,), dtype=int
            # )
            # init_weights = init_weights[:, pts_idx]
            # init_sampled_points = init_sampled_points[:, pts_idx]
            pickle.dump(
                {
                    "render_rgb": results[0]["rgb"]
                    .reshape(num_cam, H // l, W // l, 3)
                    .detach()
                    .cpu()
                    .numpy(),
                    "render_depth": results[0]["depth"]
                    .reshape(num_cam, H // l, W // l, 1)
                    .detach()
                    .cpu()
                    .numpy(),
                    "rgb": batch_rays[0]["rgb"].detach().cpu().numpy(),
                    # "scaled_points": results[0]["scaled_points"].detach().cpu().numpy(),
                    # "points": points[0].detach().cpu().numpy(),
                    # "lidar2img": np.asarray(img_metas[0]["lidar2img"])[
                    #     :, 0
                    # ],  # (N, 4, 4)
                    # 'weights': results[0]['vis_weights'].detach().cpu().numpy(),
                    # 'sampled_points': results[0]['vis_sampled_points'].detach().cpu().numpy(),
                    # "init_weights": init_weights.detach().cpu().numpy(),
                    # "init_sampled_points": init_sampled_points.detach().cpu().numpy(),
                },
                f,
            )
            print("save to outputs/{}.pkl".format(img_metas[0]["sample_idx"]))
        set_trace()
        return results

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        if points is None:
            points = [None] * len(img_metas)
        pts_feats, img_feats, img_depths = multi_apply(
            self.extract_feat, points, imgs, img_metas
        )
        return pts_feats, img_feats, img_depths
