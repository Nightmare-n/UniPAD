import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmdet.models.utils import build_transformer
from mmdet.core import multi_apply, multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
    build_bbox_coder,
)
from mmcv.cnn import xavier_init, constant_init
from .. import utils
from projects.mmdet3d_plugin.core.bbox.iou_calculators import PairedBboxOverlaps3D


@HEADS.register_module()
class UVTRDNHead(BaseModule):
    """Head of UVTR.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        in_channels,
        embed_dims=128,
        num_query=900,
        num_reg_fcs=2,
        bg_cls_weight=0,
        num_classes=10,
        code_size=10,
        sync_cls_avg_factor=False,
        unified_conv=None,
        view_cfg=None,
        with_box_refine=False,
        transformer=None,
        bbox_coder=None,
        loss_bbox=None,
        loss_cls=None,
        train_cfg=None,
        test_cfg=None,
        code_weights=None,
        split=0.75,
        dn_weight=1.0,
        iou_weight=0.5,
        scalar=10,
        bbox_noise_scale=1.0,
        bbox_noise_trans=0.0,
        init_cfg=None,
        fp16_enabled=True,
        **kwargs,
    ):
        super(UVTRDNHead, self).__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_box_refine = with_box_refine
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_reg_fcs = num_reg_fcs
        self.bg_cls_weight = bg_cls_weight
        self.unified_conv = unified_conv
        self.scalar = scalar
        self.bbox_noise_scale = bbox_noise_scale
        self.bbox_noise_trans = bbox_noise_trans
        self.dn_weight = dn_weight
        self.iou_weight = iou_weight
        self.split = split
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

        self.code_size = code_size
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.code_weights = nn.Parameter(
            torch.tensor(code_weights, requires_grad=False), requires_grad=False
        )
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if view_cfg is not None:
            vtrans_type = view_cfg.pop("type", "Uni3DViewTrans")
            self.view_trans = getattr(utils, vtrans_type)(**view_cfg)

        if self.unified_conv is not None:
            self.conv_layer = []
            in_c = (
                embed_dims * 2 if self.unified_conv["fusion"] == "cat" else embed_dims
            )
            for k in range(self.unified_conv["num_conv"]):
                conv = nn.Sequential(
                    nn.Conv3d(
                        in_c, embed_dims, kernel_size=3, stride=1, padding=1, bias=True
                    ),
                    nn.BatchNorm3d(embed_dims),
                    nn.ReLU(inplace=True),
                )
                in_c = embed_dims
                self.add_module("{}_head_{}".format("conv_trans", k + 1), conv)
                self.conv_layer.append(conv)

        self.transformer = build_transformer(transformer)
        self._init_layers()

        self.iou_calculator = PairedBboxOverlaps3D(coordinate="lidar")

        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type="PseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        iou_branch = []
        for _ in range(self.num_reg_fcs):
            iou_branch.append(Linear(self.embed_dims, self.embed_dims))
            iou_branch.append(nn.ReLU())
        iou_branch.append(Linear(self.embed_dims, 1))
        iou_branch = nn.Sequential(*iou_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.iou_branches = _get_clones(iou_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.iou_branches = nn.ModuleList([iou_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        self.query_embedding = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.reference_points = nn.Embedding(self.num_query, 3)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    def prepare_for_dn(
        self, batch_size, reference_points, gt_bboxes_3d=None, gt_labels_3d=None
    ):
        if self.training:
            assert gt_bboxes_3d is not None and gt_labels_3d is not None
            device = gt_labels_3d[0].device
            gt_bboxes_3d = [
                torch.cat(
                    [gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]], dim=1
                ).to(device)
                for gt_bboxes in gt_bboxes_3d
            ]
            known_num = [t.size(0) for t in gt_labels_3d]
            batch_idx = torch.cat(
                [torch.full_like(t, i) for i, t in enumerate(gt_labels_3d)], dim=0
            ).long()
            gt_bboxes_3d = torch.cat(gt_bboxes_3d, dim=0)  # (N1+N2+..., 9)
            gt_labels_3d = torch.cat(gt_labels_3d, dim=0)  # (N1+N2+...,)

            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_labels = gt_labels_3d.repeat(groups)
            known_bid = batch_idx.repeat(groups)
            known_bboxs = gt_bboxes_3d.repeat(groups, 1)  # (N1+N2+... +N1+N2+..., 9)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = (
                    torch.rand_like(known_bbox_center) * 2 - 1.0
                )  # (N1+N2+... + N1+N2+..., 3), [-1, 1]
                known_bbox_center += torch.mul(rand_prob, diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (
                    known_bbox_center[..., 0:1] - self.pc_range[0]
                ) / (
                    self.pc_range[3] - self.pc_range[0]
                )  # normalize to [0, 1]
                known_bbox_center[..., 1:2] = (
                    known_bbox_center[..., 1:2] - self.pc_range[1]
                ) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (
                    known_bbox_center[..., 2:3] - self.pc_range[2]
                ) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, p=2, dim=1) > self.split
                known_labels[mask] = self.num_classes  # set negative for far centers

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padded_reference_points = (
                torch.cat(
                    [reference_points.new_zeros(pad_size, 3), reference_points], dim=0
                )
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )  # (bs, pad+num_query, 3)

            map_known_indice = torch.cat(
                [torch.arange(num) for num in known_num], dim=0
            )  # (N1+N2+...,), [0,1,0,1,2,3,...]
            map_known_indice = torch.cat(
                [map_known_indice + single_pad * i for i in range(groups)], dim=0
            ).long()  # (N1+N2+ ... +N1+N2,)
            padded_reference_points[known_bid, map_known_indice] = known_bbox_center.to(
                reference_points.device
            )
            # bs:0 -> index:g0-[0,1, , , ],g1-[5,6, , , ]
            # bs:1 -> index:g0-[0,1,2,3, ],g1-[5,6,7,8, ]

            tgt_size = pad_size + self.num_query
            attn_mask = torch.zeros(
                tgt_size, tgt_size, dtype=torch.bool, device=reference_points.device
            )
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[
                        single_pad * i : single_pad * (i + 1),
                        single_pad * (i + 1) : pad_size,
                    ] = True
                if i == groups - 1:
                    attn_mask[
                        single_pad * i : single_pad * (i + 1), : single_pad * i
                    ] = True
                else:
                    attn_mask[
                        single_pad * i : single_pad * (i + 1),
                        single_pad * (i + 1) : pad_size,
                    ] = True
                    attn_mask[
                        single_pad * i : single_pad * (i + 1), : single_pad * i
                    ] = True

            mask_dict = {
                "known_bid": known_bid,
                "map_known_indice": map_known_indice,
                "known_lbs_bboxes": (known_labels, known_bboxs),
                "pad_size": pad_size,
            }
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(
        self,
        pts_feats,
        img_feats,
        img_metas,
        img_depth,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        with_image, with_point = True, True
        if img_feats is None:
            with_image = False
        elif isinstance(img_feats, dict) and img_feats["key"] is None:
            with_image = False

        if pts_feats is None:
            with_point = False
        elif isinstance(pts_feats, dict) and pts_feats["key"] is None:
            with_point = False
            pts_feats = None

        # transfer to voxel level
        if with_image:
            img_feats = self.view_trans(
                img_feats, img_metas=img_metas, img_depth=img_depth
            )
        # shape: (N, L, C, D, H, W)
        if with_point:
            if len(pts_feats.shape) == 5:
                pts_feats = pts_feats.unsqueeze(1)

        if self.unified_conv is not None:
            raw_shape = pts_feats.shape
            if self.unified_conv["fusion"] == "sum":
                unified_feats = pts_feats.flatten(1, 2) + img_feats.flatten(1, 2)
            else:
                unified_feats = torch.cat(
                    [pts_feats.flatten(1, 2), img_feats.flatten(1, 2)], dim=1
                )
            for layer in self.conv_layer:
                unified_feats = layer(unified_feats)
            unified_feats = unified_feats.reshape(*raw_shape)
        else:
            unified_feats = pts_feats if pts_feats is not None else img_feats

        unified_feats = unified_feats.squeeze(1)  # (B, C, Z, Y, X)

        bs = unified_feats.shape[0]

        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            bs, reference_points, gt_bboxes_3d, gt_labels_3d
        )

        reference_points = inverse_sigmoid(reference_points)
        query_pos = self.query_embedding(reference_points)
        hs, inter_references = self.transformer(
            query=torch.zeros_like(query_pos).permute(1, 0, 2),
            value=unified_feats,
            query_pos=query_pos.permute(1, 0, 2),
            key_pos=None,
            reference_points=reference_points,
            reg_branches=(
                self.reg_branches if self.with_box_refine else None
            ),  # noqa:E501
            attn_masks=[attn_mask, None],
        )
        hs = hs.permute(0, 2, 1, 3)  # (L, N, B, C) -> (L, B, N, C)

        outputs_classes, outputs_coords, outputs_ious = [], [], []
        for lvl in range(hs.shape[0]):
            # only backward for init_reference_points
            reference = (
                reference_points if lvl == 0 else inter_references[lvl - 1]
            )  # (B, N, 3)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            outputs_iou = self.iou_branches[lvl](hs[lvl])
            outputs_coord = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            outputs_coord[..., 0:2] = (
                outputs_coord[..., 0:2] + reference[..., 0:2]
            ).sigmoid()
            outputs_coord[..., 4:5] = (
                outputs_coord[..., 4:5] + reference[..., 2:3]
            ).sigmoid()

            # transfer to lidar system
            outputs_coord[..., 0:1] = (
                outputs_coord[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                + self.pc_range[0]
            )
            outputs_coord[..., 1:2] = (
                outputs_coord[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                + self.pc_range[1]
            )
            outputs_coord[..., 4:5] = (
                outputs_coord[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_classes.append(outputs_class)
            outputs_ious.append(outputs_iou)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)  # (L, B, N, num_class)
        outputs_ious = torch.stack(outputs_ious)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "all_cls_scores": outputs_classes,
            "all_iou_preds": outputs_ious,
            "all_bbox_preds": outputs_coords,
        }

        if mask_dict is not None and mask_dict["pad_size"] > 0:
            for key in list(outs.keys()):
                outs["dn_" + key] = outs[key][:, :, : mask_dict["pad_size"], :]
                outs[key] = outs[key][:, :, mask_dict["pad_size"] :, :]
            outs["dn_mask_dicts"] = mask_dict

        return outs

    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        try:
            assign_result = self.assigner.assign(
                bbox_pred, cls_score, gt_bboxes, gt_labels
            )
        except:
            print(
                "bbox_pred:{}, cls_score:{}, gt_bboxes:{}, gt_labels:{}".format(
                    (bbox_pred.max(), bbox_pred.min()),
                    (cls_score.max(), cls_score.min()),
                    (gt_bboxes.max(), gt_bboxes.min()),
                    gt_labels,
                )
            )
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., : gt_bboxes.shape[1]]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    def get_targets(
        self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list
    ):
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def dn_loss_single(self, cls_scores, bbox_preds, iou_preds, mask_dict):
        known_labels, known_bboxs = mask_dict["known_lbs_bboxes"]
        known_bid, map_known_indice = (
            mask_dict["known_bid"].long(),
            mask_dict["map_known_indice"].long(),
        )

        cls_scores = cls_scores[known_bid, map_known_indice]
        bbox_preds = bbox_preds[known_bid, map_known_indice]
        iou_preds = iou_preds[known_bid, map_known_indice]
        num_tgt = map_known_indice.numel()

        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split * self.split
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(bbox_preds)
        bbox_code_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_code_weights[isnotnan, :10],
            avg_factor=num_tgt,
        )

        denormalized_bbox_preds = denormalize_bbox(
            bbox_preds[isnotnan, :8].detach(), self.pc_range
        ).clone()
        denormalized_bbox_preds[:, 2] = (
            denormalized_bbox_preds[:, 2] - denormalized_bbox_preds[:, 5] * 0.5
        )

        denormalized_bbox_targets = known_bboxs[
            isnotnan, :7
        ].clone()  # (x, y, z, w, l, h, rot)
        denormalized_bbox_targets[:, 2] = (
            denormalized_bbox_targets[:, 2] - denormalized_bbox_targets[:, 5] * 0.5
        )

        iou_preds = iou_preds[isnotnan, 0]
        iou_targets = self.iou_calculator(
            denormalized_bbox_preds, denormalized_bbox_targets
        )
        valid_index = torch.nonzero(
            iou_targets * bbox_weights[isnotnan, 0], as_tuple=True
        )[0]
        num_pos = valid_index.shape[0]
        iou_targets = iou_targets * 2 - 1
        loss_iou = F.l1_loss(
            iou_preds[valid_index], iou_targets[valid_index], reduction="sum"
        ) / max(num_pos, 1)

        # loss_cls = torch.nan_to_num(loss_cls)
        # loss_bbox = torch.nan_to_num(loss_bbox)
        return (
            self.dn_weight * loss_cls,
            self.dn_weight * loss_bbox,
            self.dn_weight * self.iou_weight * loss_iou,
        )

    def loss_single(
        self, cls_scores, bbox_preds, iou_preds, gt_bboxes_list, gt_labels_list
    ):
        batch_size = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(batch_size)]
        bbox_preds_list = [bbox_preds[i] for i in range(batch_size)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_targets(
            cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list
        )

        labels = torch.cat(labels_list, dim=0)
        label_weights = torch.cat(label_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        # classification loss
        cls_scores = cls_scores.flatten(0, 1)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.flatten(0, 1)
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_code_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_code_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )

        denormalized_bbox_preds = denormalize_bbox(
            bbox_preds[isnotnan, :8].detach(), self.pc_range
        ).clone()
        denormalized_bbox_preds[:, 2] = (
            denormalized_bbox_preds[:, 2] - denormalized_bbox_preds[:, 5] * 0.5
        )

        denormalized_bbox_targets = bbox_targets[
            isnotnan, :7
        ].clone()  # (x, y, z, w, l, h, rot)
        denormalized_bbox_targets[:, 2] = (
            denormalized_bbox_targets[:, 2] - denormalized_bbox_targets[:, 5] * 0.5
        )

        iou_preds = iou_preds.flatten(0, 1)[isnotnan, 0]
        iou_targets = self.iou_calculator(
            denormalized_bbox_preds, denormalized_bbox_targets
        )
        valid_index = torch.nonzero(
            iou_targets * bbox_weights[isnotnan, 0], as_tuple=True
        )[0]
        num_pos = valid_index.shape[0]
        iou_targets = iou_targets * 2 - 1
        loss_iou = F.l1_loss(
            iou_preds[valid_index], iou_targets[valid_index], reduction="sum"
        ) / max(num_pos, 1)

        # loss_cls = torch.nan_to_num(loss_cls)
        # loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, self.iou_weight * loss_iou

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts):
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        all_iou_preds = preds_dicts["all_iou_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        # calculate class and box loss
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_iou_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            num_dec_layer += 1

        dn_all_cls_scores, dn_all_bbox_preds, dn_all_iou_preds = (
            preds_dicts["dn_all_cls_scores"],
            preds_dicts["dn_all_bbox_preds"],
            preds_dicts["dn_all_iou_preds"],
        )
        dn_all_mask_dicts = [
            preds_dicts["dn_mask_dicts"] for _ in range(num_dec_layers)
        ]
        dn_losses_cls, dn_losses_bbox, dn_losses_iou = multi_apply(
            self.dn_loss_single,
            dn_all_cls_scores,
            dn_all_bbox_preds,
            dn_all_iou_preds,
            dn_all_mask_dicts,
        )
        loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
        loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
        loss_dict["dn_loss_iou"] = dn_losses_iou[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
            dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.dn_loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.dn_loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.dn_loss_iou"] = loss_iou_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]["box_type_3d"](bboxes, bboxes.size(-1))
            scores = preds["scores"]
            labels = preds["labels"]
            ret_list.append([bboxes, scores, labels])
        return ret_list
