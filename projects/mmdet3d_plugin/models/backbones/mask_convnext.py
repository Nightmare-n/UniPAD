import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from ..utils.sparse_utils import SparseLayerNorm, SparseConvNeXtBlock
from timm.models.layers import trunc_normal_, DropPath
from itertools import chain
from typing import Sequence
from ..utils import sparse_utils


@BACKBONES.register_module()
class MaskConvNeXt(BaseModule):
    """ConvNeXt v1&v2 backbone.

    A PyTorch implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_ and
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    <http://arxiv.org/abs/2301.00808>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501

    arch_settings = {
        "tiny": {"depths": [3, 3, 9, 3], "channels": [96, 192, 384, 768]},
        "small": {"depths": [3, 3, 27, 3], "channels": [96, 192, 384, 768]},
        "base": {"depths": [3, 3, 27, 3], "channels": [128, 256, 512, 1024]},
        "large": {"depths": [3, 3, 27, 3], "channels": [192, 384, 768, 1536]},
    }

    def __init__(
        self,
        arch="tiny",
        in_channels=3,
        stem_patch_size=4,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=-1,
        norm_out=False,
        frozen_stages=0,
        with_cp=False,
        init_cfg=[
            dict(type="TruncNormal", layer=["Conv2d", "Linear"], std=0.02, bias=0.0),
            dict(type="Constant", layer=["LayerNorm"], val=1.0, bias=0.0),
        ],
        mae_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, (
                f"Unavailable arch, please choose from "
                f"({set(self.arch_settings)}) or pass a dict."
            )
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert "depths" in arch and "channels" in arch, (
                f'The arch dict must have "depths" and "channels", '
                f"but got {list(arch.keys())}."
            )

        self.depths = arch["depths"]
        self.channels = arch["channels"]
        assert (
            isinstance(self.depths, Sequence)
            and isinstance(self.channels, Sequence)
            and len(self.depths) == len(self.channels)
        ), (
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) '
            "should be both sequence with the same length."
        )

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), (
            f'"out_indices" must by a sequence or int, '
            f"get {type(out_indices)} instead."
        )
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f"Invalid out_indices {index}"
        self.out_indices = out_indices
        self.norm_out = norm_out
        self.frozen_stages = frozen_stages

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        block_idx = 0

        sparse = mae_cfg is not None
        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size,
            ),
            SparseLayerNorm(
                self.channels[0], eps=1e-6, data_format="channel_first", sparse=sparse
            ),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    SparseLayerNorm(
                        self.channels[i - 1],
                        eps=1e-6,
                        data_format="channel_first",
                        sparse=sparse,
                    ),
                    nn.Conv2d(self.channels[i - 1], channels, kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = nn.Sequential(
                *[
                    SparseConvNeXtBlock(
                        in_channels=channels,
                        drop_path_rate=dpr[block_idx + j],
                        layer_scale_init_value=layer_scale_init_value,
                        with_cp=with_cp,
                        sparse=sparse,
                    )
                    for j in range(depth)
                ]
            )
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices and self.norm_out:
                norm_layer = SparseLayerNorm(
                    channels, eps=1e-6, data_format="channel_first", sparse=sparse
                )
                self.add_module(f"norm{i}", norm_layer)

        self.mae_cfg = mae_cfg
        if mae_cfg is not None:
            self.to_sparse()
            if mae_cfg.learnable:
                for i in self.out_indices:
                    p = nn.Parameter(
                        torch.zeros(
                            1,
                            mae_cfg.downsample_dim // 2 ** (len(self.stages) - i - 1),
                            1,
                            1,
                        )
                    )
                    trunc_normal_(p, mean=0, std=0.02, a=-0.02, b=0.02)
                    self.register_parameter(f"mtoken{i}", p)

        self._freeze_stages()

    def to_sparse(self, verbose=False, sbn=False):
        for name, child in self.named_children():
            self.add_module(
                name,
                sparse_utils.dense_model_to_sparse(
                    child, name, verbose=verbose, sbn=sbn
                ),
            )

    def forward(self, x):
        input = x

        if self.mae_cfg is not None:
            B, _, H, W = x.shape
            downsample_scale = self.mae_cfg.downsample_scale  # 32
            h, w = H // downsample_scale, W // downsample_scale
            active_b1hw = sparse_utils.random_masking(
                B, h, w, self.mae_cfg.mask_ratio, x.device
            )
            sparse_utils._cur_active = active_b1hw
            active_b1HW = active_b1hw.repeat_interleave(
                downsample_scale, 2
            ).repeat_interleave(downsample_scale, 3)
            x = x * active_b1HW

        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                out_x = getattr(self, f"norm{i}")(x) if self.norm_out else x
                if self.mae_cfg is not None and self.mae_cfg.learnable:
                    scale = out_x.shape[-2] // active_b1hw.shape[-2]
                    out_x = torch.where(
                        active_b1hw.repeat_interleave(scale, 2)
                        .repeat_interleave(scale, 3)
                        .expand_as(out_x),
                        out_x,
                        getattr(self, f"mtoken{i}").expand_as(out_x).to(out_x.dtype),
                    )
                outs.append(out_x)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(), stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(MaskConvNeXt, self).train(mode)
        self._freeze_stages()
