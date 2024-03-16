_base_ = [
    "../../../configs/_base_/datasets/nus-3d.py",
    "../../../configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
unified_voxel_size = [0.8, 0.8, 1.6]
frustum_range = [-1, -1, 0.0, -1, -1, 64.0]
frustum_size = [-1, -1, 1.0]
cam_sweep_num = 1
fp16_enabled = True
unified_voxel_shape = [
    int((point_cloud_range[3] - point_cloud_range[0]) / unified_voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / unified_voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / unified_voxel_size[2]),
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    cam_sweep_num=cam_sweep_num,
)

model = dict(
    type="UVTR",
    use_grid_mask=True,
    img_backbone=dict(
        type="MaskConvNeXt",
        arch="small",
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        norm_out=True,
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="data/ckpts/processed_convnext_small_1k_224_ema.pth",
        ),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=128,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    depth_head=dict(type="SimpleDepth"),
    pts_bbox_head=dict(
        type="UVTRHead",
        view_cfg=dict(
            type="Uni3DViewTrans",
            pc_range=point_cloud_range,
            voxel_size=unified_voxel_size,
            voxel_shape=unified_voxel_shape,
            frustum_range=frustum_range,
            frustum_size=frustum_size,
            num_convs=3,
            kernel_size=(3, 3, 3),
            embed_dim=128,
            keep_sweep_dim=True,
            fp16_enabled=fp16_enabled,
        ),
        # transformer_cfg
        num_query=900,
        num_classes=10,
        in_channels=128,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type="Uni3DDETR",
            fp16_enabled=fp16_enabled,
            decoder=dict(
                type="UniTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=128,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="UniCrossAtten",
                            num_points=1,
                            embed_dims=128,
                            num_sweeps=cam_sweep_num,
                            fp16_enabled=fp16_enabled,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=128,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    norm_cfg=dict(type="LN"),
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=64, normalize=True, offset=-0.5
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            )
        )
    ),
)

dataset_type = "NuScenesSweepDataset"
data_root = "data/nuscenes/"

file_client_args = dict(backend="disk")

train_pipeline = [
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(
        type="UnifiedRotScaleTransFlip",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["gt_bboxes_3d", "gt_labels_3d", "img"]),
]
test_pipeline = [
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["img"]),
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root
        + "nuscenes_unified_infos_train.pkl",  # please change to your own info file
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d="LiDAR",
        load_interval=2,
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
    ),  # please change to your own info file
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
    ),
)  # please change to your own info file

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 12
evaluation = dict(interval=4, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)

find_unused_parameters = True
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
load_from = "work_dirs/uvtr_cam_vs0.1_pretrain/epoch_12.pth"
resume_from = None
# fp16 setting
fp16 = dict(loss_scale=32.0)
