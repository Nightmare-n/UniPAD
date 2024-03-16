_base_ = [
    "../../../configs/_base_/datasets/nus-3d.py",
    "../../../configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
unified_voxel_size = [0.6, 0.6, 1.6]
frustum_range = [0, 0, 1.0, 1600, 928, 60.0]
frustum_size = [32.0, 32.0, 0.5]
cam_sweep_num = 1
lidar_sweep_num = 10
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
    type="UVTRDN",
    use_grid_mask=True,
    img_backbone=dict(
        type="MaskConvNeXt",
        arch="small",
        drop_path_rate=0.2,
        out_indices=(3),
        norm_out=True,
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="data/ckpts/processed_convnext_small_1k_224_ema.pth",
        ),
    ),
    img_neck=dict(
        type="CustomFPN",
        in_channels=[768],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    depth_head=dict(type="ComplexDepth", use_dcn=False, aspp_mid_channels=96),
    pts_bbox_head=dict(
        type="UVTRDNHead",
        view_cfg=dict(
            type="Uni3DVoxelPoolDepth",
            pc_range=point_cloud_range,
            voxel_size=unified_voxel_size,
            voxel_shape=unified_voxel_shape,
            frustum_range=frustum_range,
            frustum_size=frustum_size,
            num_convs=3,
            kernel_size=(3, 3, 3),
            embed_dim=256,
            keep_sweep_dim=True,
            fp16_enabled=False,
            loss_cfg=dict(close_radius=3.0, depth_loss_weights=[1.0]),
        ),
        # transformer_cfg
        in_channels=256,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        code_size=10,
        bg_cls_weight=0,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        fp16_enabled=False,
        transformer=dict(
            type="Uni3DTransformer",
            fp16_enabled=False,
            decoder=dict(
                type="UniTransformerDecoderV2",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="UniCrossAttenV2", embed_dims=256, fp16_enabled=False
                        ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=256,
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
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
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
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            )
        )
    ),
)

dataset_type = "NuScenesSweepDataset"
data_root = "data/nuscenes/"

file_client_args = dict(
    backend="petrel",
    path_mapping=dict(
        {
            "../data/nuscenes/": "s3://yanghonghui/nuscenes/",
            "data/nuscenes/": "s3://yanghonghui/nuscenes/",
        }
    ),
)

# file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=lidar_sweep_num - 1,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        file_client_args=file_client_args,
    ),
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
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CollectUnified3D", keys=["gt_bboxes_3d", "gt_labels_3d", "points", "img"]
    ),
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
        + "nuscenes_unified_infos_trainval.pkl",  # please change to your own info file
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d="LiDAR",
        load_interval=1,
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
        ann_file=data_root + "nuscenes_unified_infos_test.pkl",
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

evaluation = dict(interval=24, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=2, interval=1)

find_unused_parameters = True
runner = dict(type="EpochBasedRunner", max_epochs=24)
load_from = "work_dirs/uvtr_cam_vs0.075_pretrain/epoch_12.pth"
resume_from = None
# fp16 setting
fp16 = dict(loss_scale=32.0)
