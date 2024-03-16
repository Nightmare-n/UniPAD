_base_ = [
    "../../../../configs/_base_/datasets/nus-3d.py",
    "../../../../configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
pts_voxel_size = [0.075, 0.075, 0.2]
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
    use_lidar=True,
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
            checkpoint="data/ckpts/convnext-small_3rdparty_32xb128-noema_in1k_processed.pth",
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
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=pts_voxel_size,
        max_voxels=(120000, 160000),
        deterministic=False,
    ),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=5),
    pts_middle_encoder=dict(
        type="MaskSparseEncoderHD",
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=256,
        order=("conv", "norm", "act"),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type="basicblock",
        fp16_enabled=False,
    ),  # not enable FP16 here
    pts_backbone=dict(
        type="SECOND3D",
        in_channels=[256, 256, 256],
        out_channels=[128, 256, 512],
        layer_nums=[5, 5, 5],
        layer_strides=[1, 2, 4],
        is_cascade=False,
        norm_cfg=dict(type="BN3d", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv3d", kernel=(1, 3, 3), bias=False),
    ),
    pts_neck=dict(
        type="SECOND3DFPN",
        in_channels=[128, 256, 512],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN3d", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv3d", bias=False),
        extra_conv=dict(type="Conv3d", num_conv=3, bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="UVTRDNHead",
        unified_conv=dict(type="Conv3d", num_conv=1, fusion="sum"),
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
            type="NMSFreeIoUCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10,
            nms_cfg=[
                dict(
                    class_names=[
                        "car",
                        "truck",
                        "construction_vehicle",
                        "bus",
                        "trailer",
                        "bicycle",
                        "pedestrian",
                    ],
                    indices=[0, 1, 2, 3, 4, 7, 8],
                    nms_threshold=-1,
                ),
                dict(
                    class_names=[
                        "barrier",
                    ],
                    indices=[5],
                    nms_threshold=0.2,
                ),
                dict(
                    class_names=[
                        "motorcycle",
                    ],
                    indices=[6],
                    nms_threshold=0.2,
                ),
                dict(class_names=["traffic_cone"], indices=[9], nms_threshold=0.2),
            ],
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

db_sampler = dict(
    type="UnifiedDataBaseSampler",
    data_root=data_root,
    info_path=data_root
    + "nuscenes_unified_dbinfos_train.pkl",  # please change to your own database file
    rate=1.0,
    file_client_args=file_client_args,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5,
        ),
    ),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2,
    ),
    points_loader=dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
    ),
)

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
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        translation_std=[0.5, 0.5, 0.5],
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CollectUnified3D", keys=["gt_bboxes_3d", "gt_labels_3d", "points", "img"]
    ),
]
test_pipeline = [
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
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["points", "img"]),
]


data = dict(
    samples_per_gpu=2,
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
        load_interval=5,
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
    ),
)

optimizer = dict(
    type="AdamW",
    lr=4e-5,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "img_neck": dict(lr_mult=0.1),
            "view_trans": dict(lr_mult=0.1),
            "input_proj": dict(lr_mult=0.1),
            "depth_net": dict(lr_mult=0.1),
            "pts_middle_encoder": dict(lr_mult=0.1),
            "pts_backbone": dict(lr_mult=0.1),
            "pts_neck": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="cyclic",
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy="cyclic",
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=12)

evaluation = dict(interval=12, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
load_from = "data/ckpts/merged_abl_uvtr_fusion_vs0.075_scratch.pth"  # the weights are got by merging the weights of uvtr_lidar and uvtr_cam
resume_from = None
find_unused_parameters = True
# fp16 setting
fp16 = dict(loss_scale=32.0)
