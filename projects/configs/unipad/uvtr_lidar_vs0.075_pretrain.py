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
unified_voxel_shape = [
    int((point_cloud_range[3] - point_cloud_range[0]) / unified_voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / unified_voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / unified_voxel_size[2]),
]
cam_sweep_num = 1
lidar_sweep_num = 10
fp16_enabled = True
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
    type="UVTRSSL",
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
        mae_cfg=dict(downsample_scale=8, mask_ratio=0.8, learnable=False),
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
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="RenderHead",
        fp16_enabled=False,
        in_channels=256,
        unified_voxel_size=unified_voxel_size,
        unified_voxel_shape=unified_voxel_shape,
        pc_range=point_cloud_range,
        view_cfg=None,
        render_conv_cfg=dict(out_channels=32, kernel_size=3, padding=1),
        ray_sampler_cfg=dict(
            close_radius=3.0,
            only_img_mask=False,
            only_point_mask=False,
            replace_sample=False,
            point_nsample=512,
            point_ratio=0.5,
            pixel_interval=4,
            sky_region=0.4,
            merged_nsample=512,
        ),
        render_ssl_cfg=dict(
            type="NeuSModel",
            norm_scene=True,
            field_cfg=dict(
                type="SDFField",
                sdf_decoder_cfg=dict(
                    in_dim=32, out_dim=16 + 1, hidden_size=16, n_blocks=5
                ),
                rgb_decoder_cfg=dict(
                    in_dim=32 + 16 + 3 + 3, out_dim=3, hidden_size=16, n_blocks=3
                ),
                interpolate_cfg=dict(type="SmoothSampler", padding_mode="zeros"),
                beta_init=0.3,
            ),
            collider_cfg=dict(type="AABBBoxCollider", near_plane=1.0),
            sampler_cfg=dict(
                type="NeuSSampler",
                initial_sampler="UniformSampler",
                num_samples=72,
                num_samples_importance=24,
                num_upsample_steps=1,
                train_stratified=True,
                single_jitter=True,
            ),
            loss_cfg=dict(
                sensor_depth_truncation=0.2,
                sparse_points_sdf_supervised=False,
                weights=dict(
                    depth_loss=10.0,
                    rgb_loss=0.0,
                ),
            ),
        ),
    ),
)

dataset_type = "NuScenesSweepDataset"
data_root = "data/nuscenes/"

file_client_args = dict(backend="disk")

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
    dict(
        type="UnifiedRotScaleTransFlip",
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["points", "img"]),
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
    samples_per_gpu=4,
    workers_per_gpu=8,
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
        filter_empty_gt=False,
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
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
    ),
)  # please change to your own info file

evaluation = dict(interval=4, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)

optimizer = dict(type="AdamW", lr=2e-5, weight_decay=0.01)
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
runner = dict(type="EpochBasedRunner", max_epochs=12)

find_unused_parameters = True
# fp16 setting
fp16 = dict(loss_scale=32.0)
resume_from = None
