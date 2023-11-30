import numpy as np
from numpy import random
import torch
import mmcv
import cv2
import mmdet3d
from mmdet.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    box_np_ops,
)


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        results["pad_before_shape"] = [img.shape for img in results["img"]]
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class RandomResizeCropFlipMultiViewImage(object):
    def __init__(
        self,
        image_size,
        resize_scales=None,
        crop_sizes=None,
        flip_ratio=None,
        rot_angles=None,
        training=True,
    ):
        self.image_size = image_size
        self.flip_ratio = flip_ratio
        self.resize_scales = resize_scales
        self.crop_sizes = crop_sizes
        self.rot_angles = rot_angles
        self.training = training

    def _resize_img(self, results):
        img_scale_mat = []
        new_img = []
        for img in results["img"]:
            resize = float(self.image_size[1]) / float(
                img.shape[1]
            ) + np.random.uniform(*self.resize_scales)
            new_img.append(
                mmcv.imresize(
                    img,
                    (int(img.shape[1] * resize), int(img.shape[0] * resize)),
                    return_scale=False,
                )
            )
            img_scale_mat.append(
                np.array(
                    [[resize, 0, 0, 0], [0, resize, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=np.float32,
                )
            )

        results["img"] = new_img
        results["img_scale_mat"] = img_scale_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def _crop_img(self, results):
        img_crop_mat = []
        new_img = []
        for img in results["img"]:
            crop_pos = np.random.uniform(0.0, 1.0)
            # crop from image bottom
            start_h = img.shape[0] - self.crop_sizes[0]
            start_w = (
                int(crop_pos * max(0, img.shape[1] - self.crop_sizes[1]))
                if self.training
                else max(0, img.shape[1] - self.crop_sizes[1]) // 2
            )
            new_img.append(
                img[
                    start_h : start_h + self.crop_sizes[0],
                    start_w : start_w + self.crop_sizes[1],
                    ...,
                ]
            )
            img_crop_mat.append(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [-start_w, -start_h, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            )

        results["img"] = new_img
        results["img_crop_mat"] = img_crop_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def _flip_img(self, results):
        img_flip_mat = []
        new_img = []
        for img in results["img"]:
            if np.random.rand() >= self.flip_ratio or (not self.training):
                new_img.append(img)
                img_flip_mat.append(
                    np.array(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        dtype=np.float32,
                    )
                )
            else:
                new_img.append(mmcv.imflip(img, "horizontal"))
                img_flip_mat.append(
                    np.array(
                        [
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [img.shape[1] - 1, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    )
                )

        results["img"] = new_img
        results["img_flip_mat"] = img_flip_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def _rotate_img(self, results):
        new_img = []
        img_rot_mat = []
        for img in results["img"]:
            # Rotation angle in degrees
            angle = np.random.uniform(*self.rot_angles)
            new_img.append(mmcv.imrotate(img, angle))
            h, w = img.shape[:2]
            c_x, c_y = (w - 1) * 0.5, (h - 1) * 0.5
            rot_sin, rot_cos = np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)
            img_rot_mat.append(
                np.array(
                    [
                        [rot_cos, rot_sin, 0, 0],
                        [-rot_sin, rot_cos, 0, 0],
                        [
                            (1 - rot_cos) * c_x + rot_sin * c_y,
                            (1 - rot_cos) * c_y - rot_sin * c_x,
                            1,
                            0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            )

        results["img"] = new_img
        results["img_rot_mat"] = img_rot_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def __call__(self, results):
        # TODO: aug sweep-wise or camera-wise?
        # resize image
        if self.resize_scales is not None:
            results = self._resize_img(results)
        # crop image
        if self.crop_sizes is not None:
            results = self._crop_img(results)
        # flip image
        if self.flip_ratio is not None:
            results = self._flip_img(results)
        # rotate image
        if self.rot_angles is not None:
            results = self._rotate_img(results)

        img_rot_aug = []
        for i in range(len(results["img"])):
            rot_mat = np.eye(4, dtype=np.float32)
            if "img_scale_mat" in results:
                rot_mat = results["img_scale_mat"][i].T @ rot_mat
            if "img_crop_mat" in results:
                rot_mat = results["img_crop_mat"][i].T @ rot_mat
            if "img_flip_mat" in results:
                rot_mat = results["img_flip_mat"][i].T @ rot_mat
            if "img_rot_mat" in results:
                rot_mat = results["img_rot_mat"][i].T @ rot_mat
            img_rot_aug.append(rot_mat)
        results["img_rot_aug"] = img_rot_aug

        num_cam, num_sweep = len(results["lidar2img"]), len(results["lidar2img"][0])
        img_rot_aug = np.concatenate(img_rot_aug, axis=0).reshape(
            num_cam, num_sweep, 4, 4
        )
        results["lidar2img"] = [
            img_rot_aug[_idx] @ results["lidar2img"][_idx]
            for _idx in range(len(results["lidar2img"]))
        ]
        return results


@PIPELINES.register_module()
class UnifiedRotScaleTransFlip(object):
    """
    Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(
        self,
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        flip_ratio_bev_horizontal=0.0,
        flip_ratio_bev_vertical=0.0,
        shift_height=False,
    ):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.shift_height = shift_height

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if "rot_degree" in input_dict:
            noise_rotation = input_dict["rot_degree"]
        else:
            rotation = self.rot_range
            noise_rotation = np.random.uniform(rotation[0], rotation[1])
            input_dict["rot_degree"] = noise_rotation

        # calculate rotation matrix
        rot_sin = torch.sin(torch.tensor(noise_rotation))
        rot_cos = torch.cos(torch.tensor(noise_rotation))
        # align coord system with previous version
        rot_mat_T = torch.Tensor(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        input_dict["uni_rot_mat"] = rot_mat_T

        if len(input_dict["bbox3d_fields"]) == 0:  # test mode
            input_dict["bbox3d_fields"].append("empty_box3d")
            input_dict["empty_box3d"] = input_dict["box_type_3d"](
                np.array([], dtype=np.float32)
            )

        # rotate points with bboxes
        for key in input_dict["bbox3d_fields"]:
            if "points" in input_dict:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict["points"]
                )
                input_dict["points"] = points
                input_dict["pcd_rotation"] = rot_mat_T
            else:
                input_dict[key].rotate(noise_rotation)

        input_dict["transformation_3d_flow"].append("R")

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if "pcd_scale_factor" not in input_dict:
            scale_factor = np.random.uniform(
                self.scale_ratio_range[0], self.scale_ratio_range[1]
            )
            input_dict["pcd_scale_factor"] = scale_factor

        scale = input_dict["pcd_scale_factor"]
        if "points" in input_dict:
            points = input_dict["points"]
            points.scale(scale)
            if self.shift_height:
                assert (
                    "height" in points.attribute_dims.keys()
                ), "setting shift_height=True but points have no height attribute"
                points.tensor[:, points.attribute_dims["height"]] *= scale
            input_dict["points"] = points

        input_dict["uni_scale_mat"] = torch.Tensor(
            [[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]]
        )

        for key in input_dict["bbox3d_fields"]:
            input_dict[key].scale(scale)

        input_dict["transformation_3d_flow"].append("S")

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if "pcd_trans" not in input_dict:
            translation_std = np.array(self.translation_std, dtype=np.float32)
            trans_factor = np.random.normal(scale=translation_std, size=3).T
            input_dict["pcd_trans"] = trans_factor
        else:
            trans_factor = input_dict["pcd_trans"]

        input_dict["uni_trans_mat"] = torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [trans_factor[0], trans_factor[1], trans_factor[2], 1],
            ]
        )

        if "points" in input_dict:
            input_dict["points"].translate(trans_factor)

        for key in input_dict["bbox3d_fields"]:
            input_dict[key].translate(trans_factor)

        input_dict["transformation_3d_flow"].append("T")

    def _flip_bbox_points(self, input_dict):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """

        def _flip_single(input_dict, direction="horizontal"):
            assert direction in ["horizontal", "vertical"]
            if len(input_dict["bbox3d_fields"]) == 0:  # test mode
                input_dict["bbox3d_fields"].append("empty_box3d")
                input_dict["empty_box3d"] = input_dict["box_type_3d"](
                    np.array([], dtype=np.float32)
                )
            assert len(input_dict["bbox3d_fields"]) == 1
            for key in input_dict["bbox3d_fields"]:
                if "points" in input_dict:
                    input_dict["points"] = input_dict[key].flip(
                        direction, points=input_dict["points"]
                    )
                else:
                    input_dict[key].flip(direction)

        if "pcd_horizontal_flip" not in input_dict:
            flip_horizontal = (
                True if np.random.rand() < self.flip_ratio_bev_horizontal else False
            )
            input_dict["pcd_horizontal_flip"] = flip_horizontal
        if "pcd_vertical_flip" not in input_dict:
            flip_vertical = (
                True if np.random.rand() < self.flip_ratio_bev_vertical else False
            )
            input_dict["pcd_vertical_flip"] = flip_vertical

        # flips the y (horizontal) or x (vertical) axis
        flip_mat = torch.Tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        if input_dict["pcd_horizontal_flip"]:
            _flip_single(input_dict, "horizontal")
            input_dict["transformation_3d_flow"].append("HF")
            flip_mat[1, 1] *= -1
        if input_dict["pcd_vertical_flip"]:
            _flip_single(input_dict, "vertical")
            input_dict["transformation_3d_flow"].append("VF")
            flip_mat[0, 0] *= -1

        input_dict["uni_flip_mat"] = flip_mat

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if "transformation_3d_flow" not in input_dict:
            input_dict["transformation_3d_flow"] = []

        self._rot_bbox_points(input_dict)

        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        self._flip_bbox_points(input_dict)

        # unified augmentation for point and voxel
        uni_rot_aug = (
            input_dict["uni_flip_mat"].T
            @ input_dict["uni_trans_mat"].T
            @ input_dict["uni_scale_mat"].T
            @ input_dict["uni_rot_mat"].T
        )
        input_dict["uni_rot_aug"] = uni_rot_aug
        if "lidar2img" in input_dict:
            input_dict["lidar2img"] = [
                input_dict["lidar2img"][_idx] @ np.linalg.inv(uni_rot_aug)
                for _idx in range(len(input_dict["lidar2img"]))
            ]
        if "lidar2cam" in input_dict:
            input_dict["lidar2cam"] = [
                input_dict["lidar2cam"][_idx] @ np.linalg.inv(uni_rot_aug)
                for _idx in range(len(input_dict["lidar2cam"]))
            ]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(rot_range={self.rot_range},"
        repr_str += f" scale_ratio_range={self.scale_ratio_range},"
        repr_str += f" translation_std={self.translation_std},"
        repr_str += f" flip_ratio_bev_horizontal={self.flip_ratio_bev_horizontal},"
        repr_str += f" flip_ratio_bev_vertical={self.flip_ratio_bev_vertical},"
        repr_str += f" shift_height={self.shift_height})"
        return repr_str


@PIPELINES.register_module()
class UnifiedObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(
        self, db_sampler, sample_2d=False, sample_method="depth", modify_points=False
    ):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.modify_points = modify_points
        if "type" not in db_sampler.keys():
            db_sampler["type"] = "DataBaseSampler"
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]

        # change to float for blending operation
        points = input_dict["points"]
        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=True
            )
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=False
            )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points_idx = -1 * np.ones(len(points), dtype=np.int)
            # check the points dimension
            # points = points.cat([sampled_points, points])
            points = points.cat([points, sampled_points])
            points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            if self.sample_2d:
                imgs = input_dict["img"]
                lidar2img = input_dict["lidar2img"]
                sampled_img = sampled_dict["images"]
                sampled_num = len(sampled_gt_bboxes_3d)
                imgs, points_keep = self.unified_sample(
                    imgs,
                    lidar2img,
                    points.tensor.numpy(),
                    points_idx,
                    gt_bboxes_3d.corners.numpy(),
                    sampled_img,
                    sampled_num,
                )

                input_dict["img"] = imgs

                if self.modify_points:
                    points = points[points_keep]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d.astype(np.long)
        input_dict["points"] = points

        return input_dict

    def unified_sample(
        self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num
    ):
        # for boxes
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw) - sampled_num
        # for point cloud
        points_3d = points[:, :4].copy()
        points_3d[:, -1] = 1
        points_keep = np.ones(len(points_3d)).astype(np.bool)
        new_imgs = imgs

        assert len(imgs) == len(lidar2img) and len(sampled_img) == sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            assert len(_lidar2img) == 1, "only support sweep == 1"
            _lidar2img = _lidar2img[0]  # (4, 4)
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[..., :2] /= coord_img[..., 2:3]
            depth = coord_img[..., 2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[..., :2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:, 0::2] = np.clip(bbox[:, 0::2], a_min=0, a_max=_img.shape[1] - 1)
            bbox[:, 1::2] = np.clip(bbox[:, 1::2], a_min=0, a_max=_img.shape[0] - 1)
            img_mask = ((bbox[:, 2:] - bbox[:, :2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if "depth" in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1] : _box[3], _box[0] : _box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    _img[_box[1] : _box[3], _box[0] : _box[2]] = raw_img.pop(0)
                    fg_mask[_box[1] : _box[3], _box[0] : _box[2]] = 1
                else:
                    img_crop = sampled_img[_count - raw_num]
                    if len(img_crop) == 0:
                        continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2, 3]] - _box[[0, 1]]))
                    _img[_box[1] : _box[3], _box[0] : _box[2]] = img_crop

                paste_mask[_box[1] : _box[3], _box[0] : _box[2]] = _count

            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points:
                points_img = points_3d @ _lidar2img.T
                points_img[:, :2] /= points_img[:, 2:3]
                depth = points_img[:, 2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (
                    (points_img[:, 0] > 0)
                    & (points_img[:, 0] < _img.shape[1])
                    & (points_img[:, 1] > 0)
                    & (points_img[:, 1] < _img.shape[0])
                    & img_mask
                )
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:, 1], points_img[:, 0]] == (
                    points_idx[img_mask] + raw_num
                )
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = (
                    raw_fg[points_img[:, 1], points_img[:, 0]]
                    | raw_bg[points_img[:, 1], points_img[:, 0]]
                )
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f" sample_2d={self.sample_2d},"
        repr_str += f" data_root={self.sampler_cfg.data_root},"
        repr_str += f" info_path={self.sampler_cfg.info_path},"
        repr_str += f" rate={self.sampler_cfg.rate},"
        repr_str += f" prepare={self.sampler_cfg.prepare},"
        repr_str += f" classes={self.sampler_cfg.classes},"
        repr_str += f" sample_groups={self.sampler_cfg.sample_groups}"
        return repr_str


@PIPELINES.register_module()
class NormalizeIntensity(object):
    """
    Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self, mean=[127.5], std=[127.5], use_dim=[3]):
        self.mean = mean
        self.std = std
        self.use_dim = use_dim

    def __call__(self, input_dict):
        points = input_dict["points"]
        # overwrite
        mean = points.tensor.new_tensor(self.mean)
        std = points.tensor.new_tensor(self.std)
        points.tensor[:, self.use_dim] = (points.tensor[:, self.use_dim] - mean) / std
        input_dict["points"] = points
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f" mean={self.mean},"
        repr_str += f" std={self.std},"
        repr_str += f" use_dim={self.use_dim}"
        return repr_str
