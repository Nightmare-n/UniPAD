# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv
import numpy as np
import os

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet.datasets import PIPELINES
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines.dbsampler import BatchSampler
import cv2


@OBJECTSAMPLERS.register_module()
class UnifiedDataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 file_client_args=dict(backend='disk'),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3])):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    def _load_image(self, img_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            img_bytes = self.file_client.get(img_filename)
            image = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except ConnectionError:
            image = mmcv.imread(img_filename, 'unchanged')
        return image

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_bboxes, gt_labels, with_img=False):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []

        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]
            s_points_list = []
            s_idx_list = []
            s_imgs_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)
                if 'nori_id' in info:
                    results['pts_nori_path'] = info['nori_id']
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])
                idx_points = count * np.ones(len(s_points), dtype=np.int)
                s_points_list.append(s_points)
                s_idx_list.append(idx_points)
                count += 1

                if with_img:
                    if len(info['image_path']) > 0:
                        img_path = os.path.join(
                            self.data_root,
                            info['image_path']) if self.data_root else info['image_path']
                        s_img = self._load_image(img_path)
                    else:
                        s_img = []
                    s_imgs_list.append(s_img)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)
            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                "points_idx": 
                np.concatenate(s_idx_list, axis=0),
                'images':
                s_imgs_list,
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
