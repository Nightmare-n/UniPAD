import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet3d.core.bbox.iou_calculators import BboxOverlaps3D


@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class BBox3DIoUCost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., coordinate='lidar'):
        self.weight = weight
        self.iou_calculator = BboxOverlaps3D(coordinate=coordinate)

    def __call__(self, bbox_pred, gt_bboxes):
        iou_cost = -self.iou_calculator(bbox_pred, gt_bboxes)
        return iou_cost * self.weight
