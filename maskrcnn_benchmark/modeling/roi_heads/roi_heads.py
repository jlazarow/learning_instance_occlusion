# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .order_head.order_head import build_roi_order_head

import numpy as np
import pdb

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

        self.feed_ground_truth_instances = cfg.TEST.FEED_GROUND_TRUTH_INSTANCES

    def forward(self, features, proposals, targets=None, semantic_targets=None):
        losses = {}
        mask_logits = None        
        
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)

        # replace the detections once again.
        if self.feed_ground_truth_instances:
            detections = proposals
        
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x

            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask, mask_logits, mask_scores = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

            if self.cfg.MODEL.ORDER_ON:
                # mask head does this in a nicer way by "sharing" the feature extractor and
                # opting not to compute it again at training time. TODO!
                order_features = features
                if self.cfg.MODEL.ROI_ORDER_HEAD.SHARE_MASK_FEATURE_EXTRACTOR:
                    order_features = x

                loss_order = self.order(order_features, detections, mask_logits, targets)
                losses.update(loss_order)

        return x, detections, losses

def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))
    if cfg.MODEL.ORDER_ON:
        roi_heads.append(("order", build_roi_order_head(cfg)))        

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
