# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .panoptic_fpn import PanopticFPN


# we could probably get away with re-using GeneralizedRCNN for panoptic at some point.
_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    "PanopticFPN": PanopticFPN
}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
