"""
Panoptic FPN
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list, ImageList
from maskrcnn_benchmark.structures.panoptic import PanopticTarget

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..semantic.segmentation import build_semantic_segmentation

import pdb

class PanopticFPN(nn.Module):
    """
    Main class for Panoptic FPN. Takes a panoptic target.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        _segmentations now_ and then detections / masks from it.
    """

    def __init__(self, cfg):
        super(PanopticFPN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.semantic_segmentation = build_semantic_segmentation(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

        self.prime_rpn = cfg.MODEL.RPN.USE_SEMANTIC_FEATURES
        self.prime_roi = cfg.MODEL.ROI_HEADS.USE_SEMANTIC_FEATURES
        self.feed_ground_truth_instances = cfg.TEST.FEED_GROUND_TRUTH_INSTANCES

        if self.prime_rpn:
            print("priming RPN")

        if self.prime_roi:
            print("priming ROI heads")

        if self.feed_ground_truth_instances:
            print("feeding ground truth instances")

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[PanopticTarget]): ground-truth boxes/segmentation in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
        else:
            if self.feed_ground_truth_instances:
                if targets is None:
                    raise ValueError("expected to feed ground truth instances but no ground truth provided")

                # remove images/targets without any instances.
                keep_indexes = [i for i in range(images.tensors.shape[0]) if len(targets[i].region) > 0]
                image_tensors = images.tensors[keep_indexes]
                
                image_sizes = [images.image_sizes[i] for i in range(images.tensors.shape[0]) if len(targets[i].region) > 0]
                images = ImageList(image_tensors, image_sizes)

                targets = [target for target in targets if len(target.region) > 0]

        # usually, it seems this is already an ImageList.
        images = to_image_list(images)

        # note that these are already run through FPN if FPN is included.
        features = self.backbone(images.tensors)

        semantic_targets = None
        box_targets = None

        if not (targets is None):
            semantic_targets = [f.segmentation for f in targets]
            box_targets = [f.region for f in targets]

        semantic_masks, semantic_losses, semantic_features_per_scale = self.semantic_segmentation(images, features, semantic_targets)

        primed_features = features
        # # combine these if we're asked to.
        # if self.prime_rpn or self.prime_roi:
        #     # features is shallow to deep, semantic is deep to shallow.            
        #     # todo, make this computed.
        #     FPN_FEATURES_START = 2
        #     primed_features = []
        #     for other in features[FPN_FEATURES_START + 1:]:
        #         primed_features.append(other)
            
        #     for i, feature_per_scale in enumerate(semantic_features_per_scale):
        #         primed_feature_index = FPN_FEATURES_START - i
        #         primed_feature = features[primed_feature_index] + feature_per_scale

        #         # going backwards.
        #         primed_features.insert(0, primed_feature)

        # we already know what we want.
        if self.feed_ground_truth_instances:
            proposals = [boxes.to(images.tensors.device) for boxes in box_targets]
            proposal_losses = None
        else:
            proposals, proposal_losses = self.rpn(images, primed_features if self.prime_rpn else features, box_targets)

        if self.roi_heads:
            x, boxes, detector_losses = self.roi_heads(primed_features if self.prime_roi else features, proposals, box_targets, semantic_targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            boxes = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(semantic_losses)
            losses.update(proposal_losses)
            return losses

        for i, semantic_mask in enumerate(semantic_masks):
            box = boxes[i]
            mask = semantic_mask.mask

            # the boxes record the original shape (I hope).                
            given_width, given_height = box.size
            padded_shape = mask.shape

            extra_bottom = padded_shape[0] - given_height
            if extra_bottom > 0:
                mask = mask[:-extra_bottom, :]

            extra_right = padded_shape[1] - given_width
            if extra_right > 0:
                mask = mask[:, :-extra_right]

            semantic_mask.mask = mask
        
        result = [PanopticTarget(box, mask) for (box, mask) in zip(boxes, semantic_masks)]        
        return result
