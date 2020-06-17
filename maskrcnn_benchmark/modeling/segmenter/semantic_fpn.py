"""
Semantic FPN
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..semantic.segmentation import build_semantic_segmentation

import pdb

class SemanticFPN(nn.Module):
    """
    Main class for Semantic FPN. Takes a semantic target.
    It consists of three main parts:
    - backbone
    """

    def __init__(self, cfg):
        super(SemanticFPN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.semantic_segmentation = build_semantic_segmentation(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[SemanticSegmentation]): ground-truth segmentation in the image (optional)

        Returns:
            result (list[Tensor] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[Tensor] contains the segmentation image.

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # usually, it seems this is already an ImageList.
        images = to_image_list(images)

        # note that these are already run through FPN if FPN is included.
        features = self.backbone(images.tensors)

        semantic_targets = [f.segmentation for f in targets]
        mask, semantic_losses = self.semantic_segmentation(images, features, semantic_targets)

        if self.training:
            losses = {}
            losses.update(semantic_losses)
            return losses

        return mask
