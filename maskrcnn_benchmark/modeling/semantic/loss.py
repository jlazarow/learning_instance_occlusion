"""
This file contains specific functions for computing losses from
semantic segmentation
"""

import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.utils import cat
import pdb

class SemanticLossComputation(object):
    """
    This class computes the semantic segmentation loss.
    """

    def __init__(self):
        pass
    
    def __call__(self, class_logits, targets):
        """
        Arguments:
            logits (list[Tensor])
            targets (list[SemanticSegmentation])

        Returns:
            semantic_loss (Tensor)
        """
        
        # it appears that "class_logits" might be right/bottom zero padding. take
        # care of any differences there and add them as not having a label on
        # a per image basis.
        labels = [target.mask for target in targets]
        has_label_mask = [target.has_label_mask for target in targets]

        # todo, easier way?
        for i in range(len(targets)):
            given_shape = targets[i].mask.shape
            padded_shape = class_logits[i].shape[1:]
            extra_bottom = padded_shape[0] - given_shape[0]
            extra_right = padded_shape[1] - given_shape[1]

            labels[i] = torch.unsqueeze(F.pad(labels[i], (0, extra_right, 0, extra_bottom), value=0), dim=0)
            has_label_mask[i] = torch.unsqueeze(F.pad(has_label_mask[i], (0, extra_right, 0, extra_bottom), value=0), dim=0)

        labels = cat(labels, dim=0).long()
        has_label_mask = cat(has_label_mask, dim=0).float()

        # this might be interesting to play with when the ontology is dense.
        # e.g. do we penalize getting a "stuff" class when a "thing" is present in a different
        # manner to getting a _wrong_ "thing" class?
        classification_loss = F.cross_entropy(class_logits, labels, reduction="none")

        # multiply by the "has label" mask
        masked_classification_loss = has_label_mask * classification_loss
        mean_classification_loss = torch.mean(masked_classification_loss)

        return mean_classification_loss

def make_semantic_loss_evaluator(cfg):
    loss_evaluator = SemanticLossComputation()
    return loss_evaluator
