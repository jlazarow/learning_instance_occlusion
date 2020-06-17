import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import keep_only_positive_boxes
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import project_masks_on_boxes
from maskrcnn_benchmark.modeling.poolers import Pooler

import pycocotools.mask as mask_util

from .roi_order_feature_extractors import make_roi_order_feature_extractor
from .roi_order_predictors import make_roi_order_predictor
from .loss import prepare_mask_intersection_matrix, compute_overlap_matrix

import pdb

class ROIOrderHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIOrderHead, self).__init__()
        self.cfg = cfg.clone()

        self.pooler = None
        if not self.cfg.MODEL.ROI_ORDER_HEAD.SHARE_MASK_FEATURE_EXTRACTOR:
            resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
            sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
            pooler = Pooler(
                output_size=(resolution, resolution),
                scales=scales,
                sampling_ratio=sampling_ratio,
            )

            input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
            self.pooler = pooler

        self.feature_extractor = make_roi_order_feature_extractor(cfg)
        self.predictor = make_roi_order_predictor(cfg)

        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False)
        
        mask_threshold = self.cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        self.masker = Masker(threshold=mask_threshold, padding=1)

        self.intersect_threshold = self.cfg.MODEL.ROI_ORDER_HEAD.OVERLAP_THRESHOLD
        print("order head: using intersection threshold of {0}".format(self.intersect_threshold))
        
        self.maximum_per_image = self.cfg.MODEL.ROI_ORDER_HEAD.BATCH_SIZE_PER_IMAGE
        self.ensure_consistency = self.cfg.MODEL.ROI_ORDER_HEAD.ENSURE_CONSISTENCY
        self.overlap_validation = self.cfg.TEST.ORDER_ONLY
        self.mask_resolution = self.cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        self.intraclass_occlusion = self.cfg.TEST.INTRACLASS_OCCLUSION

    def forward(self, x, boxes_per_image, mask_logits, targets=None):
        """
        Arguments:
            x: features? should these be shared.
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """        
        if self.training:
            boxes_per_image, positive_inds = keep_only_positive_boxes(boxes_per_image)

            if self.pooler:
                x = self.pooler(x, boxes_per_image)                
        else:
            if self.pooler:
                x = self.pooler(x, boxes_per_image)                
            
            if not (targets is None):
                if not self.overlap_validation:
                    raise ValueError("don't use a validation set with overlaps unless this is \"order only\"")

                number_boxes_per_image = [len(boxes) for boxes in boxes_per_image]
                features_per_image = x.split(number_boxes_per_image, dim=0)

                errors = []
                for target, features in zip(targets, features_per_image):
                    target = target.to(x.device)
                    overlaps = target.get_field("overlaps")
                    overlap_pairs = torch.nonzero(overlaps >= 0)

                    if overlap_pairs.numel() == 0:
                        continue

                    first_idxs = overlap_pairs[:, 0]
                    second_idxs = overlap_pairs[:, 1]

                    masks = target.get_field("masks").to(x.device)

                    # we need to get the 28x28 masks.                    
                    compatible_masks = project_masks_on_boxes(
                        masks, target, self.mask_resolution)
                    paired_masks = compatible_masks[overlap_pairs]
                    subsampled_masks = torch.unsqueeze(F.max_pool2d(paired_masks, kernel_size=2, stride=2), dim=2)
                    paired_features = features[overlap_pairs[:]]
                    
                    paired_combined = torch.cat([subsampled_masks, paired_features], dim=2)

                    combined_shape = (paired_combined.shape[0], 2 * (256 + 1)) + tuple(paired_combined.shape[3:])
                    paired_combined = torch.reshape(paired_combined, combined_shape)

                    extracted_features = self.feature_extractor(paired_combined)
                    overlap_predicted = torch.squeeze(self.predictor(extracted_features), dim=1) > 0.5
                    overlap_expected = overlaps[first_idxs, second_idxs].byte()

                    number_overlaps = float(overlap_pairs.shape[0])
                    count_correct = torch.nonzero(overlap_predicted == overlap_expected).numel()

                    errors.append(count_correct / number_overlaps)

                if errors:
                    return np.mean(errors)

                return None
            else:
                targets = len(boxes_per_image) * [None]

        number_boxes_per_image = [len(boxes) for boxes in boxes_per_image]
        masks_per_image = mask_logits.split(number_boxes_per_image, dim=0)
        features_per_image = x.split(number_boxes_per_image, dim=0)

        # note this needs to work differently at inference time..
        aggregate_combined = []
        aggregate_overlaps = []
        for boxes, masks, features, target in zip(boxes_per_image, masks_per_image, features_per_image, targets):
            # select the masks corresponding to the ground truth label of the "positive" proposal.
            number_masks = masks.shape[0]
            labels = boxes.get_field("labels")
            index = torch.arange(number_masks, device=labels.device)

            selected_masks = masks[index, labels][:, None].sigmoid()
            
            if self.training:
                # the hard decisions here will be to decide how to "pair" these.
                # before? after?
                match_quality_matrix = boxlist_iou(target, boxes)

                # by definition, I believe these should only be "positive" matches.
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                different_matched_idx = torch.unsqueeze(matched_idxs, dim=0) != torch.unsqueeze(matched_idxs, dim=1)
            else:
                if number_masks == 0:
                    continue

                # allow an option to allow intra-class overlap.
                if self.intraclass_occlusion:
                    different_matched_idx = torch.ones((number_masks, number_masks), dtype=torch.uint8).to(labels.device)
                else:
                    different_matched_idx = torch.unsqueeze(labels, dim=0) != torch.unsqueeze(labels, dim=1)
                # we should also probably kill masks that aren't > 0.5 confidence here (or whatever the threshold is).

            # we don't necessarily need to intersect _all_, but across matched indexes.
            # take hard masks and encode.
            hard_masks = self.masker([selected_masks], boxes)[0].cpu()
            mask_intersection_matrix = prepare_mask_intersection_matrix(
                boxes, hard_masks)

            # this is _not_ a symmetric matrix.
            sufficient_intersection_matrix = mask_intersection_matrix >= self.intersect_threshold
            selected_pairs = torch.nonzero(different_matched_idx & sufficient_intersection_matrix)
            if selected_pairs.numel() == 0:
                continue
            
            if self.training:
                if not target.has_field("overlaps"):
                    raise ValueError("overlaps do not exist on target")
            
                overlaps = target.get_field("overlaps")
                print("expecting {0} overlaps from ground truth".format(overlaps.shape[0]))

                # ground truth.
                first_idxs = torch.unsqueeze(matched_idxs[selected_pairs[:, 0]], dim=1)
                second_idxs = torch.unsqueeze(matched_idxs[selected_pairs[:, 1]], dim=1)
                
                selected_overlaps = overlaps[first_idxs, second_idxs]
                
                actual_overlaps = selected_overlaps >= 0            
                mask_of_overlaps = torch.nonzero(actual_overlaps)[:, 0]
                
                if mask_of_overlaps.numel() == 0:
                    continue
            
                number_masked = mask_of_overlaps.shape[0]
                print("found {0} candidate overlaps".format(number_masked))
                subsample_size = min(self.maximum_per_image, number_masked)
                subsample_perm = torch.randperm(number_masked, device=mask_of_overlaps.device)[:subsample_size]
                mask_of_overlaps = mask_of_overlaps[subsample_perm]

                # subselect the selected pairs corresponding to real overlaps.
                selected_pairs = selected_pairs[mask_of_overlaps]
                selected_overlaps = selected_overlaps[mask_of_overlaps][:, 0]

            paired_masks = selected_masks[selected_pairs][:, :, 0]

            # sigmoid or not?
            subsampled_masks = torch.unsqueeze(F.max_pool2d(paired_masks, kernel_size=2, stride=2), dim=2)
            paired_features = features[selected_pairs[:]]
            paired_combined = torch.cat([subsampled_masks, paired_features], dim=2)

            combined_shape = (paired_combined.shape[0], 2 * (256 + 1)) + tuple(paired_combined.shape[3:])
            paired_reshaped = torch.reshape(paired_combined, combined_shape)

            # we should replicate the reversed order too (todo!).
            if self.training:
                # technically we could optimize this if we flip after feature extraction.
                if self.ensure_consistency:
                    # flip the order and the expectation.
                    paired_reversed = torch.flip(paired_combined, dims=[1])
                    paired_reversed =  torch.reshape(paired_reversed, combined_shape)
                    paired_reshaped = torch.cat([paired_reshaped, paired_reversed], dim=0)

                    # negate this.
                    reversed_overlaps = 1 - selected_overlaps
                    selected_overlaps = torch.cat([selected_overlaps, reversed_overlaps], dim=0)

            # concatenate the soft-masks and collapse the pair/channel dimension.
            # sometimes _no pairs_ might be selected (check?).
            # we also probably need to subsample these.
            if self.training:
                aggregate_combined.append(paired_reshaped)
                aggregate_overlaps.append(selected_overlaps)
            else:
                # just run these separately for now for logical ease.
                extracted_features = self.feature_extractor(paired_reshaped)
                # while this _should_ be sigmoid, it appears to work better with a higher threshold (without sigmoid).
                overlap_predicted = torch.squeeze(self.predictor(extracted_features), dim=1) > 0.5

                boxes.add_field("overlap_pairs", selected_pairs)
                boxes.add_field("overlap", overlap_predicted)

        if self.training:
            if len(aggregate_overlaps) == 0:
                fake_graph = self.predictor(self.feature_extractor(torch.zeros(1, 2 * (256 + 1), x.shape[2], x.shape[3]).to(x.device)))
                return {
                    "loss_order": 0.0 * torch.mean(fake_graph)
                }

            aggregate_overlaps = torch.cat(aggregate_overlaps, dim=0)

            # no overlaps that qualify (todo, combine with previous condition).
            if aggregate_overlaps.numel() == 0:
                fake_graph = self.predictor(self.feature_extractor(torch.zeros(1, 2 * (256 + 1), x.shape[2], x.shape[3]).to(x.device)))
                return {
                    "loss_order": 0.0 * torch.mean(fake_graph)
                }
        else:
            return {}

        # testing code should never get here.
        aggregate_combined = torch.cat(aggregate_combined, dim=0)        
        
        aggregate_features = self.feature_extractor(aggregate_combined)
        aggregate_predicted = self.predictor(aggregate_features)

        aggregate_expected = aggregate_overlaps

        # compute cross entropy
        order_loss = F.binary_cross_entropy_with_logits(
            aggregate_predicted[:, 0], aggregate_expected)

        # print this out since the median is generally not helpful.
        print("now: order_loss {0}".format(order_loss.detach().cpu()))

        return {
            "loss_order": order_loss
        }

def build_roi_order_head(cfg):
    return ROIOrderHead(cfg)
