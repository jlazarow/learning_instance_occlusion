import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3, make_dfconv3x3, conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.structures.panoptic import SemanticSegmentation

from .loss import make_semantic_loss_evaluator

import numpy as np
import pdb


# these could probably be integrated into a "number_convs" parameter in the future.
@registry.UPSAMPLE_MODULES.register("One3x3ReLU")
class OneConvUpsampleStage(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, mode="bilinear"):
        super(OneConvUpsampleStage, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.conv = make_conv3x3(
            in_channels, out_channels, use_gn=cfg.MODEL.SEMANTIC.USE_GN, use_relu=True)

    def forward(self, x):
        x = self.conv(x)

        return F.interpolate(x, scale_factor=2, mode=self.mode)

@registry.UPSAMPLE_MODULES.register("StraightDeconv")
class StraightDeconv(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, factor, mode="bilinear"):
        super(StraightDeconv, self).__init__()

        self.cfg = cfg
        self.factor = factor
        self.mode = mode
        
        self.dconv1 = make_dfconv3x3(in_channels, out_channels, use_gn=False, use_relu=True)
        self.dconv2 = make_dfconv3x3(out_channels, out_channels, use_gn=False, use_relu=True)

    def forward(self, x):
        x = self.dconv1(x)
        x = self.dconv2(x)
        
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode, align_corners=False)
    
@registry.UPSAMPLE_MODULES.register("Two3x3ReLU")
class TwoConvUpsampleStage(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, mode="bilinear"):
        super(TwoConvUpsampleStage, self).__init__()

        self.mode = mode
        self.conv1 = make_conv3x3(
            in_channels, out_channels, use_gn=cfg.MODEL.SEMANTIC.USE_GN, use_relu=True)
        self.conv2 = make_conv3x3(
            in_channels, out_channels, use_gn=cfg.MODEL.SEMANTIC.USE_GN, use_relu=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)
    
class SetOfUpsamplingStages(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, count, mode="bilinear"):
        super(SetOfUpsamplingStages, self).__init__()

        priming = cfg.MODEL.RPN.USE_SEMANTIC_FEATURES or cfg.MODEL.ROI_HEADS.USE_SEMANTIC_FEATURES
        upsample_module = registry.UPSAMPLE_MODULES[cfg.MODEL.SEMANTIC.UPSAMPLE_MODULE]

        self.stages = []
        for i in range(count):
            stage_name = "upsample_{0}".format(i)

            # currently, only consider in_channels at the first upsampling.
            stage = upsample_module(cfg, in_channels if i == 0 else out_channels, out_channels, mode=mode)
            self.stages.append(stage)

            if priming:
                self.add_module(stage_name, stage)

        # optimize this if we only want the last.
        if not priming:
            self.stages = nn.Sequential(*self.stages)

    def forward(self, x):
        result = None

        if isinstance(self.stages, list):
            result = []

            current = x
            for stage in self.stages:
                current = stage(current)
                result.append(current)
        else:
            result = self.stages(x)

        return result

class StraightUpsamplingStages(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, count, mode="bilinear"):
        super(StraightUpsamplingStages, self).__init__()

        upsample_module = registry.UPSAMPLE_MODULES[cfg.MODEL.SEMANTIC.UPSAMPLE_MODULE]

        self.stage = upsample_module(cfg, in_channels, out_channels, 2 ** count, mode=mode) 

    def forward(self, x):
        result = self.stage(x)

        return result
    
# panoptic FPN paper references an older implementation that concats instead of sums. TODO.
class FPNBasedSemanticSegmentationHead(nn.Module):
    def __init__(self, cfg, in_channels_list, in_channels_scale, out_channels, one_by_one_in_channels, mode="bilinear"):
        super(FPNBasedSemanticSegmentationHead, self).__init__()
 
        self.mode = mode
        self.upsampling_blocks = []
        self.number_upsamples_per = []

        priming = cfg.MODEL.RPN.USE_SEMANTIC_FEATURES or cfg.MODEL.ROI_HEADS.USE_SEMANTIC_FEATURES        
        # skip the possible "top" features?
        target_scale = cfg.MODEL.SEMANTIC.COMBINE_AT_SCALE
        
        for idx, in_channels in enumerate(in_channels_list):
            upsampler_name = "upsample_scale{0}".format(idx)
            in_channels = in_channels_list[idx]
            scale = in_channels_scale[idx]
            
            number_upsamples = int(np.log2(target_scale / scale))
            self.number_upsamples_per.append(number_upsamples)
            if number_upsamples == 0:
                # paper is not quite clear what happens here. my guess is the usual but no upsampling.
                upsampler = make_conv3x3(
                    in_channels, out_channels, use_gn=cfg.MODEL.SEMANTIC.USE_GN, use_relu=True)
                # upsample1 = make_dfconv3x3(
                #     in_channels, out_channels, use_gn=False, use_relu=True)
                # upsample2 = make_dfconv3x3(
                #     out_channels, out_channels, use_gn=False, use_relu=True)
                # upsampler = nn.Sequential(*[upsample1, upsample2])
            else:
                upsampler = SetOfUpsamplingStages(cfg, in_channels, out_channels, count=number_upsamples, mode=self.mode)
                #upsampler = StraightUpsamplingStages(cfg, in_channels, out_channels, count=number_upsamples, mode=self.mode)

            self.add_module(upsampler_name, upsampler)
            self.upsampling_blocks.append(upsampler)
                    
        if not cfg.MODEL.RPN.USE_SEMANTIC_FEATURES:
            # unsure if there should be a ReLU here.
            make_1x1_conv = conv_with_kaiming_uniform(use_gn=False, use_relu=True)
            self.conv = make_1x1_conv(
                one_by_one_in_channels, out_channels, kernel_size=1, stride=1)

        make_project = conv_with_kaiming_uniform(use_gn=False, use_relu=False)
        # add VOID + THING vs VOID + THINGS + STUFF
        number_classes = (1 + cfg.MODEL.SEMANTIC.NUM_CLASSES + 1 if cfg.MODEL.SEMANTIC.COLLAPSE_THING_ONTOLOGY
                          else cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES + cfg.MODEL.SEMANTIC.NUM_CLASSES)
        self.project = make_project(
            out_channels, number_classes, kernel_size=1, stride=1)

@registry.SEMANTIC_HEADS.register("UpsampleConvSumCombineScales")
class CombineScalesFPNBasedSemanticSegmentationHead(FPNBasedSemanticSegmentationHead):
    # note that out_channels is expected to be uniform across the scales.
    def __init__(self, cfg, in_channels_list, in_channels_scale, out_channels, mode="bilinear"):
        super(CombineScalesFPNBasedSemanticSegmentationHead, self).__init__(
            cfg, in_channels_list, in_channels_scale, out_channels, out_channels, mode=mode)

        priming = cfg.MODEL.RPN.USE_SEMANTIC_FEATURES or cfg.MODEL.ROI_HEADS.USE_SEMANTIC_FEATURES
        if not priming:
            raise ValueError("not supported without USE_SEMANTIC_FEATURES")

        make_1x1_conv = conv_with_kaiming_uniform(use_gn=False, use_relu=True)
        self.conv = make_1x1_conv(
            out_channels, out_channels, kernel_size=1, stride=1)

        self.convs = []
        for index in range(len(self.upsampling_blocks) - 1):
            # for each scale (except the first), add a conv to "combine".
            conv = make_1x1_conv(
                out_channels, 256, kernel_size=1, stride=1)
            self.convs.append(conv)

            combine_name = "combine{0}".format(index)
            self.add_module(combine_name, conv)

    def forward(self, features):
        # this should ignore the "top blocks" which I don't think are used.
        max_upsamples = max(self.number_upsamples_per)
        results = max_upsamples * [None]

        # the "deepest" map will play in every map.
        # the "second" deepest will play in all but the first.
        start = 0

        # note "features" is shallow to deep. ignore the last block.
        for idx, upsampler in enumerate(self.upsampling_blocks[::-1]):
            features_at_idx = features[len(features) - idx - 1 - 1]

            # this is now a list.
            upsampled_features = upsampler(features_at_idx)

            # last level gets no love (usually).
            if not isinstance(upsampled_features, list):
                results[-1] = results[-1] + upsampled_features
                break

            for i, upsampled_feature in enumerate(upsampled_features):
                target_index = start + i
                result = results[target_index]

                if result is None:
                    results[target_index] = upsampled_feature
                else:
                    results[target_index] = result + upsampled_feature

            start += 1

        # 1x1 conv after summation.
        last_result = self.conv(results[-1])
        for i, result in enumerate(results):
            results[i] = self.convs[i](results[i])

        # now bilinearly upsample 4x and project (only the last one)
        mask = F.interpolate(last_result, scale_factor=4, mode=self.mode, align_corners=False)
        mask = self.project(mask)

        return mask, results
     
@registry.SEMANTIC_HEADS.register("UpsampleEqualSizeConvSum")
class SumFPNBasedSemanticSegmentationHead(FPNBasedSemanticSegmentationHead):
    # note that out_channels is expected to be uniform across the scales.
    def __init__(self, cfg, in_channels_list, in_channels_scale, out_channels, mode="bilinear"):
        super(SumFPNBasedSemanticSegmentationHead, self).__init__(
            cfg, in_channels_list, in_channels_scale, out_channels, out_channels, mode=mode)

    def forward(self, features):
        # this should ignore the "top blocks" which I don't think are used.
        result = None
        for idx, upsampler in enumerate(self.upsampling_blocks):
            features_at_idx = features[idx]
            upsampled = upsampler(features_at_idx)

            if result is None:
                result = upsampled
            else:
                result = result + upsampled

        # 1x1 conv after summation.
        result = self.conv(result)
                
        # now bilinearly upsample 4x.
        result = F.interpolate(result, scale_factor=4, mode=self.mode, align_corners=False)
        result = self.project(result)

        return result, None

@registry.SEMANTIC_HEADS.register("UpsampleEqualSizeConvConcat")
class ConcatFPNBasedSemanticSegmentationHead(FPNBasedSemanticSegmentationHead):
    # note that out_channels is expected to be uniform across the scales.
    def __init__(self, cfg, in_channels_list, in_channels_scale, out_channels, mode="bilinear"):
        super(ConcatFPNBasedSemanticSegmentationHead, self).__init__(
            cfg, in_channels_list, in_channels_scale, out_channels, len(in_channels_list) * out_channels, mode=mode)

    def forward(self, features):
        # this should ignore the "top blocks" which I don't think are used.
        results = []
        for idx, upsampler in enumerate(self.upsampling_blocks):
            features_at_idx = features[idx]
            upsampled = upsampler(features_at_idx)
            results.append(upsampled)

        # 1x1 conv after summation.
        result = torch.cat(results, dim=1)
        result = self.conv(result)
                
        # now bilinearly upsample 4x.
        result = F.interpolate(result, scale_factor=4, mode=self.mode, align_corners=False)
        result = self.project(result)

        return result, None
    
class SemanticSegmentationModule(torch.nn.Module):
    """
    Module for semantic segmentation. Takes feature maps from the backbone and
    applies them to produce a pixel-wise semgnetation.
    """

    def __init__(self, cfg):
        super(SemanticSegmentationModule, self).__init__()

        self.cfg = cfg.clone()

        # this seems to be hardcoded (see backbone.py)
        in_channels_list = [
            cfg.MODEL.BACKBONE.OUT_CHANNELS,
            cfg.MODEL.BACKBONE.OUT_CHANNELS,
            cfg.MODEL.BACKBONE.OUT_CHANNELS,
            cfg.MODEL.BACKBONE.OUT_CHANNELS
        ]

        in_channels_scale = cfg.MODEL.SEMANTIC.POOLER_SCALES
        output_channels = cfg.MODEL.SEMANTIC.CONV_HEAD_DIM
        upsample_mode = cfg.MODEL.SEMANTIC.UPSAMPLE_METHOD
        print("using upsample mode: {0}".format(upsample_mode))
        
        semantic_head = registry.SEMANTIC_HEADS[cfg.MODEL.SEMANTIC.SEMANTIC_HEAD]            
        head = semantic_head(cfg, in_channels_list, in_channels_scale, output_channels, mode=upsample_mode)
        loss_evaluator = make_semantic_loss_evaluator(cfg)

        self.head = head
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, predict_if_test=True):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[SemanticSegmentation]): ground-truth semantic segmentations (optional)

        Returns:
            mask (Tensor): the predicted masks from the model, one tensor per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        mask, features_per_scale = self.head(features)
        
        if self.training:
            return self._forward_train(mask, targets, features_per_scale)
        else:
            return self._forward_test(mask, features_per_scale, predict=predict_if_test)

    def _forward_train(self, mask, targets, features_per_scale):
        # note, the loss evaluator needs to be aware of "unlabeled" pixels.
        loss_semantic = self.loss_evaluator(mask, targets)
        losses = {
            "loss_semantic": loss_semantic
        }

        return mask, losses, features_per_scale

    def _forward_test(self, mask, features_per_scale, predict=True):
        result = []

        # pick the most probable class.
        if predict:
            mask = torch.argmax(mask, dim=1)
            
        for i in range(mask.shape[0]):
            result.append(
                SemanticSegmentation(mask[i], has_label_mask=None))

        return result, None, features_per_scale

def build_semantic_segmentation(cfg):
    return SemanticSegmentationModule(cfg)
