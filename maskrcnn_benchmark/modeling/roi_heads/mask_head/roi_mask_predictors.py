# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d

import pdb


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))

        return self.mask_fcn_logits(x), None

class WithScorePredictor(nn.Module):
    def __init__(self, cfg):
        super(WithScorePredictor, self).__init__()

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_inputs = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        # do we need more parameters?
        self.score_avgpool = nn.AvgPool2d(kernel_size=14, stride=14)
        self.mask_score = nn.Linear(num_inputs, num_classes)

        nn.init.normal_(self.mask_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.mask_score.bias, 0)

    def forward(self, x):
        pool = self.score_avgpool(x)
        flat = pool.view(pool.size(0), -1)
        return self.mask_score(flat)

class MaskRCNNC4WithScorePredictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4WithScorePredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        self.score_predictor = WithScorePredictor(cfg)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        y = self.score_predictor(x)
        x = F.relu(self.conv5_mask(x))

        return self.mask_fcn_logits(x), y

_ROI_MASK_PREDICTOR = {"MaskRCNNC4Predictor": MaskRCNNC4Predictor, "MaskRCNNC4WithScorePredictor": MaskRCNNC4WithScorePredictor}

def make_roi_mask_predictor(cfg):
    func = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg)
