import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d

class OrderHeadFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(OrderHeadFeatureExtractor, self).__init__()

        self.cfg = cfg.clone()
        
        # paired.
        input_channels = self.cfg.MODEL.ROI_ORDER_HEAD.NUMBER_INPUT_CHANNELS
        number_channels = 512

        self.conv1 = Conv2d(input_channels, number_channels, 3, 1, 1) 
        self.conv2 = Conv2d(number_channels, number_channels, 3, 1, 1) 
        self.conv3 = Conv2d(number_channels, number_channels, 3, 1, 1)

        # stride 2.
        self.conv4 = Conv2d(number_channels, number_channels, 3, 2, 1) 

        for l in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
 
        return x

def make_roi_order_feature_extractor(cfg):
    return OrderHeadFeatureExtractor(cfg)
