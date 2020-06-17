from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d

class OrderPredictor(nn.Module):
    def __init__(self, cfg):
        super(OrderPredictor, self).__init__()

        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.project = nn.Linear(1024, 1)

        nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.normal_(self.project.weight, mean=0, std=0.01)
        nn.init.constant_(self.project.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.project(x)
        
        return x

class OrderPredictorThing(nn.Module):
    def __init__(self, cfg):
        super(OrderPredictorThing, self).__init__()

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        # not sure about this one.
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.project = nn.Linear(1024, num_classes)

        nn.init.kaiming_uniform_(self.fc1.weight, a=1)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.normal_(self.project.weight, mean=0, std=0.01)
        nn.init.constant_(self.project.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.project(x)
        
        return x

_ROI_ORDER_PREDICTOR = {
    "OrderPredictor": OrderPredictor,
    "OrderPredictorThing": OrderPredictorThing
}

def make_roi_order_predictor(cfg):
    func = _ROI_ORDER_PREDICTOR["OrderPredictor"]
    return func(cfg)
