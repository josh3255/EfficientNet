import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.efficientNet = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc2 = nn.Linear(1000, 6)

    def forward(self, x):
        out = self.efficientNet(x)
        out = self.fc2(out)

        return out