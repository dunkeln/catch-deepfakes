import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):
    """
    docstring bru
    """
    def __init__(self):
        super(VGGNet, self).__init__()
        def create(in_channels: int, out_channels: int, max_pool: bool = False) -> nn.Sequential:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.layer_1 = create(3, 64)
        self.layer_2 = create(64, 64)
        self.layer_3 = create(64, 128)
        self.layer_4 = create(128, 128, max_pool=True)
        self.layer_5 = create(128, 256)
        self.layer_6 = create(256, 256)
        self.layer_7 = create(256, 256)
        self.layer_8 = create(256, 512)
        self.layer_9 = create(512, 512)
        self.layer_10 = create(512, 512, max_pool=True)
        self.layer_11 = create(512, 512)
        self.layer_12 = create(512, 512)
        self.layer_13 = create(512, 512, max_pool=True)

        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(.5),
            nn.Linear(4096, 4096),
            nn.ReLU(in_place=True),
            nn.Dropout(.5),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        logits = self.layer_1(x)
        logits = self.layer_2(logits)
        logits = self.layer_3(logits)
        logits = self.layer_4(logits)
        logits = self.layer_5(logits)
        logits = self.layer_6(logits)
        logits = self.layer_7(logits)
        logits = self.layer_8(logits)
        logits = self.layer_9(logits)
        logits = self.layer_10(logits)
        logits = self.layer_11(logits)
        logits = self.layer_12(logits)
        logits = self.layer_13(logits)
        logits = self.fcn(logits)
        return logits