import torch
import torch.nn as nn
import torch.functional as F

class ResNetBlock(nn.Module):
    """
    creates instances of blocks in the ResNet Architecture that involve taking the shortcut and adding to the tensor
    passed through the convolutional layers.

    The model contains 3 convolutinal layers with #channels as:
        (input_channels, output_channels // 4),
        (output_channels // 4, output_channels // 4),
        (output_channels, output_channels // 4),

    eg:- given channels size as 32 and 64, the model creates 3 convolutional layers of sizes (32, 16), (16, 16)
    and (16, 64)

    Atrributes:
        in_channels (int): #channels fed into first convolutional layer
        out_channels (int): #channels after convolutional layers' operations

    Returns:
        torch.Tensor: tensor of shape (batches, channels, width, height)
    """
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        logits = self.block_1(x)
        logits = self.block_2(logits)
        logits = self.block_3(logits)
        F.relu_(logits)
        return logits


class ResNet50(nn.Module):
    """
    creates a 50 layer deep neural network and as per the standard accepts 3-channel images of dimensions 224X224

    Attributes:
        repeats (List[int]): number of times a configuration of convolutional layers is repeated
        classes (int): the number of categories that the model needs to learn for to classify

    Returns:
        torch.Tensor: a tensor of shape (batches, classes) specifying the class the model predicts the input(s)
        belong(s) to
    """
    def __init__(self, repeats=[ 3, 4, 6, 3 ], classes=10):
        super(ResNet50, self).__init__()
        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )


        channels = [64, 128, 256, 256,  512]
        self.res_layer = nn.Sequential()

        for idx, x in enumerate(repeats):
            for xdx in range(x):
                if xdx == 0:
                    self.res_layer.add_module(
                        f'conv[{x}/{xdx+1}][{channels[idx]}]',
                        ResNetBlock(channels[idx], channels[idx + 1])
                    )
                else:
                    self.res_layer.add_module(
                        f'conv[{x}/{xdx+1}][{channels[idx]}]',
                        ResNetBlock(channels[idx + 1], channels[idx + 1])
                    )

        self.res_layer.add_module('avg pooling', nn.AdaptiveAvgPool2d(1))

        self.output_layer = nn.Sequential(
            nn.Linear(channels[-1], classes),
            nn.Softmax(dim=1),
        )

        # initializing with Xavier uniform aka Glorot uniform
        nn.init.xavier_uniform_(self.stage_1[0].weight)
        nn.init.xavier_uniform_(self.output_layer[0].weight)

    def forward(self, x):
        logits = self.stage_1(x)
        logits = self.res_layer(logits)
        logits = torch.flatten(logits, start_dim=1)
        logits = self.output_layer(logits)
        return logits