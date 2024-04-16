import torch
import torch.nn as nn
import torch.nn.functional as F

class MesoNet(nn.Module):
    """
        takes in 256x256 images
    """
    def __init__(self):
        super(MesoNet, self).__init__()
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.cnn_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.cnn_3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.cnn_4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        self.fcn = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(8 * 8, 16),
            nn.Dropout(.5),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
    
    def __call__(self, x):
        logits = self.cnn_1(x)
        logits = self.cnn_2(logits)
        logits = self.cnn_3(logits)
        logits = self.cnn_4(logits)
        logits = self.fcn(logits)
        return logits