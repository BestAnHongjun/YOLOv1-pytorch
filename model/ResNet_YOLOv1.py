"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet_YOLOv1(nn.Module):
    def __init__(self):
        super(ResNet_YOLOv1, self).__init__()

        # Layer 1-4 (Use ResNet50 instead)
        resnet18 = models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(resnet18.children())[:-2])

        # Layer 5
        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Layer 6
        self.layer_6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Fully-Connected Layer
        self.fc1 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=3),
            nn.Linear(in_features=1024 * 7 * 7, out_features=4096),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=30 * 7 * 7),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.feature(x)
        y = self.layer_5(y)
        y = self.layer_6(y)
        y = self.fc1(y)
        y = self.fc2(y)
        y = y.reshape(-1, 7, 7, 30)
        return y


def test_train():
    model = ResNet_YOLOv1()

    x = torch.zeros((1, 3, 448, 448))
    y_pre = model(x)

    print("===TEST_TRAIN===")
    print("Input:", x.shape)
    print("Output:", y_pre.shape)


if __name__ == "__main__":
    test_train()
