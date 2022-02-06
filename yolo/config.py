"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch
"""

import os
import torch

from yolo.loss import YOLOv1_LOSS
from yolo.model.ResNet18_YOLOv1 import ResNet18_YOLOv1

# project dir
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# YOLO basic config
YOLO_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLO_MODEL = ResNet18_YOLOv1().to(YOLO_DEVICE)
YOLO_LOSS = YOLOv1_LOSS(YOLO_DEVICE)
