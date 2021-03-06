# YOLOv1-pytorch
This is a pure pytorch implementation for YOLOv1. 
Note that this is not an official implementation, 
only for me to learn details of **Object Detection** and 
entertainment. 

In the process of my study, I found that it is
not easy to build DarkNet environment, especailly on Windows,
so I wrote this repo, YOLOv1 in pure pytorch. I'm going to 
write a simple and easy-to-use reproduction version of yolov1.

## Experimental Result
To skip the pre-train process, I directly use ResNet-18 model inside PyTorch 
(pre-trained on ImageNet-1000 dataset) without the last two classify layers 
instead of the first 20 layers in the original paper. See more details at 
yolo.model.ResNet18_YOLOv1.

Then I trained my model on the VOC-2007-trainval dataset and validated on the
VOC-2007-test dataset. After about 2 days training, I got the best mAP of **58.2**
at last.

There are some gaps with the original. The paper said that they got the best mAP
of **63.4**. But they got this data after training on the VOC-2007 & VOC-2012 trainval
set and validated on the VOC-2012-test. It was the different training data and different
model structure made the gaps.

Here are some visual data of training process.

| Loss      | mAP |
| :---        |    :----:   |
| <img src="doc/Global_Loss (train).svg" width="300" height="200"/> | <img src="doc/Global_mAP (eval).svg" width="300" height="200"/>      |

The following sections are under preparation...

## Demo: Quick Start

The part will guide you to detect a single image with my model.

### 1. Create a virtual environment and activate it
```shell
conda create -n yolov1_pytorch python=3.7
conda activate yolov1_pytorch
```

### 2. Install PyTorch environment


## Train & Eval

## API