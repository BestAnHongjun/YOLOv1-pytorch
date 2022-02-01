"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/AnHongjun001/YOLOv1-pytorch
"""

import torch


def iou(bbox_pre, bbox_true, grid_i, grid_j, img_size=448, grid_num=7):
    """
    :param bbox_pre: Tensor, shape:[4], 4:bbox(cx, cy, w, h)
    :param bbox_true: Tensor, shape:[4], 4:bbox(cx, cy, w, h)
    :param grid_i: The row index (or y index) of the grid
    :param grid_j: The col index (or x index) of the grid
    :param img_size: The input image's shape of the model is (img_size, img_size, 3)
    :param grid_num: In the paper, this num is 7
    :return: Tensor, shape:[batch_size], iou
    """
    # 栅格大小
    grid_size = img_size / grid_num

    # 将归一化的预测值转化为实际的预测值
    cx_pre = bbox_pre[0] * grid_size + grid_j * grid_size
    cy_pre = bbox_pre[1] * grid_size + grid_i * grid_size
    w_pre = bbox_pre[2] * img_size
    h_pre = bbox_pre[3] * img_size

    # 将预测值(cx, cy, w, h)转成(x_min, y_min, x_max, y_max)
    x_min_pre = cx_pre - w_pre / 2
    y_min_pre = cy_pre - h_pre / 2
    x_max_pre = cx_pre + w_pre / 2
    y_max_pre = cy_pre + h_pre / 2

    # 将归一化的标定值转化为实际的标定值
    cx_true = bbox_true[0] * grid_size + grid_j * grid_size
    cy_true = bbox_true[1] * grid_size + grid_i * grid_size
    w_true = bbox_true[2] * img_size
    h_true = bbox_true[3] * img_size

    # 标定值(x_min, y_min, x_max, y_max)
    x_min_true = cx_true - w_true / 2
    y_min_true = cy_true - h_true / 2
    x_max_true = cx_true + w_true / 2
    y_max_true = cy_true + w_true / 2

    # 相交区域矩形(x_min, y_min, x_max, y_max)
    union_x_min = max(x_min_pre, x_min_true)
    union_x_max = min(x_max_pre, x_max_true)
    union_y_min = max(y_min_pre, y_min_true)
    union_y_max = min(y_max_pre, y_max_true)

    # 无相交区域，交并比为0
    if union_x_min >= union_x_max or union_y_min >= union_y_max:
        return 0

    # 相交区域矩形面积，预测区域矩形面积，标定区域矩形面积
    area_union = (union_x_max - union_x_min) * (union_y_max - union_y_min)
    area_pre = (x_max_pre - x_min_pre) * (y_max_pre - y_min_pre)
    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)

    # 计算交并比
    res = area_union / (area_pre + area_true - area_union)
    return res


def test_iou():
    bbox_pre = torch.zeros(4, dtype=torch.float32)
    bbox_true = torch.zeros(4, dtype=torch.float32)
    res = iou(bbox_pre, bbox_true, grid_i=0, grid_j=0, img_size=448, grid_num=7)
    print(res)


if __name__ == "__main__":
    test_iou()
