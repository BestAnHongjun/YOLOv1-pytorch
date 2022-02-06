"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch

Reference:
[1]Code in function __viz_bbox refers to code by Ge Zheng et al. in project YOLO-X,
see more detail at https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/visualize.py,
which is under Apache License 2.0.
"""

import cv2
import numpy as np


def viz_vdict(vdict):
    image = vdict.get("image").copy()
    objects = vdict.get("objects")
    for obj in objects:
        if "confidence" in obj:
            confidence = obj.get("confidence")
        else:
            confidence = -1
        __viz_bbox(image, obj.get("bbox"), obj.get("class_id"), obj.get("class_name"), confidence)
    return image


def viz_yolo_ground_truth(source, target, voc_class_list):
    """
    :param source: Tensor [c, h, w], c = 3
    :param target: Tensor [grid_num, grid_num, 8], grid_num = 7, h = w = 448
                   8:[class_id, cx, cy, w, h, filename(int), src_width, src_height]
    :param voc_class_list: A list contains classes' name of VOC dataset
    :return: numpy.array([[[...]]]), Cv2 image Mat with shape [src_height, src_width, 3]
    """
    src_image = source.numpy().astype(np.uint8).transpose(1, 2, 0).copy()
    src_width = int(target[0, 0, 6])
    src_height = int(target[0, 0, 7])
    grid_size = (src_width / 7, src_height / 7)

    src_image = cv2.resize(src_image, (src_width, src_height))
    res_image = src_image.copy()

    for grid_i in range(7):
        for grid_j in range(7):
            if target[grid_i, grid_j, 0] == 0:
                continue

            w = target[grid_i, grid_j, 3] * src_width
            h = target[grid_i, grid_j, 4] * src_height

            cx = target[grid_i, grid_j, 1] * grid_size[0] + grid_j * grid_size[0]
            cy = target[grid_i, grid_j, 2] * grid_size[1] + grid_i * grid_size[1]

            cls_id = int(target[grid_i, grid_j, 0])

            bbox = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            __viz_bbox(res_image, bbox, cls_id, voc_class_list[cls_id])

    return src_image, res_image


def __viz_bbox(image, bbox, cls_id, cls_name, confidence=-1):
    x_min, y_min, x_max, y_max = bbox
    color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
    if confidence == -1:
        text = '{}'.format(cls_name)
    else:
        text = '{}:{:.2f}%'.format(cls_name, confidence * 100)
    txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        image,
        (x_min, y_min + 1),
        (x_min + txt_size[0] + 1, y_min + int(1.5 * txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(image, text, (x_min, y_min + txt_size[1]), font, 0.4, txt_color, thickness=1)


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
