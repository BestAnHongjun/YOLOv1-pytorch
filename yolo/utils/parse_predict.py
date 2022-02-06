"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch
"""

import torch


def predict2vdict(predict, filename_list, src_shape_list, threshold=0, class_list=None):
    """
    :param predict: Tensor, yolo output, [batch_size, c, h, w]
    :param filename_list: A list of filename with length of batch_size
    :param src_shape_list: A list of image shape with length of batch_size
    :param threshold: Threshold of confidence
    :param class_list: A list contains classes of dataset
    :return: A list of vdict with length of batch_size
    """

    res = []
    grid_size = 448 / 7

    for bid in range(predict.shape[0]):
        filename = filename_list[bid]
        w = src_shape_list[bid][0]
        h = src_shape_list[bid][1]

        fh = h / 448.0
        fw = w / 448.0

        vdict = dict()
        vdict["filename"] = filename
        vdict["src_width"] = w
        vdict["src_height"] = h
        vdict["objects"] = list()

        for grid_i in range(7):
            for grid_j in range(7):
                # 找置信概率较大的框
                if predict[bid, grid_i, grid_j, 4] > predict[bid, grid_i, grid_j, 9]:
                    confidence_pre = predict[bid, grid_i, grid_j, 4]
                    coordinate_pre = predict[bid, grid_i, grid_j, 0:4]
                else:
                    confidence_pre = predict[bid, grid_i, grid_j, 9]
                    coordinate_pre = predict[bid, grid_i, grid_j, 5:9]

                # I can fuck this statement billions of years!!!!
                # class_tensor = predict[bid, grid_i, grid_j, 10:20].unsqueeze(0)

                class_tensor = predict[bid, grid_i, grid_j, 10:].unsqueeze(0)

                class_id = torch.argmax(class_tensor, dim=1)
                confidence = class_tensor[class_id[0]] * confidence_pre

                if confidence < threshold:
                    continue

                cx_pre = coordinate_pre[0] * grid_size + grid_j * grid_size
                cy_pre = coordinate_pre[1] * grid_size + grid_i * grid_size
                w_pre = coordinate_pre[2] * 448
                h_pre = coordinate_pre[3] * 448

                x_min = max(int((cx_pre - w_pre / 2) * fw), 0)
                y_min = max(int((cy_pre - h_pre / 2) * fh), 0)
                x_max = min(int((cx_pre + w_pre / 2) * fw), w - 1)
                y_max = min(int((cy_pre + h_pre / 2) * fh), h - 1)

                obj = dict()
                obj["class_id"] = int(class_id.item()) + 1
                if class_list is not None:
                    obj["class_name"] = class_list[obj.get("class_id")]
                obj["confidence"] = float(confidence.item())
                obj["bbox"] = (x_min, y_min, x_max, y_max)
                vdict["objects"].append(obj)

        res.append(vdict)

    return res
