"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch
"""

import torch
import torch.nn as nn
from yolo.IOU import iou


class YOLOv1_LOSS(nn.Module):
    def __init__(self, device, grid_num=7, img_size=448, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOv1_LOSS, self).__init__()
        self.device = device
        self.grid_num = grid_num
        self.img_size = img_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, y_pre, y_true, writer=None, batch_id=0):
        """
        :param y_pre: Tensor, [batch_size, S, S, 30], 30: ((cx, cy, w, h, c)*2, class*20)
        :param y_true:  Tensor, [batch_size, S, S, 8],
                        8: (class_id, cx, cy, w, h, filename(int), src_width, src_height)
        :param writer: Record tensorboard log
        :param batch_id: Global index of tensorboard log
        :return: Scaler, loss
        """
        eps = 1e-12
        loss_confidence = torch.tensor([0], dtype=torch.float32).to(self.device)
        loss_coordinate = torch.tensor([0], dtype=torch.float32).to(self.device)
        loss_scale = torch.tensor([0], dtype=torch.float32).to(self.device)
        loss_classification = torch.tensor([0], dtype=torch.float32).to(self.device)
        batch_size = y_pre.shape[0]
        for bid in range(batch_size):
            for grid_i in range(self.grid_num):
                for grid_j in range(self.grid_num):
                    if y_true[bid, grid_i, grid_j, 0] != 0.0:
                        # 选择IOU较大的预测框
                        confidence_true = list()
                        confidence_true.append(iou(
                            y_pre[bid, grid_i, grid_j, 0:4],
                            y_true[bid, grid_i, grid_j, 1:5],
                            grid_i=grid_i,
                            grid_j=grid_j,
                            img_size=self.img_size,
                            grid_num=self.grid_num
                        ))
                        confidence_true.append(iou(
                            y_pre[bid, grid_i, grid_j, 5:9],
                            y_true[bid, grid_i, grid_j, 1:5],
                            grid_i=grid_i,
                            grid_j=grid_j,
                            img_size=self.img_size,
                            grid_num=self.grid_num
                        ))
                        if confidence_true[0] > confidence_true[1]:
                            choose_bbox = 0
                        else:
                            choose_bbox = 1

                        # 计算置信度损失
                        confidence_pre = y_pre[bid, grid_i, grid_j, 5 * choose_bbox + 4]
                        loss_confidence += torch.pow(confidence_pre - confidence_true[choose_bbox], 2)

                        # 计算中心坐标损失
                        loss_coordinate += self.lambda_coord * torch.pow(
                            y_pre[bid, grid_i, grid_j, (5 * choose_bbox):(5 * choose_bbox + 2)] -
                            y_true[bid, grid_i, grid_j, 1:3], 2
                        ).sum()

                        # 计算尺度损失
                        loss_scale += self.lambda_coord * torch.pow(
                            torch.sqrt(y_pre[bid, grid_i, grid_j, (5 * choose_bbox + 2):(5 * choose_bbox + 4)] + eps) -
                            torch.sqrt(y_true[bid, grid_i, grid_j, 3:5] + eps), 2
                        ).sum()

                        # 计算分类损失
                        class_pre = y_pre[bid, grid_i, grid_j, 10:]
                        class_true = torch.zeros(20, dtype=torch.float32).to(self.device)
                        class_true[int(y_true[bid, grid_i, grid_j, 0]) - 1] = 1
                        loss_classification += torch.pow(class_pre - class_true, 2).sum()
                    else:
                        # 该格子没有物体
                        confidence_true = 0
                        # 第一个预测框
                        confidence_pre = y_pre[bid, grid_i, grid_j, 4]
                        loss_confidence += self.lambda_noobj * torch.pow(confidence_pre - confidence_true, 2)
                        # 第二个预测框
                        confidence_pre = y_pre[bid, grid_i, grid_j, 9]
                        loss_confidence += self.lambda_noobj * torch.pow(confidence_pre - confidence_true, 2)

        loss_confidence /= batch_size
        loss_coordinate /= batch_size
        loss_scale /= batch_size
        loss_classification /= batch_size
        loss = loss_confidence + loss_coordinate + loss_scale + loss_classification

        if writer is not None:
            writer.add_scalar('Global/Loss (train)', loss.item(), batch_id)
            writer.add_scalar('Loss/Confidence', loss_confidence.item(), batch_id)
            writer.add_scalar('Loss/Coordinate', loss_coordinate.item(), batch_id)
            writer.add_scalar('Loss/Scale', loss_scale.item(), batch_id)
            writer.add_scalar('Loss/Classification', loss_classification.item(), batch_id)

        return loss


def test_loss():
    y_pre = torch.zeros((5, 7, 7, 30))
    y_true = torch.zeros((5, 7, 7, 8))
    loss_func = YOLOv1_LOSS(device=torch.device("cpu"))
    loss = loss_func(y_pre, y_true)
    print(loss.item())


if __name__ == "__main__":
    test_loss()
