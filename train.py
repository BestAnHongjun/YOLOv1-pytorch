"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/AnHongjun001/YOLOv1-pytorch
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

import os.path
import torch
from model.ResNet_YOLOv1 import ResNet_YOLOv1
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.loss import YOLO_LOSS
from utils.dataset.voc import VOC_DATASET, VOC_CLASS
from utils.dataset.tools.voc_eval import Evaluator


train_dataset = VOC_DATASET(
    dataset_dir=r"E:\dataset\PASCAL_VOC\VOC_2007_trainval",
    txt_file_name="trainval",
    augmentation=True
)
eval_dataset = VOC_DATASET(
    dataset_dir=r"E:\dataset\PASCAL_VOC\VOC_2007_test",
    txt_file_name="test",
    augmentation=False
)

batch_size = 8
epoch_n = 105
batch_n_train = train_dataset.__len__() // batch_size
batch_n_eval = eval_dataset.__len__() // batch_size

train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
eval_DataLoader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLO_net = ResNet_YOLOv1().to(device)
loss_func = YOLO_LOSS(device)

model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "autosave")
log_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
writer = SummaryWriter(log_dir=log_save_dir)
n = 0


def train_one_epoch(model, epoch, optimizer, evaluator, best_ap, latesd_ap):
    global n

    # train
    model.train()
    for batch, (source, target) in enumerate(train_DataLoader):
        n += 1
        source = source.to(device)
        target = target.to(device)
        predict = model(source)

        optimizer.zero_grad()
        loss = loss_func(predict, target)
        loss.backward()
        optimizer.step()

        print("[best mAP:{:.4f} latest mAP:{:.4f}] Train Epoch:{}/{} Batch:{}/{} Loss:{:5f}".format(
            best_ap,
            latesd_ap,
            epoch,
            epoch_n,
            batch,
            batch_n_train,
            loss.item()
        ))
        writer.add_scalar('Train/Loss', loss.item(), n)

    # saving latest model
    print("Saving latest model...")
    torch.save(YOLO_net.state_dict(), os.path.join(model_save_dir, "model_latest.pth"))

    # eval
    model.eval()
    evaluator.clear_results()
    grid_size = 448 / 7
    for batch, (source, target) in enumerate(eval_DataLoader):
        source = source.to(device)
        predict = model(source)

        for bid in range(batch_size):
            filename = "{:06d}".format(int(target[bid, 0, 0, 5]))
            w = int(target[bid, 0, 0, 6])
            h = int(target[bid, 0, 0, 7])

            fh = h / 448.0
            fw = w / 448.0

            for grid_i in range(7):
                for grid_j in range(7):
                    # 找置信概率较大的框
                    if predict[bid, grid_i, grid_j, 4] > predict[bid, grid_i, grid_j, 9]:
                        confidence_pre = predict[bid, grid_i, grid_j, 4]
                        coordinate_pre = predict[bid, grid_i, grid_j, 0:4]
                    else:
                        confidence_pre = predict[bid, grid_i, grid_j, 9]
                        coordinate_pre = predict[bid, grid_i, grid_j, 5:9]
                    class_tensor = predict[bid, grid_i, grid_j, 10:20].unsqueeze(0)

                    class_id = torch.argmax(class_tensor, dim=1)
                    confidence = predict[bid, grid_i, grid_j, 10 + class_id[0]] * confidence_pre

                    cx_pre = coordinate_pre[0] * grid_size + grid_j * grid_size
                    cy_pre = coordinate_pre[1] * grid_size + grid_i * grid_size
                    w_pre = coordinate_pre[2] * 448
                    h_pre = coordinate_pre[3] * 448

                    x_min = max(int((cx_pre - w_pre / 2) * fw), 0)
                    y_min = max(int((cy_pre - h_pre / 2) * fh), 0)
                    x_max = min(int((cx_pre + w_pre / 2) * fw), w - 1)
                    y_max = min(int((cy_pre + h_pre / 2) * fh), h - 1)

                    evaluator.add_result(int(class_id), filename, confidence, x_min, y_min, x_max, y_max)

        print("[best mAP:{:.4f} latest mAP:{:.4f}] Eval Epoch:{}/{} Batch:{}/{} ...".format(
            best_ap,
            latesd_ap,
            epoch,
            epoch_n,
            batch,
            batch_n_eval
        ))

    m_ap, aps, recs, precs = evaluator.eval()
    writer.add_scalar('mAP', m_ap, epoch)
    for cls_id in range(len(aps)):
        writer.add_scalar('AP/{}'.format(VOC_CLASS[cls_id + 1]), aps[cls_id], epoch)

    if m_ap > best_ap:
        best_ap = m_ap
        print("Saving best model...")
        torch.save(YOLO_net.state_dict(), os.path.join(model_save_dir, "model_best.pth"))

    return best_ap, m_ap


def train():
    evaluator = Evaluator(
        dataset_dir=r"E:\dataset\PASCAL_VOC\VOC_2007_test",
        txt_file_name="test",
        voc_class_list=VOC_CLASS,
        cache_dir=cache_dir,
        use_07_metric=True
    )
    best_ap = 0.0
    latest_ap = 0.0

    optimizer = torch.optim.SGD(YOLO_net.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1):
        best_ap, latest_ap = train_one_epoch(YOLO_net, epoch, optimizer, evaluator, best_ap, latest_ap)

    optimizer = torch.optim.SGD(YOLO_net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, 76):
        best_ap, latest_ap = train_one_epoch(YOLO_net, epoch, optimizer, evaluator, best_ap, latest_ap)

    optimizer = torch.optim.SGD(YOLO_net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    for epoch in range(76, 500):
        best_ap, latest_ap = train_one_epoch(YOLO_net, epoch, optimizer, evaluator, best_ap, latest_ap)


if __name__ == "__main__":
    train()




