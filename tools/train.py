"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch
"""

import os
import sys
import json
import pickle
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(PROJECT_DIR)

from yolo.config import PROJECT_DIR, YOLO_MODEL, YOLO_DEVICE, YOLO_LOSS
from yolo.utils.parse_predict import predict2vdict
from yolo.utils.parse_cfg_json import parse_dataset, parse_data_loader, parse_evaluator


def train_one_epoch(model, name, optimizer, evaluator, writer):
    """
    train one epoch and eval one epoch
    """
    global class_list
    global epoch_id, batch_id
    global best_ap, latest_ap
    global batch_n_train, batch_n_eval
    global model_save_dir, log_save_dir
    global train_data_loader, eval_data_loader

    # train
    model.train()
    for batch, (source, target) in enumerate(train_data_loader):
        batch_id += 1

        source = source.to(YOLO_DEVICE)
        target = target.to(YOLO_DEVICE)
        predict = model(source)

        optimizer.zero_grad()
        loss = YOLO_LOSS(predict, target, writer, batch_id)
        loss.backward()
        optimizer.step()

        format_str = "[{}] [best-mAP:{:.4f} | latest-mAP:{:.4f}] Train Epoch:{} Batch:{}/{} Loss:{:5f}"
        print(format_str.format(name, best_ap, latest_ap, epoch_id, batch + 1, batch_n_train, loss.item()))

    # eval
    model.eval()
    evaluator.clear_results()

    for batch, (source, target) in enumerate(eval_data_loader):
        source = source.to(YOLO_DEVICE)
        predict = model(source)

        local_batch_size = int(source.shape[0])
        filename_list = ["{:06d}".format(int(target[bid, 0, 0, 5])) for bid in range(local_batch_size)]
        src_shape_list = [(int(target[bid, 0, 0, 6]), int(target[bid, 0, 0, 7])) for bid in range(local_batch_size)]

        result_vdict_list = predict2vdict(predict, filename_list, src_shape_list)

        for result_vdict in result_vdict_list:
            filename = result_vdict.get("filename")
            for object in result_vdict.get("objects"):
                cls_id = object.get("class_id")
                confidence = object.get("confidence")
                x_min, y_min, x_max, y_max = object.get("bbox")
                evaluator.add_result(int(cls_id), filename, confidence, x_min, y_min, x_max, y_max)

        format_str = "[{}] [best-mAP:{:.4f} | latest-mAP:{:.4f}] Eval Epoch:{} Batch:{}/{} ..."
        print(format_str.format(name, best_ap, latest_ap, epoch_id, batch + 1, batch_n_eval))

    latest_ap, aps, recs, precs = evaluator.eval()
    writer.add_scalar('Global/mAP (eval)', latest_ap, epoch_id)
    for cls_id in range(len(aps)):
        writer.add_scalar('AP/{}'.format(class_list[cls_id + 1]), aps[cls_id], epoch_id)

    if latest_ap > best_ap:
        best_ap = latest_ap
        print("Saving the best model...")
        torch.save(model.state_dict(), os.path.join(model_save_dir, "[{}]model_best.pth".format(name)))

    # Cache the model data for recovery
    print("Caching the model data...")
    torch.save(model.state_dict(), os.path.join(cache_save_dir, "model.cache"))
    # Cache the optimizer for recovery
    print("Caching the optimizer...")
    torch.save(optimizer.state_dict(), os.path.join(cache_save_dir, "optimizer.cache"))
    # Cache the AP data for recovery
    print("Caching the basic data...")
    with open(os.path.join(cache_save_dir, "basic.cache"), "wb") as f:
        pickle.dump((best_ap, latest_ap, epoch_id, batch_id), f)


def train(train_cfg_name, class_cfg_name):
    # read class list
    global class_list
    with open(os.path.join(PROJECT_DIR, "tools", "config", "{}.cfg.json".format(class_cfg_name)), "r") as json_file:
        class_list = json.load(json_file).get("class-list")

    # read offline config
    with open(os.path.join(PROJECT_DIR, "tools", "config", "{}.cfg.json".format(train_cfg_name)), "r") as json_file:
        train_config = json.load(json_file)
    train_config_offline = train_config.get("offline-config")
    train_config_online = train_config.get("online-config")

    # name
    name = train_config_offline.get("name")

    # dataset
    train_dataset = parse_dataset(train_config_offline.get("dataset").get("train"), class_list)
    eval_dataset = parse_dataset(train_config_offline.get("dataset").get("eval"), class_list)

    # data loader
    global batch_n_train, batch_n_eval
    global train_data_loader, eval_data_loader
    train_data_loader = parse_data_loader(train_config_offline.get("data-loader").get("train"), train_dataset)
    eval_data_loader = parse_data_loader(train_config_offline.get("data-loader").get("eval"), eval_dataset)
    batch_n_train = train_dataset.__len__() // train_config_offline.get("data-loader").get("train").get("batch-size")
    batch_n_eval = eval_dataset.__len__() // train_config_offline.get("data-loader").get("eval").get("batch-size")

    # initialize dir
    global model_save_dir, log_save_dir, cache_save_dir
    model_save_dir = train_config_offline.get("model-save-dir").replace("{PROJECT_DIR}", PROJECT_DIR)
    log_save_dir = train_config_offline.get("log-save-dir").replace("{PROJECT_DIR}", PROJECT_DIR)
    cache_save_dir = train_config_offline.get("cache-save-dir").replace("{PROJECT_DIR}", PROJECT_DIR)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)
    if not os.path.exists(cache_save_dir):
        os.mkdir(cache_save_dir)

    # evaluator
    evaluator = parse_evaluator(train_config_offline.get("dataset").get("eval"), class_list, cache_save_dir)

    # log writer
    writer = SummaryWriter(log_dir=log_save_dir)

    # initialize
    optimizer_history_cfg = train_config_online.get("optimizer")
    optimizer = torch.optim.SGD(
        YOLO_MODEL.parameters(),
        lr=optimizer_history_cfg.get("lr"),
        momentum=optimizer_history_cfg.get("momentum"),
        weight_decay=optimizer_history_cfg.get("weight-decay")
    )

    # use pre-trained model
    if train_config.get("use-pre-trained-model").get("enable"):
        model_path = train_config.get("use-pre-trained-model").get("model-path").replace("{PROJECT_DIR}", PROJECT_DIR)
        YOLO_MODEL.load_state_dict(torch.load(model_path))

    # checkpoint-recovery
    global best_ap, latest_ap, epoch_id, batch_id
    if train_config.get("checkpoint-recovery"):
        print("Run checkpoint recovery...")
        if os.path.exists(os.path.join(cache_save_dir, "model.cache")):
            print("Recovering model...")
            YOLO_MODEL.load_state_dict(torch.load(os.path.join(cache_save_dir, "model.cache")))
        if os.path.exists(os.path.join(cache_save_dir, "optimizer.cache")):
            print("Recovering optimizer...")
            optimizer.load_state_dict(torch.load(os.path.join(cache_save_dir, "optimizer.cache")))
        if os.path.exists(os.path.join(cache_save_dir, "basic.cache")):
            print("Recovering basic data...")
            with open(os.path.join(cache_save_dir, "basic.cache"), "rb") as f:
                best_ap, latest_ap, epoch_id, batch_id = pickle.load(f)

    while True:
        epoch_id += 1

        # read online config
        with open(os.path.join(PROJECT_DIR, "tools", "config", "train.cfg.json"), "r") as json_file:
            train_config_online = json.load(json_file).get("online-config")

        optimizer_cfg = train_config_online.get("optimizer")
        if optimizer_cfg != optimizer_history_cfg:
            optimizer = torch.optim.SGD(
                YOLO_MODEL.parameters(),
                lr=optimizer_cfg.get("lr"),
                momentum=optimizer_cfg.get("momentum"),
                weight_decay=optimizer_cfg.get("weight-decay")
            )

        train_one_epoch(YOLO_MODEL, name, optimizer, evaluator, writer)


best_ap = 0.0
latest_ap = 0.0
batch_n_train = 0
batch_n_eval = 0
model_save_dir = ""
log_save_dir = ""
cache_save_dir = ""
class_list = []
train_data_loader = []
eval_data_loader = []
epoch_id = 0
batch_id = 0


def make_parser():
    parser = argparse.ArgumentParser("YOLO trainer")
    parser.add_argument("-t", "--train_cfg", type=str, default="train")
    parser.add_argument("-c", "--class_cfg", type=str, default="class")
    return parser


if __name__ == "__main__":
    args = vars(make_parser().parse_args())
    train(args.get("train_cfg"), args.get("class_cfg"))
