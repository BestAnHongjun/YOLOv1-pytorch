import os
import cv2
import sys
import json
import matplotlib.pyplot as plt

import torch

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(PROJECT_DIR)

from yolo.config import PROJECT_DIR, YOLO_MODEL, YOLO_DEVICE
from yolo.utils.parse_predict import predict2vdict
from yolo.utils.viz_bbox import viz_vdict


def init_detector():
    with open(os.path.join(PROJECT_DIR, "tools", "config", "class.cfg.json"), "r") as json_file:
        class_list = json.load(json_file).get("class-list")

    with open(os.path.join(PROJECT_DIR, "tools", "config", "infer.cfg.json"), "r") as json_file:
        infer_cfg = json.load(json_file)

    return class_list, infer_cfg


def detect(source, filename_list, src_shape_list, class_list, infer_cfg):
    YOLO_MODEL.load_state_dict(
        torch.load(os.path.join(infer_cfg.get("model-path").replace("{PROJECT_DIR}", PROJECT_DIR)))
    )
    predict = YOLO_MODEL(source)
    result_vdict = predict2vdict(predict, filename_list, src_shape_list, infer_cfg.get("threshold"), class_list)
    return result_vdict


def infer_one_dir():
    pass


def infer_one_image(image_path, class_list, infer_cfg):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filename_list = [os.path.basename(image_path)]
    src_shape_list = [(image.shape[1], image.shape[0])]

    image_process = cv2.resize(image, (448, 448)).transpose(2, 0, 1)

    source = torch.tensor(image_process, dtype=torch.float32).unsqueeze(0).to(YOLO_DEVICE)
    result_vdict = detect(source, filename_list, src_shape_list, class_list, infer_cfg)
    result_vdict[0]["image"] = image

    return result_vdict


def infer():
    class_list, infer_cfg = init_detector()

    if infer_cfg.get("mode") == "image":
        image_path = infer_cfg.get("mode-path")
        result_vdict = infer_one_image(image_path, class_list, infer_cfg)
    elif infer_cfg.get("mode") == "dir":
        image_dir = infer_cfg.get("mode-path")


if __name__ == "__main__":
    infer()
