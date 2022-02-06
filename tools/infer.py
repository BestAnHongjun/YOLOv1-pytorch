import os.path

import cv2
import json
import torch
import matplotlib.pyplot as plt

from yolo.config import PROJECT_DIR, YOLO_MODEL, YOLO_DEVICE
from yolo.utils.parse_predict import predict2vdict
from yolo.utils.viz_bbox import viz_vdict


if __name__ == "__main__":
    image_path = r"E:\dataset\PASCAL_VOC\VOC_2007_test\JPEGImages\000085.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(os.path.join(PROJECT_DIR, "tools", "config", "class.cfg.json"), "r") as json_file:
        class_list = json.load(json_file).get("class-list")

    img = cv2.resize(image, (448, 448)).transpose(2, 0, 1)
    source = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(YOLO_DEVICE)

    YOLO_MODEL.load_state_dict(torch.load(os.path.join(PROJECT_DIR, "yolo", "model", "pth", "model_resnet_best.pth")))

    predict = YOLO_MODEL(source)
    result_vdict = predict2vdict(predict, ["001593"], [(image.shape[1], image.shape[0])], 0.2, class_list)
    result_vdict[0]["image"] = image

    result_image = viz_vdict(result_vdict[0])

    # print(result_vdict)
    plt.imshow(result_image)
    plt.show()
