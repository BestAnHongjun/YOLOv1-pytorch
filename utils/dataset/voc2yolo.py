import cv2
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class voc2yolo_v1:
    """
    A transform which can convert VOC annotation(by voc.py) to yolo format
    """

    def __init__(self, grid_num=7, input_size=448):
        self.grid_num = grid_num
        self.input_size = input_size
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])

    def __call__(self, annotation):
        image_shape = annotation.get("image_shape")
        grid_size = image_shape[0] / self.grid_num, image_shape[1] / self.grid_num

        img = annotation.get("image")
        # img = self.image_transform(img)
        img = cv2.resize(img, (self.input_size, self.input_size)).transpose(2, 0, 1)
        source = torch.tensor(img, dtype=torch.float32)

        target = torch.zeros((self.grid_num, self.grid_num, 6))

        for object_to_detect in annotation.get("objects"):
            bbox = object_to_detect.get("bbox")
            class_id = object_to_detect.get("class_id")

            w = (bbox[2] - bbox[0]) / image_shape[0]
            h = (bbox[3] - bbox[1]) / image_shape[1]

            cx = (bbox[2] + bbox[0]) / 2
            cy = (bbox[3] + bbox[1]) / 2

            grid_j = int(cx // grid_size[0])
            grid_i = int(cy // grid_size[1])

            cx = (cx - grid_j * grid_size[0]) / grid_size[0]
            cy = (cy - grid_i * grid_size[1]) / grid_size[1]

            target[grid_i, grid_j, 0] = 1
            target[grid_i, grid_j, 1] = cx
            target[grid_i, grid_j, 2] = cy
            target[grid_i, grid_j, 3] = w
            target[grid_i, grid_j, 4] = h
            target[grid_i, grid_j, 5] = class_id

        return source, target

    def __repr__(self):
        return self.__class__.__name__ + '()'
