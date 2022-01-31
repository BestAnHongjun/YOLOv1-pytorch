import cv2
import torch
import torchvision.transforms as transforms


class vdict2yolo_v1:
    """
    A transform which can convert VOC annotation(by voc.py) to yolo format
    """

    def __init__(self, grid_num=7, input_size=448):
        self.grid_num = grid_num
        self.input_size = input_size

    def __call__(self, vdict):
        image = vdict.get("image")
        image_shape = image.shape
        grid_size = image_shape[1] / self.grid_num, image_shape[0] / self.grid_num

        img = cv2.resize(image, (self.input_size, self.input_size)).transpose(2, 0, 1)
        source = torch.tensor(img, dtype=torch.float32)

        target = torch.zeros((self.grid_num, self.grid_num, 8))
        target[:, :, 5] = int(vdict.get("filename"))  # filename
        target[:, :, 6] = image_shape[1]  # src_width
        target[:, :, 7] = image_shape[0]  # src_height

        for object_to_detect in vdict.get("objects"):
            bbox = object_to_detect.get("bbox")
            class_id = object_to_detect.get("class_id")

            w = (bbox[2] - bbox[0]) / image_shape[1]
            h = (bbox[3] - bbox[1]) / image_shape[0]

            cx = (bbox[2] + bbox[0]) / 2
            cy = (bbox[3] + bbox[1]) / 2

            grid_j = int(cx // grid_size[0])
            grid_i = int(cy // grid_size[1])

            cx = (cx - grid_j * grid_size[0]) / grid_size[0]
            cy = (cy - grid_i * grid_size[1]) / grid_size[1]

            target[grid_i, grid_j, 0] = class_id
            target[grid_i, grid_j, 1] = cx
            target[grid_i, grid_j, 2] = cy
            target[grid_i, grid_j, 3] = w
            target[grid_i, grid_j, 4] = h

        return source, target

    def __repr__(self):
        return self.__class__.__name__ + '()'
