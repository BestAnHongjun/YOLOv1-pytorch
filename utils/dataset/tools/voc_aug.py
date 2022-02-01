"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch

Reference:
[1]Code in method voc_aug.random_hsv_augmentation refers to code by 太阳花的小绿豆 at CSDN platform,
see more detail at https://blog.csdn.net/qq_37541097/article/details/119478023,
which is under CC 4.0 BY-SA License.
"""

import cv2
import numpy as np


class voc_aug:
    """
    A transformer which can randomly augment VOC format dataset online.
    The highlight is, 1) it augments both image and b-box!!!
                      2) it only use cv2 & numpy, means it could be used simply without any other awful packages!!!
                      3) it is an online transformer!!!

    It contains methods of:
        1. Random HSV augmentation
        2. Random Cropping augmentation
        3. Random Flipping augmentation
        4. Random Noise augmentation
        5. Random rotation or translation augmentation

    All the methods can adjust abundant arguments in the constructed function.
    """
    def __init__(self):
        # Probabilities to augmentation methods [0 - 1]
        self.hsv_pro = 0.8              # HSV augment
        self.crop_pro = 0.8             # crop augment.
        self.flip_pro = 0.5             # horizontally flip augmentation
        self.noise_pro = 0.3            # salt and pepper noise
        self.rotate_or_trans_pro = 0.8  # rotation augment

        # args of HSV augment
        self.h_gain = 0.5
        self.s_gain = 0.5
        self.v_gain = 0.5

        # args of crop augmentation
        self.min_crop_rate = 0.8
        self.max_crop_rate = 1.0

        # args of noise augment
        self.min_snr = 0.95
        self.max_snr = 1.0

        # args of rotation augmentation
        self.min_degree = -30
        self.max_degree = 30

        # args of translation augment
        self.x_trans_rate = (0, 0.3)
        self.y_trans_rate = (0, 0.3)

        # threshold of discard bbox
        self.bbox_threshold = 10

    def __call__(self, vdict):
        # Random HSV augment
        random_num = np.random.random()
        if random_num < self.hsv_pro:
            vdict = self.random_hsv_augmentation(vdict, self.h_gain, self.s_gain, self.v_gain)

        # Random cropping augmentation
        random_num = np.random.random()
        if random_num < self.crop_pro:
            vdict = self.random_cropping_augmentation(vdict, self.min_crop_rate, self.max_crop_rate)

        # Random flipping augmentation
        vdict = self.random_flipping_augmentation(vdict, self.flip_pro)

        # Random noise augmentation
        random_num = np.random.random()
        if random_num < self.noise_pro:
            vdict = self.random_noise_augmentation(vdict, self.min_snr, self.max_snr)

        # Random rotation augment or translation augment
        random_num = np.random.random()
        if random_num < self.rotate_or_trans_pro:
            random_num = np.random.random()
            if random_num < 0.5:
                vdict = self.random_rotation_augmentation(vdict, self.min_degree, self.max_degree)
            else:
                vdict = self.random_translation_augmentation(vdict, self.x_trans_rate, self.y_trans_rate)

        return vdict

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def random_hsv_augmentation(vdict, h_gain, s_gain, v_gain):
        img = vdict.get("image")
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(np.uint8)
        aug_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        vdict["image"] = aug_img
        return vdict

    @staticmethod
    def random_flipping_augmentation(vdict, flip_pro):
        random_num = np.random.random()
        if random_num >= flip_pro:
            return vdict
        img = vdict.get("image")
        cx = img.shape[1] / 2.0
        img = cv2.flip(img, 1)
        vdict["image"] = img

        objects = []
        for ob in vdict.get("objects"):
            bbox = ob.get("bbox")
            x_min, y_min, x_max, y_max = bbox

            new_x_min = int(max(cx - (x_max - cx), 0))
            new_x_max = int(min(cx + (cx - x_min), img.shape[1] - 1))

            objects.append({
                "class_name": ob.get("class_name"),
                "class_id": ob.get("class_id"),
                "bbox": (new_x_min, y_min, new_x_max, y_max)
            })
        vdict["objects"] = objects
        return vdict

    @staticmethod
    def random_noise_augmentation(vdict, min_snr, max_snr):
        snr = (max_snr - min_snr) * np.random.random() + min_snr
        img = vdict.get("image")
        h, w, c = img.shape
        mask = np.random.choice([0, 1, 2], size=(h, w, 1), p=[snr, (1-snr)/2.0, (1-snr)/2.0])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 1] = 255
        img[mask == 2] = 0
        vdict["image"] = img
        return vdict

    def random_cropping_augmentation(self, vdict, min_crop_rate, max_crop_rate):
        crop_rate = (max_crop_rate - min_crop_rate) * np.random.random() + min_crop_rate
        img = vdict.get("image")

        new_height = int(img.shape[0] * crop_rate)
        new_width = int(img.shape[1] * crop_rate)

        top = int(np.random.random() * (img.shape[0] - new_height))
        bottom = int(min(top + new_height, img.shape[0]))

        left = int(np.random.random() * (img.shape[1] - new_width))
        right = int(min(left + new_width, img.shape[1]))

        vdict["image"] = img[top:bottom, left:right]

        objects = []
        for ob in vdict.get("objects"):
            bbox = ob.get("bbox")
            bbox = self.__cut_bbox(bbox, top, bottom, left, right)
            if bbox is None:
                continue
            objects.append({
                "class_name": ob.get("class_name"),
                "class_id": ob.get("class_id"),
                "bbox": bbox
            })
        vdict["objects"] = objects
        return vdict

    def random_rotation_augmentation(self, vdict, min_degree, max_degree):
        img = vdict.get("image")
        angle = (max_degree - min_degree) * np.random.random() + min_degree
        h, w, c = img.shape
        center = (h / 2.0, w / 2.0)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h))
        vdict["image"] = img

        objects = []
        for ob in vdict.get("objects"):
            bbox = ob.get("bbox")
            x_min, y_min, x_max, y_max = bbox

            p1 = np.array([x_min, y_min, 1], dtype=np.float32)
            p2 = np.array([x_min, y_max, 1], dtype=np.float32)
            p3 = np.array([x_max, y_min, 1], dtype=np.float32)
            p4 = np.array([x_max, y_max, 1], dtype=np.float32)

            p1 = np.dot(p1, matrix.T)
            p2 = np.dot(p2, matrix.T)
            p3 = np.dot(p3, matrix.T)
            p4 = np.dot(p4, matrix.T)

            x_min = int(min(p1[0], p2[0], p3[0], p4[0]))
            y_min = int(min(p1[1], p2[1], p3[1], p4[1]))
            x_max = int(max(p1[0], p2[0], p3[0], p4[0]))
            y_max = int(max(p1[1], p2[1], p3[1], p4[1]))

            bbox = self.__cut_bbox((x_min, y_min, x_max, y_max), 0, img.shape[0], 0, img.shape[1])
            if bbox is None:
                continue

            objects.append({
                "class_name": ob.get("class_name"),
                "class_id": ob.get("class_id"),
                "bbox": bbox
            })
        vdict["objects"] = objects
        return vdict

    def random_translation_augmentation(self, vdict, x_rate, y_rate):
        img = vdict.get("image")

        tx = ((x_rate[1] - x_rate[0]) * np.random.random() + x_rate[0]) * img.shape[1]
        ty = ((y_rate[1] - y_rate[0]) * np.random.random() + y_rate[0]) * img.shape[0]

        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        vdict["image"] = img

        objects = []
        for ob in vdict.get("objects"):
            bbox = ob.get("bbox")
            x_min, y_min, x_max, y_max = bbox

            x_min, y_min, x_max, y_max = int(x_min + tx), int(y_min + ty), int(x_max + tx), int(y_max + ty)

            bbox = self.__cut_bbox((x_min, y_min, x_max, y_max), 0, img.shape[0], 0, img.shape[1])
            if bbox is None:
                continue

            objects.append({
                "class_name": ob.get("class_name"),
                "class_id": ob.get("class_id"),
                "bbox": bbox
            })
        vdict["objects"] = objects
        return vdict

    def __cut_bbox(self, bbox, top, bottom, left, right):
        threshold = self.bbox_threshold
        x_min, y_min, x_max, y_max = bbox

        if x_min > right or x_max < left:
            return None
        if y_min > bottom or y_max < top:
            return None

        x_min, x_max, y_min, y_max = max(left, x_min), min(right, x_max), max(top, y_min), min(bottom, y_max)

        if x_max - x_min < threshold or y_max - y_min < threshold:
            return None

        bbox = (int(x_min - left), int(y_min - top), int(x_max - left), int(y_max - top))
        return bbox


def test_voc_aug(filename):
    import os
    import sys
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    sys.path.append(PROJECT_ROOT)

    from utils.dataset.voc import VOC_CLASS
    from utils.dataset.transfrom.voc2vdict import voc2vdict
    from utils.dataset.tools.viz_bbox import viz_vdict
    import matplotlib.pyplot as plt

    transform = voc2vdict()
    augmentation = voc_aug()

    xml_file_path = r"E:\dataset\PASCAL_VOC\VOC_2007_trainval\Annotations\{}.xml".format(filename)
    image_file_path = r"E:\dataset\PASCAL_VOC\VOC_2007_trainval\JPEGImages\{}.jpg".format(filename)
    vdict = transform(xml_file_path, image_file_path, VOC_CLASS)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("src")
    plt.imshow(vdict.get("image"))

    plt.subplot(2, 2, 3)
    plt.title("src_bbox")
    image_src_show = viz_vdict(vdict)
    plt.imshow(image_src_show)

    plt.subplot(2, 2, 2)
    plt.title("aug")
    vdict_aug = augmentation(vdict)
    plt.imshow(vdict_aug.get("image"))

    plt.subplot(2, 2, 4)
    plt.title("aug_bbox")
    image_aug_show = viz_vdict(vdict_aug)
    plt.imshow(image_aug_show)

    plt.show()


if __name__ == "__main__":
    test_voc_aug("001593")
