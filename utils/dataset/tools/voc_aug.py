"""
Under GPL
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
        4. Random Noisy augmentation
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

    def __call__(self, annotation):
        # Randomly HSV augment
        random_num = np.random.random()
        if random_num < self.hsv_pro:
            annotation = self.randomly_hsv_augmentation(annotation, self.h_gain, self.s_gain, self.v_gain)

        # Randomly crop image
        random_num = np.random.random()
        if random_num < self.crop_pro:
            annotation = self.randomly_crop_image(annotation, self.min_crop_rate, self.max_crop_rate)

        # Randomly flip
        annotation = self.randomly_flip_image(annotation, self.flip_pro)

        # Randomly noisy augment
        random_num = np.random.random()
        if random_num < self.noise_pro:
            annotation = self.randomly_add_noise(annotation, self.min_snr, self.max_snr)

        # Randomly rotation augment or translation augment
        random_num = np.random.random()
        if random_num < self.rotate_or_trans_pro:
            random_num = np.random.random()
            if random_num < 0.5:
                annotation = self.randomly_rotate_image(annotation, self.min_degree, self.max_degree)
            else:
                annotation = self.randomly_translation_image(annotation, self.x_trans_rate, self.y_trans_rate)

        return annotation

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def randomly_hsv_augmentation(annotation, h_gain, s_gain, v_gain):
        img = annotation.get("image")
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(np.uint8)
        aug_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        annotation["image"] = aug_img
        return annotation

    @staticmethod
    def randomly_flip_image(annotation, flip_pro):
        random_num = np.random.random()
        if random_num >= flip_pro:
            return annotation
        img = annotation.get("image")
        cx = img.shape[1] / 2.0
        img = cv2.flip(img, 1)
        annotation["image"] = img

        objects = []
        for ob in annotation.get("objects"):
            bbox = ob.get("bbox")
            x_min, y_min, x_max, y_max = bbox

            new_x_min = int(max(cx - (x_max - cx), 0))
            new_x_max = int(min(cx + (cx - x_min), img.shape[1] - 1))

            objects.append({
                "class_name": ob.get("class_name"),
                "class_id": ob.get("class_id"),
                "bbox": (new_x_min, y_min, new_x_max, y_max)
            })
        annotation["objects"] = objects
        return annotation

    @staticmethod
    def randomly_add_noise(annotation, min_snr, max_snr):
        snr = (max_snr - min_snr) * np.random.random() + min_snr
        img = annotation.get("image")
        h, w, c = img.shape
        mask = np.random.choice([0, 1, 2], size=(h, w, 1), p=[snr, (1-snr)/2.0, (1-snr)/2.0])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 1] = 255
        img[mask == 2] = 0
        annotation["image"] = img
        return annotation

    def randomly_crop_image(self, annotation, min_crop_rate, max_crop_rate):
        crop_rate = (max_crop_rate - min_crop_rate) * np.random.random() + min_crop_rate
        img = annotation.get("image")

        new_height = int(img.shape[0] * crop_rate)
        new_width = int(img.shape[1] * crop_rate)

        top = int(np.random.random() * (img.shape[0] - new_height))
        bottom = int(min(top + new_height, img.shape[0]))

        left = int(np.random.random() * (img.shape[1] - new_width))
        right = int(min(left + new_width, img.shape[1]))

        annotation["image"] = img[top:bottom, left:right]

        objects = []
        for ob in annotation.get("objects"):
            bbox = ob.get("bbox")
            bbox = self.__cut_bbox(bbox, top, bottom, left, right)
            if bbox is None:
                continue
            objects.append({
                "class_name": ob.get("class_name"),
                "class_id": ob.get("class_id"),
                "bbox": bbox
            })
        annotation["objects"] = objects
        return annotation

    def randomly_rotate_image(self, annotation, min_degree, max_degree):
        img = annotation.get("image")
        angle = (max_degree - min_degree) * np.random.random() + min_degree
        h, w, c = img.shape
        center = (h / 2.0, w / 2.0)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h))
        annotation["image"] = img

        objects = []
        for ob in annotation.get("objects"):
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
        annotation["objects"] = objects
        return annotation

    def randomly_translation_image(self, annotation, x_rate, y_rate):
        img = annotation.get("image")

        tx = ((x_rate[1] - x_rate[0]) * np.random.random() + x_rate[0]) * img.shape[1]
        ty = ((y_rate[1] - y_rate[0]) * np.random.random() + y_rate[0]) * img.shape[0]

        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        annotation["image"] = img

        objects = []
        for ob in annotation.get("objects"):
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
        annotation["objects"] = objects
        return annotation

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
    from voc import VOC_CLASS
    from transfrom.voc2vdict import voc2vdict
    from tools.viz_bbox import viz_annotation
    import matplotlib.pyplot as plt

    transform = voc2vdict()
    augmentation = voc_aug()

    xml_file_path = r"E:\dataset\PASCAL_VOC\VOC_2007_trainval\Annotations\{}.xml".format(filename)
    image_file_path = r"E:\dataset\PASCAL_VOC\VOC_2007_trainval\JPEGImages\{}.jpg".format(filename)
    annotation = transform(xml_file_path, image_file_path, VOC_CLASS)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("src")
    plt.imshow(annotation.get("image"))

    plt.subplot(2, 2, 3)
    plt.title("src_bbox")
    image_src_show = viz_annotation(annotation)
    plt.imshow(image_src_show)

    plt.subplot(2, 2, 2)
    plt.title("aug")
    annotation_aug = augmentation(annotation)
    plt.imshow(annotation_aug.get("image"))

    plt.subplot(2, 2, 4)
    plt.title("aug_bbox")
    image_aug_show = viz_annotation(annotation_aug)
    plt.imshow(image_aug_show)

    plt.show()


if __name__ == "__main__":
    test_voc_aug("001593")
