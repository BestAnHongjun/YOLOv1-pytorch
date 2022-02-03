"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch
"""

import os
import sys
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(PROJECT_ROOT)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from yolo.utils.viz_bbox import viz_yolo_ground_truth
from yolo.utils.dataset.transfrom.voc2vdict import voc2vdict
from yolo.utils.dataset.transfrom.vdict2yolo import vdict2yolo_v1
from yolo.utils.dataset.tools.voc_aug import voc_aug


class VOC_DATASET(Dataset):
    """
    Apply to general VOC format dataset, to be more precise,
    the directory structure of the dataset should be like this below:

        dataset_dir
            |- Annotations
            |      |- 1.xml
            |      |- 2.xml
            |      |- ..
            |- ImageSets
            |      |- Main
            |          |- txt_file_name.txt
            |- JPEGImages
                   |- 1.jpg
                   |- 2.jpg
                   |- ..

    The dataset_dir and the txt_file_name should be put in as arguments of the
    constructed function.

    See docs of VOC_DATASET.__getitem__() to explore more about the return format.
    """

    def __init__(self,
                 dataset_dir,
                 txt_file_name,
                 class_list,
                 transform=vdict2yolo_v1(),
                 augmentation=False,
                 ):
        """
        :param dataset_dir: The absolute path to VOC format dataset directory.
        :param txt_file_name: The filename of the txt file without ".txt"
        :param class_list: A list contains all classes' name
        :param transform: Transform the default return format to satisfy your requirement.
        :param augmentation: Apply VOC augmentation
        """

        # check dataset_dir
        # assert os.path.isdir(dataset_dir)

        # check the txt file
        txt_file_path = os.path.join(
            dataset_dir,
            "ImageSets",
            "Main",
            "{file_name}.txt".format(file_name=txt_file_name)
        )
        # assert os.path.isfile(txt_file_path)

        # initialize attributes
        self.transform = transform
        self.class_list = class_list
        self.data = []
        self.augmentation = None

        # append data to data_list
        self.append_data(dataset_dir, txt_file_path)

        if augmentation:
            self.augmentation = voc_aug()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        :param index: index of the item
        :return: The default return format is called VOC-dict format, as same as vdict.
        For more precise, it is a python dict with structure like this:

            vdict = {
                "image": numpy.array([[[....]]]),   # Cv2 image Mat. (Shape:[h, w, 3], RGB format)
                "filename": 000048,                 # filename without suffix
                "objects": [{                       # A list of dicts representing b-boxes
                    "class_name": "house",
                    "class_id": 2,                  # index of the class name in arg: voc_class_list
                    "bbox": (x_min, y_min, x_max, y_max)
                }, {
                    ...
                }]
            }

        This format can not be loaded by torch.utils.data.DataLoader, but it is simple to apply voc-augmentation on it.
        So You should convert it to satisfy your requirement, for example, a Tensor, by input a "transform" to
        constructed function. By default, we use vdict2yolo_v1 transformer in module transform.vdict2yolo.
        """

        xml_file_path, image_file_path = self.data[index]
        voc2vdict_transform = voc2vdict()
        vdict = voc2vdict_transform(xml_file_path, image_file_path, self.class_list)

        if self.augmentation is not None:
            vdict = self.augmentation(vdict)

        return self.transform(vdict)

    def append_data(self, dataset_dir, txt_file_path):
        with open(txt_file_path, "r") as f:
            filename_list = f.read().strip().split()  # split by row
        print("Reading file {file_path}...".format(file_path=txt_file_path))

        for filename in filename_list:
            # filename must be a number!
            # assert filename.isdigit()

            image_path = os.path.join(
                dataset_dir,
                "JPEGImages",
                "{file_name}.jpg".format(file_name=filename)
            )
            xml_file_path = os.path.join(
                dataset_dir,
                "Annotations",
                "{file_name}.xml".format(file_name=filename)
            )

            # assert os.path.isfile(image_path)
            # assert os.path.isfile(xml_file_path)

            self.data.append((xml_file_path, image_path))


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from yolo.config import PROJECT_DIR

    with open(os.path.join(PROJECT_DIR, "tools", "config", "class.cfg.json"), "r") as json_file:
        class_list = json.load(json_file).get("class-list")

    voc_transform = vdict2yolo_v1()
    dataset = VOC_DATASET(
        r"E:\dataset\PASCAL_VOC\VOC_2007_test",
        txt_file_name="test",
        class_list=class_list,
        transform=voc_transform,
        augmentation=True
    )

    train_DataLoader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_id, (x, y) in enumerate(train_DataLoader):
        src_image, res_image = viz_yolo_ground_truth(x[0], y[0], class_list)

        plt.figure(figsize=(15, 10))

        plt.subplot(1, 2, 1)
        plt.title('data loader')
        plt.imshow(src_image)

        plt.subplot(1, 2, 2)
        plt.title("ground truth")
        plt.imshow(res_image)

        plt.show()
        exit()
