"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/YOLOv1-pytorch
"""

from torch.utils.data import DataLoader

from yolo.config import PROJECT_DIR
from yolo.utils.dataset.voc import VOC_DATASET
from yolo.utils.dataset.tools.voc_eval import Evaluator


def parse_dataset(dataset_cfg, class_list):
    """
    :param dataset_cfg: config dict
    :param class_list: A list contains dataset classes, for example:

        class_list = [
            "__background__",
            "cat",
            "dog",
            ...
        ]

    :return: torch.utils.data.Dataset
    """
    if dataset_cfg.get("type").lower() == "voc":
        dataset = VOC_DATASET(
            dataset_dir=dataset_cfg.get("dir-path").replace("{PROJECT_DIR}", PROJECT_DIR),
            class_list=class_list,
            txt_file_name=dataset_cfg.get("txt-file-name"),
            augmentation=dataset_cfg.get("augmentation")
        )
        return dataset


def parse_data_loader(dataloader_cfg, dataset):
    """
    :param dataloader_cfg: config dict
    :param dataset: torch.utils.data.Dataset
    :return: torch.utils.data.DataLoader
    """
    data_loader = DataLoader(
        dataset,
        batch_size=dataloader_cfg.get("batch-size"),
        shuffle=dataloader_cfg.get("shuffle"),
        drop_last=dataloader_cfg.get("drop-last")
    )
    return data_loader


def parse_evaluator(eval_dataset_cfg, class_list, cache_save_dir):
    """
    :param eval_dataset_cfg: config dict
    :param class_list: A list contains dataset classes
    :param cache_save_dir:
    :return: evaluator
    """
    evaluator = Evaluator(
        dataset_dir=eval_dataset_cfg.get("dir-path").replace("{PROJECT_DIR}", PROJECT_DIR),
        txt_file_name=eval_dataset_cfg.get("txt-file-name"),
        voc_class_list=class_list,
        cache_dir=cache_save_dir,
        use_07_metric=eval_dataset_cfg.get("use-07-metric")
    )
    return evaluator
