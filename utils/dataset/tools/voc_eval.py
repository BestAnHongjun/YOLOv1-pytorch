"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact at an.hongjun@foxmail.com
Github: https://github.com/AnHongjun001/YOLOv1-pytorch

Reference:
[1]Code in methods Evaluator.__parse_rec, Evaluator.__voc_ap and Evaluator.__voc_eval refers to Bharath Hariharan's
creation, see more detail at https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py,
which is under MIT License.
[2]Code in method Evaluator.eval refers to code by sihaiyinan at CSDN platform,
see more detail at https://blog.csdn.net/sihaiyinan/article/details/89417963,
which is under CC 4.0 BY-SA License.
"""

import xml.etree.ElementTree as ET
import os
import _pickle as cPickle
import numpy as np


class Evaluator:
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
    """

    def __init__(self, dataset_dir, txt_file_name, voc_class_list,
                 cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "voc_cache"),
                 use_07_metric=False
                 ):
        """
        :param dataset_dir: The absolute path to VOC format dataset directory.
        :param txt_file_name: The filename of the txt file without ".txt"
        :param voc_class_list: A list contains all classes' name
        :param cache_dir: We will put some cache file in this dir.
        :param use_07_metric: If you want to evaluate VOC-2007-test dataset, set this arg to True
        """

        # check dataset_dir
        self.dataset_dir = dataset_dir
        assert os.path.isdir(dataset_dir)

        # check the txt file
        self.txt_file_path = os.path.join(
            dataset_dir,
            "ImageSets",
            "Main",
            "{file_name}.txt".format(file_name=txt_file_name)
        )
        assert os.path.isfile(self.txt_file_path)

        self.voc_class_list = voc_class_list

        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(cache_dir)

        self.det_dir = os.path.join(self.cache_dir, "results")
        if not os.path.exists(self.det_dir):
            os.mkdir(self.det_dir)

        self.det_file_paths = []
        for cls_name in self.voc_class_list:
            if cls_name == "__background__":
                self.det_file_paths.append("")
                continue
            det_file_path = os.path.join(self.det_dir, "{}.txt".format(cls_name))
            print("initializing {}..".format(det_file_path))
            self.det_file_paths.append(det_file_path)

        # initialize attributes
        self.cache_file_path = os.path.join(self.cache_dir, 'annots.pkl')
        self.anno_path = os.path.join(self.dataset_dir, "Annotations", "{:s}.xml")
        self.use_07_metric = use_07_metric

        self.clear_results()

    def clear_results(self):
        for det_file_path in self.det_file_paths:
            if det_file_path == "":
                continue
            with open(det_file_path, "w") as f:
                f.write("")

    def add_result(self, cls_id, filename, pro, x_min, y_min, x_max, y_max):
        with open(self.det_file_paths[cls_id + 1], "a") as f:
            f.write(f"{filename} {pro} {x_min} {y_min} {x_max} {y_max}\n".format(
                filename=filename,
                pro=pro,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max
            ))

    def eval(self):
        aps = []
        recs = []
        precs = []
        for i, cls in enumerate(self.voc_class_list):
            if cls == '__background__':
                continue

            det_file_path = self.det_file_paths[i]

            rec, prec, ap = self.__voc_eval(  # 调用voc_eval.py计算cls类的recall precision ap
                det_path=det_file_path,
                anno_path=self.anno_path,
                class_name=cls,
                ovthresh=0,
            )

            aps.append(ap)
            recs.append(rec[-1])
            precs.append(prec[-1])

        m_ap = np.mean(aps)
        return m_ap, aps, recs, precs

    def __voc_eval(self,
                   det_path,
                   anno_path,
                   class_name,
                   ovthresh=0.5,
                   ):
        """
        Top level function that does the PASCAL VOC evaluation.

        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt
        cachefile = os.path.join(self.cache_dir, 'annots.pkl')
        # read list of images
        with open(self.txt_file_path, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.__parse_rec(anno_path.format(imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                        i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                cPickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = cPickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == class_name]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        detfile = os.path.join(self.det_dir, "{}.txt".format(class_name))
        # detfile = det_path.format(class_name)
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        # print(BB)

        if not len(BB):
            return [0], [0], 0

        BB = BB[sorted_ind, :]

        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.__voc_ap(rec, prec, self.use_07_metric)

        return rec, prec, ap

    @staticmethod
    def __parse_rec(filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects

    @staticmethod
    def __voc_ap(rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
