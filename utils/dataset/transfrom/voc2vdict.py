import os
import cv2
import xml.dom.minidom as xml


class voc2vdict:
    """
    A transformer which can convert VOC format data to "annotation" format dict.
    """
    def __init__(self):
        pass

    def __call__(self, xml_file_path, image_file_path, voc_class_list):
        """
        :param xml_file_path: The absolute path of your xml file.
        :param image_file_path: The absolute path of your image file with a suffix of ".jpg".
        :param voc_class_list: A list contains all of your VOC dataset classes, for example:

            voc_class_list = [
                __background__,     # Attention, always the first index.
                person,
                house,
                bus,
                ...
            ]

        :return: An "annotation" format dict. For example:

            annotation = {
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

        """
        # check files
        assert os.path.isfile(xml_file_path)
        assert os.path.isfile(image_file_path)

        # read image from disk
        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # initialize annotation
        annotation = dict()
        annotation["image"] = image

        # parse xml file
        xml_file = xml.parse(xml_file_path)
        xml_elements = xml_file.documentElement

        # filename
        annotation["filename"] = xml_elements.getElementsByTagName("filename")[0].firstChild.data.split(".")[0]

        # objects
        annotation["objects"] = list()
        all_object_elements = xml_elements.getElementsByTagName("object")
        for object_element in all_object_elements:
            object_class_name = object_element.getElementsByTagName("name")[0].firstChild.data
            assert object_class_name in voc_class_list
            object_class_id = voc_class_list.index(object_class_name)
            object_bbox_element = object_element.getElementsByTagName("bndbox")[0]
            object_bbox = \
                int(float(object_bbox_element.getElementsByTagName("xmin")[0].firstChild.data)), \
                int(float(object_bbox_element.getElementsByTagName("ymin")[0].firstChild.data)), \
                int(float(object_bbox_element.getElementsByTagName("xmax")[0].firstChild.data)), \
                int(float(object_bbox_element.getElementsByTagName("ymax")[0].firstChild.data))

            annotation["objects"].append({
                "class_name": object_class_name,
                "class_id": object_class_id,
                "bbox": object_bbox
            })
        return annotation

    def __repr__(self):
        return self.__class__.__name__ + '()'