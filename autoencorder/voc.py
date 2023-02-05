import os.path as osp
# import os
#  p66
def make_filepath_list(rootpath):
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()
        # %s の置き換え
        img_path = (imgpath_template % file_id)
        # %s の置き換え
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list
# p66

# p71
import xml.etree.ElementTree as ElementTree
import numpy as np

# xml の処理
class GetBBoxAndLabel(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, width, height):
        annotation = []

        xml = ElementTree.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)

            if difficult == 1:
                continue
            bndbox = []

            name = obj.find('name').text.lower().strip()

            bbox = obj.find('bndbox')

            grid = ['xmin', 'ymin', 'xmax', 'ymax']

            for gr in (grid):
                axis_value = int(bbox.find(gr).text) - 1

                if gr == 'xmin' or gr == 'xmax':
                    axis_value /= width
                else:
                    axis_value /= height

                bndbox.append(axis_value)

            label_idx = self.classes.index(name)

            bndbox.append(label_idx)

            annotation += [bndbox]

        return np.array(annotation)
# p73


# augmentations.py からの前処理をするクラスをインポート
from augmentations import Compose, ConvertFormInts, ToAsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

class DataTransform(object):
    def __init__(self, input_size, color_mean):
        self.transform = {
            'train': Compose([
                ConvertFormInts(),
                ToAsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean),
            ]),

            'val': Compose([
                ConvertFormInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.transform[phase](img, boxes, labels)
