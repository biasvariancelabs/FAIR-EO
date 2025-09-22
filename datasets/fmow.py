import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ..utils import image_loader
from .object_detection import BaseObjectDetectionDataset
from .schemas import FAIREOObjectDetectionCocoSchema


'''
The dataset aims to inspire the development of machine learning models capable of predicting the functional 
purpose of buildings and land use from temporal sequences of satellite images and a rich set of metadata 
features. It contains more than 1 million images and 63 classes.
'''

class FMOWDataset(BaseObjectDetectionDataset):

    schema = FAIREOObjectDetectionCocoSchema
    name = 'FMOW dataset'

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the config
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir

        # load the data
        self.labels, self.data, self.annotations, self.file_names = self.load_dataset(
            self.data_dir, self.annotation_dir
        )

    def __getitem__(self, index):
        """

        :param index: Index
        :type index: int
        :return: tuple where target is a dictionary where target is index of the target class
        :rtype: tuple of (image, target)

        """
        img_data = self.file_names[index]
        image = image_loader(os.path.join(self.data_dir, img_data)) / 255.0

        # annotation file
        name = img_data[: img_data.rfind('.jpg')]
        annotations_file_path = os.path.join(self.annotation_dir,name+'.json')
        annotations = json.load(open(annotations_file_path, "r"))
        
        boxes = []
        labels = []
        image_id = []
        filename = annotations['img_filename']

        for annotation in annotations["bounding_boxes"]:
            labels.append(self.labels.index(annotation["category"])) 
            image_id.append(filename)

            bbox = annotation["box"]

            if len(bbox) > 0:
                # bounding box
                xmin = bbox[0]
                xmax = bbox[0] + bbox[2]

                ymin = bbox[1]
                ymax = bbox[1] + bbox[3]

                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return self.apply_transformations(image, target)
        
    def load_dataset(self, data_dir=None, annotation_dir=None):
        labels = []
        annotations = []
        data = []
        file_names = []
        for json_file in os.listdir(annotation_dir):
            coco = json.load(open(annotation_dir+json_file, "r"))

            # read labels
            #labels = [None]  # add none for background
            labels += [y["category"] for y in sorted(coco["bounding_boxes"], key=lambda x: x["category"])]
            # create data
            filename = coco['img_filename']
            dataf = [
                {
                    "file_name": coco["img_filename"],
                    **{
                        "annotations": [],
                        **{"file_name": coco["img_filename"]},
                    },
                }
            ]
            data_inverted = {x["file_name"]: i for i, x in enumerate(dataf)}
            annotationsf = coco["bounding_boxes"]

            # create index and annotations
            for annotation in annotationsf:
                bbox = []
                for coor in annotation["box"]:
                    bbox.append(max(coor, 0))
                annotation["bbox"] = bbox
                key = data_inverted[filename]
                dataf[key]["annotations"].append(annotation)
                annotations.append(annotation)

            data.append(dataf)
            file_names.append(coco["img_filename"])

        labels = list(set(labels))

        return labels, data, annotations, file_names

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df = df.drop(['bbox','box','visible'], axis=1)        
        df_count = df.groupby("category").count()
        df_count = df_count.iloc[:, ::-1].reset_index()
        df_count.columns = ["Label", "Count"]
        
        return df_count

    def data_distribution_barchart(self):
        df_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=df_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig

    def show_samples(self):
        df = pd.DataFrame(self.annotations)
        return df.head(20)