import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import torch
import csv
import numpy as np

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
AU-AIR is a multi-modal dataset collected from a UAV platform, containing over 32K images with 
bounding box annotations for 8 object categories (car, van, truck, human, trailer, bus, motorbike, 
and bicycle). It includes RGB images, IMU data, and flight metadata to support deep learning-based 
perception tasks for traffic surveillance from low altitude.
'''

class AUAIRDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "AU-AIR dataset"
    url = "https://bozcani.github.io/auairdataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file
        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )


    def __getitem__(self, index):
        for i in np.arange(10):
            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg"))/255
            img_h, img_w = image.shape[:2]

            with open(self.annotation_dir, "r") as f:
                annot_json = json.load(f)["annotations"]

            boxes = []
            labels = []

            for img in annot_json:
                if img_name == os.path.splitext(img["image_name"])[0]:

                    for bbox in img["bbox"]:

                        ymin = bbox["top"]
                        xmin = bbox["left"]
                        w = bbox["width"]
                        h = bbox["height"]

                        xmax = xmin + w
                        ymax = ymin + h


                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(img_w, xmax)
                        ymax = min(img_h, ymax)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(int(bbox["class"]))
                    break

            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "iscrowd": iscrowd,
            }
            image, target = self.apply_transformations(image, target)

            if len(target['boxes'])==0:
                index = index +1
            elif len(target['boxes']) != len(target['labels']):
                index = index +1
            else:
                break

        return image, target


    def load_dataset(self, data_dir, annotation_dir, csv_file):
        annotations = []
        data = []
        labels = ['Human', 'Car', 'Truck', 'Van', 'Motorbike', 'Bicycle', 'Bus', 'Trailer']

        split_list = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])

        with open(annotation_dir, "r") as f:
            annot = json.load(f)["annotations"]

        for img in annot:
            img_name = img["image_name"]
            img_name = os.path.splitext(img_name)[0]
            img_name_csv = 'images/'+img_name+'.jpg'

            if csv_file:
                if img_name_csv in split_list:
                    data.append(img_name)
            else:
                data.append(img_name)

            for bbox in img["bbox"]:
                category_id = bbox["class"]
                annotations.append(category_id)

        return labels, annotations, data

    def data_distribution_table(self):

        df = pd.DataFrame(self.annotations, columns=["label"])
        df_count = df["label"].value_counts().reset_index()
        df_count.columns = ["Label", "Count"]
        df_count["Label"] = df_count["Label"].apply(lambda i: self.labels[i])

        return df_count


    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig