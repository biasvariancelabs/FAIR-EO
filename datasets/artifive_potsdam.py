import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import torch

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionCocoSchema
from .object_detection import BaseObjectDetectionDataset

'''
A benchmark dataset for vehicle detection in aerial imagery. Combining real data from the 
Potsdam dataset with artificially generated vehicle images, to study the impact of artificial 
data enrichment on machine learning performance.
'''

class ArtifiVePotsdamDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionCocoSchema
    name = "ArtifiVe-Potsdam"
    url = "https://rs.ipb.uni-bonn.de/data/artifive-potsdam/index.html"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.selection = self.config.selection
        self.labels, self.annot, self.data = self.load_dataset(self.annotation_dir, self.selection)

    def __getitem__(self, index):
        for i in np.arange(2):
            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg"))/255
            img_h, img_w = image.shape[:2]

            boxes = []
            labels = []

            with open(self.annotation_dir, "r") as f:
                annot_json = json.load(f)

            for item in annot_json:
                img_name_json = item['image_filename'].split(".")[0]
                if img_name_json != img_name:
                    continue

                annotations = item['annotations']['polygons']
                for annotation in annotations:
                    coords = annotation['coordinates']
                    coords = coords[0]
                    xs = [point[0] for point in coords]
                    ys = [point[1] for point in coords]

                    ymin=min(ys)
                    xmin=min(xs)
                    ymax=max(ys)
                    xmax=max(xs)

                    if ymin < 0:
                        ymin = 0
                    if xmin < 0:
                        xmin = 0
                    if ymax > img_h:
                        ymax = img_h
                    if xmax > img_w:
                        xmax = img_w
                    if xmax > xmin and ymax > ymin:
                        labels.append(int(1))
                        boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

            image, target = self.apply_transformations(image, target)

            if len(target['boxes']) != len(target['labels']):
                index = index + 1
            else:
                break

        return image, target


    def load_dataset(self, annotation_dir, selection):

        data=[]
        annot=[]
        labels=["no-data", "car"]

        with open(self.annotation_dir, "r") as f:
            annot_json = json.load(f)

        for item in annot_json:
            annotations = item['annotations']['labels']
            if len(annotations) == 0:
                if self.selection == "all":
                    img_name = item['image_filename'].split(".")[0]
                    data.append(img_name)     
            else:
                for vehicle in annotations:
                    annot.append({"label": int(1)})    
                img_name = item['image_filename'].split(".")[0]
                data.append(img_name)         

        return labels, annot, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", "Car", True)
        df_count.columns = ["Label", "Count"]

        return df_count

    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig