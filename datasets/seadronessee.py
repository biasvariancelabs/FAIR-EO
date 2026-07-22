import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import torch

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionCocoSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
SeaDronesSee is a large-scale data set aimed at helping develop systems for Search
and Rescue (SAR) using Unmanned Aerial Vehicles (UAVs) in maritime scenarios. Building
highly complex autonomous UAV/drone systems that aid in SAR missions requires robust
computer vision algorithms to detect and track objects or persons of interest. This
data set provides three sets of tracks: object detection, single-object tracking and
multi-object tracking. Each track consists of its own fully labeled data set and for
most there is a leaderboard.
'''

class SeaDronesSeeDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionCocoSchema
    name = "SeaDronesSee dataset"
    url = "https://seadronessee.cs.uni-tuebingen.de/home"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.selection = self.config.selection
        self.labels, self.annotations, self.data = self.load_dataset(self.annotation_dir)

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg"))/255
        img_h, img_w = image.shape[:2]

        boxes = []
        labels = []

        with open(self.annotation_dir, "r") as f:
            annot_json = json.load(f)

        for annotation in annot_json["annotations"]:
            if img_name != annotation["image_id"]:
                continue

            category = annotation["category_id"]

            xmin, ymin, w, h = annotation["bbox"]
            xmax = xmin + w
            ymax = ymin + h

            if ymin < 0:
                ymin = 0
            if xmin < 0:
                xmin = 0
            if ymax > img_h:
                ymax = img_h
            if xmax > img_w:
                xmax = img_w
            if xmax > xmin and ymax > ymin:
                labels.append(int(category))
                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if len(boxes) == 0 or len(labels) == 0:
            if self.selection != "all":
                return None

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

        return self.apply_transformations(image, target)


    def load_dataset(self, annotation_dir):

        data = []
        annot = []
        # supercatagories: person, boat, object
        labels=["ignored","swimmer", "boat",  "jetski", "life_saving_appliances", "buoy"]

        with open(self.annotation_dir, "r") as f:
            annot_json = json.load(f)

        for img in annot_json["images"]:
            img_name = img["id"]
            data.append(img_name)

        for annot_item in annot_json["annotations"]:
            category = annot_item["category_id"]
            annot.append({"label": int(category)})

        return labels, annot, data

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