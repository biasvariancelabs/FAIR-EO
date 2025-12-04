import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import torch

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionCocoSchema
from .object_detection import BaseObjectDetectionDataset

'''
This is a single class dataset consisting of tiles of satellite imagery labeled 
with potential 'targets'. It was designed to improve the search for missing persons.
'''

class SearchAndRescueDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionCocoSchema
    name = "Search And Rescue dataset"
    url = "https://github.com/michaelthoreau/SearchAndRescueNet"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir
        )


    def __getitem__(self, index):
        img_name = self.data[index][0]
        img_id = self.data[index][1]
        image = image_loader(os.path.join(self.data_dir, img_name))/255
        img_h, img_w = image.shape[:2]

        boxes = []
        labels = []

        with open(self.annotation_dir, "r") as f:
            annot = json.load(f)

        for ann in annot["annotations"]:
            if img_id == ann["image_id"]:
                xmin, ymin, w, h = ann["bbox"]
                xmax = xmin + w
                ymax = ymin + h

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(ann["category_id"]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
        }

        image, target = self.apply_transformations(image, target)
         
        return image, target


    def load_dataset(self, data_dir, annotation_dir):
        annotations = []
        data = []
        labels = ["no-data","Target"]

        with open(annotation_dir, "r") as f:
            annot = json.load(f)

        for images in annot["images"]:
            file_name = images["file_name"]
            img_id = images["id"]
            data.append([file_name, img_id])

        for ann in annot["annotations"]:
            category_id = ann["category_id"]
            annotations.append({"label": int(category_id)})

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df["label"].value_counts().reset_index()
        df_count.columns = ["Label", "Count"]
        df_count["Label"][0] = 'Target'
        return df_count

    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig