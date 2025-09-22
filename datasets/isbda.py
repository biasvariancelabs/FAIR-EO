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
User-generated aerial video frames from natural disasters annotated with instance-level building damage
masks and bounding boxes. This dataset cointains 1030 images sampled from 10 aerial videos, which were
recorded after Hurricane Harvey in 2017, Hurricane Micheal and Hurricane Florence in 2018 and other three
tornadoes (EF-2 or EF-3) in 2017, 2018 and 2019, respectively.
'''

class ISBDADataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionCocoSchema
    name = "ISBDA dataset"
    url = "https://github.com/zgzxy001/MSNET?tab=readme-ov-file"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir
        )


    def __getitem__(self, index):
        for i in np.arange(10):
            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg"))/255
            img_h, img_w = image.shape[:2]

            with open(self.annotation_dir, "r") as f:
                annot_json = json.load(f)

            img_id= None
            for img in annot_json["images"]:
                if os.path.splitext(img["file_name"])[0] == img_name:
                    img_id = img["id"]
                    break

            boxes = []
            labels = []

            for ann in annot_json["annotations"]:
                if ann["image_id"] != img_id:
                    continue

                xmin, ymin, w, h = ann["house_bbox"]
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
         
            if len(target['boxes'])==0:
                index = index+1
            else: 
                break

        return image, target


    def load_dataset(self, data_dir, annotation_dir):
        annotations = []
        data = []
        labels = ["No-damage", "Slight", "Severe", "Debris"]

        with open(annotation_dir, "r") as f:
            annot = json.load(f)

        image_id_to_filename = {img["id"]: os.path.splitext(img["file_name"])[0] for img in annot["images"]}

        img_ann_map = {fn: [] for fn in image_id_to_filename.values()}

        for ann in annot["annotations"]:
            img_id = ann["image_id"]
            category_id = ann["category_id"]
            file_stem = image_id_to_filename[img_id]

            img_ann_map[file_stem].append({"label": int(category_id)})

        data = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".jpg")]

        annotations = [img_ann_map.get(img, []) for img in data]

        return labels, annotations, data

    def data_distribution_table(self):

        flat_anns = [ann["label"] for sample in self.annotations for ann in sample]
        df = pd.DataFrame(flat_anns, columns=["label"])
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