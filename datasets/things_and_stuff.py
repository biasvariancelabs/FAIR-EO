import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
This dataset was used to develop the things and stuff (TAS) context model.
'''

class ThingsAndStuffDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "Things and stuff"
    url = "https://ai.stanford.edu/~gaheitz/Research/TAS/"

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir

        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.png")) / 255.0
        img_h, img_w = image.shape[:2]

        # annotation file
        boxes = []
        labels = []
        for label_name in os.listdir(self.annotation_dir):

            annot_file_path = os.path.join(self.annotation_dir, label_name, f"{img_name}.gt")
            annot = open(annot_file_path, "r")
            annot = annot.readlines()

            # box coordinates for txt files are extracted
            for annotation in annot:
                lines = annotation[0 :-1] 
                elements = lines.split()
                # bounding box
                xmin = round(float(elements[0]))
                ymin = round(float(elements[1]))

                xmax = round(float(elements[2])) 
                ymax = round(float(elements[3])) 

                if ymax > img_h:
                    ymax = img_h
                if xmax > img_w:
                    xmax = img_w
                if ymin < 0:
                    ymin = 0
                if xmin < 0:
                    xmin = 0

                if xmax > xmin and ymax > ymin:
                    labels.append(self.labels.index(label_name))
                    boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 

        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def load_dataset(self, data_dir, annotation_dir):
        labels = []
        annotations = []
        data = []
        for label_name in os.listdir(annotation_dir):
            for img in os.listdir(annotation_dir + label_name):
                annot_file_path = os.path.join(annotation_dir, label_name, img) 
                annot = open(annot_file_path, "r")
                annot = annot.readlines()
                
                img_name = img[: img.rfind('.gt')]
                image_file_path = os.path.join(data_dir, f"{img_name}.png")

                labels = ["no-data","VOCbicycles", "VOCcars", "VOCmotorbikes", "VOCpeople"]
                
                # box coordinates for txt files are extracted
                
                for annotation in annot:

                    lines = annotation[0 :-1]
                    elements = lines.split()
                    # bounding box
                    xmin = round(float(elements[0]))
                    ymin = round(float(elements[1]))

                    xmax = round(float(elements[2])) 
                    ymax = round(float(elements[3])) 

                    if xmax > xmin and ymax > ymin:
                        annotations.append({"label": label_name})

        for img in os.listdir(data_dir):
            name = img[: img.rfind('.png')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["VOCbicycles", "VOCcars", "VOCmotorbikes", "VOCpeople"], True)
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