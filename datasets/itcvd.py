import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv
import scipy.io

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset

'''
ITCVD dataset is a large-scale, well annotated (all vehicles in a scene are annotated) and challenging 
vehicle detection dataset. The images were taken from an airplane platform at 330m height over Enschede 
in The Netherlands.
'''

class ITCVDDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "ITCVD"
    url = "https://hyper.ai/cn/datasets/5419"

    # labels: 0 index is reserved for background
    labels = [None]

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
        img_name = self.data[index]

        image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg")) / 255.0
        img_h, img_w = image.shape[:2]

        # annotation file
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.mat")
        mat = scipy.io.loadmat(annot_file_path)
        
        boxes = []
        labels = []

        # box coordinates for txt files are extracted
        for annotation in mat[f"x{img_name}"]:

            # bounding box
            xmin = annotation[0]
            ymin = annotation[1]

            xmax = annotation[2]
            ymax = annotation[3]

            if xmax > xmin and ymax > ymin:
                labels.append(1)
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

    def load_dataset(self, data_dir, annotation_dir, csv_file):
        labels = []
        annotations = []
        data = []

        for img in os.listdir(data_dir):
            annot_file_path = os.path.join(annotation_dir, img[: img.rfind('.jpg')]+'.mat') 
            mat = scipy.io.loadmat(annot_file_path)
            
            img_name = img[: img.rfind('.jpg')]

            image_file_path = os.path.join(data_dir, f"{img_name}.jpg")
            image = cv2.imread(image_file_path)
            img_h, img_w = image.shape[:2]

            labels = ["no-data","vehicle"]
            
            for annotation in mat[f"x{img_name}"]:

                # bounding box
                xmin = annotation[0]
                ymin = annotation[1]

                xmax = annotation[2]
                ymax = annotation[3]

                if xmax > xmin and ymax > ymin:
                    annotations.append({"label": 1})

        for img in os.listdir(data_dir):
            name = img[: img.rfind('.jpg')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["vehicle"], True)
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