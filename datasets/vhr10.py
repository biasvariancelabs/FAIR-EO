import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset

'''
This very-high-resolution (VHR) remote sensing image dataset was constructed by Dr. Gong Cheng et al for 
object detection tasks. These images were cropped from Google Earth and Vaihingen data set and then manually 
annotated by experts.
'''

class VHR10Dataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "VHR-10"
    url = "https://drive.google.com/file/d/1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE/view"

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
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.txt")
        annot = open(annot_file_path, "r")
        annot = annot.readlines()
        boxes = []
        labels = []

        # box coordinates for txt files are extracted
        for annotation in annot:
            lines = annotation[:-1] #removes /n
            elements = lines.split(',')
            elements[0] = elements[0].translate({ord('('): None}).strip()
            elements[1] = elements[1].translate({ord(')'): None}).strip()
            elements[2] = elements[2].translate({ord('('): None}).strip()
            elements[3] = elements[3].translate({ord(')'): None}).strip()
            elements[4] = elements[4].translate({ord(' '): None}).strip()
            # bounding box
            xmin = int(elements[0])
            ymin = int(elements[1])

            xmax = int(elements[2])
            ymax = int(elements[3])

            if xmax > xmin and ymax > ymin:
                labels.append(int(elements[4]))
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

        ids = os.listdir(data_dir)
        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'images/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        for img in ids:
            annot_file_path = os.path.join(annotation_dir, img[: img.rfind('.jpg')]+'.txt') 
            annot = open(annot_file_path, "r")
            annot = annot.readlines()
            
            img_name = img[: img.rfind('.jpg')]
            image_file_path = os.path.join(data_dir, f"{img_name}.jpg")
            image = cv2.imread(image_file_path)
            img_h, img_w = image.shape[:2]

            labels = ["no-data","airplane","storage tank","ship","baseball diamond","tennis court",
                      "basketball court","ground track field","harbor","bridge","vehicle"]
            
            # box coordinates for txt files are extracted
            
            for annotation in annot:

                lines = annotation[:-1] #removes /n
                elements = lines.split(",")
                elements[0] = elements[0].translate({ord('('): None}).strip()
                elements[1] = elements[1].translate({ord(')'): None}).strip()
                elements[2] = elements[2].translate({ord('('): None}).strip()
                elements[3] = elements[3].translate({ord(')'): None}).strip()
                elements[4] = elements[4].translate({ord(' '): None}).strip()
                # bounding box
                xmin = elements[0]
                ymin = elements[1]

                xmax = elements[2]
                ymax = elements[3]

                if xmax > xmin and ymax > ymin:
                    annotations.append({"label": int(elements[4])})

        for img in ids:
            name = img[: img.rfind('.jpg')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["airplane","storage tank","ship","baseball diamond","tennis court",
                                     "basketball court","ground track field","harbor","bridge","vehicle"], True)
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