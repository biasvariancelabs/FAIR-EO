import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
HRPlanesv2 is a high-resolution satellite imagery dataset for aircraft detection. 
It contains 2120 very high-resolution Google Earth images, with a total of 14335 aircraft labeled.
'''

class HRPlanesv2Dataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "HRPlanesv2"
    url = "https://zenodo.org/records/7331974"

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
            elements = lines.split(' ')
            # bounding box
            xmin = round((float(elements[1])-(float(elements[3])/2))*img_w)
            xmax = round((float(elements[1])+(float(elements[3])/2))*img_w)

            ymin = round((float(elements[2])-(float(elements[4])/2))*img_h)
            ymax = round((float(elements[2])+(float(elements[4])/2))*img_h)

            if ymax > img_h:
                ymax = img_h
            if xmax > img_w:
                xmax = img_w
            if ymin < 0:
                ymin = 0
            if xmin < 0:
                xmin = 0

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

            labels = ["no-data","aircraft"]
            
            # box coordinates for txt files are extracted
            
            for annotation in annot:

                lines = annotation[:-1] #removes /n
                elements = lines.split(" ")
                # bounding box
                xmin = round((float(elements[1])-(float(elements[3])/2))*img_w)
                xmax = round((float(elements[1])+(float(elements[3])/2))*img_w)

                ymin = round((float(elements[2])-(float(elements[4])/2))*img_h)
                ymax = round((float(elements[2])+(float(elements[4])/2))*img_h)

                if xmax > xmin and ymax > ymin:
                    annotations.append({"label": 1})

        for img in ids:
            name = img[: img.rfind('.jpg')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["aircraft"], True)
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