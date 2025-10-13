import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv 

from ..utils import image_loader
from .object_detection import BaseObjectDetectionDataset
from .schemas import FAIREOObjectDetectionYoloSchema

'''
The NZ-Cattle dataset (New Zealand Cattle) is a remote sensing dataset for detecting cows from 
high resolution aerial images. It contains 655 images of 0.1m resolution. And a total of 29803 
cow instances.
'''

class NZCattleDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "New Zealand Cattle Detection"
    url = "https://zenodo.org/records/5908869"

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
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.png")) / 255.0
        image = image[:,:,:3]
        img_h, img_w = image.shape[:2]

        # annotation file
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.png.mask.0.txt")
        annot = open(annot_file_path,"r")
        boxes = []
        labels = []
        # annotations are points centered on the cows, we create a small bounding box around them
        for annotation in annot:
            elements = annotation.split(",")
            x_center = elements[0]
            y_center = elements[1]
            # bounding box
            xmin = int(x_center) - 5
            xmax = int(x_center) + 5
            
            ymin = int(y_center) - 5
            ymax = int(y_center) + 5

            if xmax<=img_w and ymax<=img_h and xmin>=0 and ymin >=0:
                label_value = 1
                labels.append(int(label_value))
                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}
        
        # image_id
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
                    split_list.append(lines[0]+','+lines[1])
            for i in ids:
                i_csv = 'cow_images/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        for img in ids:
            annot_file_path = os.path.join(annotation_dir, f"{img}.mask.0.txt")
            annot = open(annot_file_path,"r")
            annot = annot.readlines()
            for annotation in annot:
                label_value = 1
                annotations.append({"label": int(label_value)})

                labels = ["no-data","cow"]

    
            name = img[: img.rfind('.png')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["cow"], True)
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