import os
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
The dataset was created to use deep learning to detect automatically the number, size and 
type of aircrafts present on the site. In turn, this can provide information about the activity 
of any airport. The dataset contains 103 images accompanied by aircraft bounding boxes.
'''

class AirbusAircraftDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "Airbus Aircraft Detection"
    url = "https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset"

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
        annot_file_path = self.annotation_dir
        annot = open(annot_file_path,"r")
        annot = annot.readlines()[1 :] #ignore the first line (column names)
        boxes = []
        labels = []

        for annotation in annot:
            elements = annotation.split(",")
            if elements[1]==img_name+'.jpg':
            
                # bounding box
                xmin = int("".join([ele for ele in elements[2] if ele.isdigit()]))
                ymin = int("".join([ele for ele in elements[3] if ele.isdigit()]))

                xmax = int("".join([ele for ele in elements[6] if ele.isdigit()]))
                ymax = int("".join([ele for ele in elements[7] if ele.isdigit()]))

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > img_w:
                    xmax = img_w
                if ymax > img_h:
                    ymax = img_h

                if xmax > xmin and ymax > ymin:
                    if elements[12][: -1] == 'Airplane':
                        label_value = 1
                    else:
                        label_value = 2
                    labels.append(int(label_value))
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
            annot_file_path = annotation_dir
            annot = open(annot_file_path,"r")
            annot = annot.readlines()
            for annotation in annot:
                elements = annotation.split(",")
                if elements[1]==img:
                    # bounding box
                    xmin = int("".join([ele for ele in elements[2] if ele.isdigit()]))
                    ymin = int("".join([ele for ele in elements[3] if ele.isdigit()]))

                    xmax = int("".join([ele for ele in elements[6] if ele.isdigit()]))
                    ymax = int("".join([ele for ele in elements[7] if ele.isdigit()]))

                    if xmax > xmin and ymax > ymin:
                        if elements[12][: -1] == 'Airplane':
                            label_value = 1
                        else:
                            label_value = 2
                        annotations.append({"label": int(label_value)})

                    labels = ["no-data","airplane","truncated airplane"]

    
            name = img[: img.rfind('.jpg')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["airplane", "truncated airplane"], True)
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