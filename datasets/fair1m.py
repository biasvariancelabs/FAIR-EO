import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xml.etree.ElementTree as ET
import csv

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
FAIR1M is a benchmark dataset for fine-grained object recognition in high resolution remote sensing imagery. 
It contains more than 15000 images and 1 milion instances. All instances are annotated with respect to 5 
categories and 37 sub-categories using oriented bounding boxes.
'''

class FAIR1MDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "FAIR1M"
    url = "https://www.gaofen-challenge.com/benchmark"
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
        for i in np.arange(10):
            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}.tif"))[:,:,:3]/255
            img_h, img_w = image.shape[:2]

            #annotation file
            annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.xml")
            tree = ET.parse(annot_file_path)
            root = tree.getroot()

            boxes = []
            labels = []
            for obj in root[3]:
                points = [[float(x) for x in point.text.split(',')] for point in obj.find('.//points')]
                xs, ys = zip(*points) if points else ([], [])

                # Get bounding box from min/max coordinates
                xmin, xmax = max(0, min(xs)), min(img_w, max(xs))
                ymin, ymax = max(0, min(ys)), min(img_h, max(ys))

                if xmax > xmin and ymax > ymin:
                    category = obj.find('.//name').text
                    value = next((i for i, label in enumerate(self.labels) if label == category), 0)
                    labels.append(int(value))
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

            image, target = self.apply_transformations(image, target)

            if len(target["boxes"]) != len(target["labels"]):
                index = index + 1
            else:
                break

        return image, target


    def load_dataset(self, data_dir, annotation_dir, csv_file):
        annotations = []
        split_list = []
        data = []

        labels = ["no_data", "Boeing737", "Boeing747", "Boeing777", "Boeing787", "C919", "A220", "A321", "A330",
                   "A350", "ARJ21", "other-airplane", "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", 
                   "Engineering Ship","Liquid Cargo Ship", "Dry Cargo Ship", "Warship", "other-ship", "Small Car",
                   "Bus", "Cargo Truck", "Dump Truck", "Van", "Trailer", "Tractor", "Excavator", "Truck Tractor", 
                   "other-vehicle", "Basketball Court", "Tennis Court", "Football Field", "Baseball Field", 
                   "Intersection", "Roundabout", "Bridge"]

        ids = list(set(os.path.splitext(f)[0] for f in os.listdir(annotation_dir) if f.endswith('.xml')))

        for annot in os.listdir(annotation_dir):

            tree = ET.parse(os.path.join(annotation_dir,annot))
            root = tree.getroot()

            if csv_file: 
                with open(csv_file, mode ='r')as file:
                    csvFile = csv.reader(file)
                    for lines in csvFile:
                        split_list.append(lines[0])

                name = 'images/' + root[0].find('./filename').text

                if name in split_list:
                    data.append(name[7 : -4])
                    for obj in root[3]:
                        category = obj.find('.//name').text
                        for i, label in enumerate(labels):
                            if label == category:
                                annotations.append({"label": int(i)})

            else:
                data = ids
                for obj in root[3]:
                    category = obj.find('.//name').text
                    for i, label in enumerate(labels):
                        if label == category:
                            annotations.append({"label": int(i)})
                            
        return labels, annotations, data


    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").size().reset_index(name="Count")
        
        # Map numeric labels to their string names
        label_names = {i: name for i, name in enumerate(self.labels)}
        df_count["Label"] = df_count["label"].map(label_names)
        df_count = df_count[["Label", "Count"]]

        return df_count


    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig