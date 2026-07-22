import os
import torch
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from aitlas.utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
SWIM-Ship Wake Imagery Mass dataset is a benchmark dataset built for deep learning-based ship wake 
detection. The objects to be detected include the wakes of ships, ranging from small yacht to huge 
container ships, and the backgrounds include open seas, ports, straits, and canals. Ship wakes are 
annotated with oriented bounding boxes.
'''

class SWIMDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "SWIM"
    url = "https://www.kaggle.com/datasets/lilitopia/swimship-wake-imagery-mass"
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
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg"))/255

        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.xml")
        tree = ET.parse(annot_file_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall('object'):
            if obj.find('type').text == 'robndbox':
                robndbox = obj.find('robndbox')
                cx = float(robndbox.find('cx').text)
                cy = float(robndbox.find('cy').text)
                w = float(robndbox.find('w').text)
                h = float(robndbox.find('h').text)
                angle = float(robndbox.find('angle').text)
                # translate the oriented bbox to an axis-aligned bbox
                boxes.append(self.get_bbox(cx, cy, w, h, angle))
                labels.append(int(1)) # ship wake label value

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

        return self.apply_transformations(image, target)


    def load_dataset(self, data_dir, annotation_dir, csv_file):
        annotations = []
        labels = ["none", "ship wake"]

        data = list(set(os.path.splitext(f)[0] for f in os.listdir(annotation_dir) if f.endswith('.xml')))

        if csv_file:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                split_list = [line.strip() for line in lines]
            data = split_list
                
        for annot in os.listdir(annotation_dir):
            tree = ET.parse(os.path.join(annotation_dir,annot))
            root = tree.getroot()

            for obj in root.findall('object'):
                annotations.append({"label": int(1)})

        return labels, annotations, data
    

    def get_bbox(self, centerx, centery, sizew, sizeh, angle, max_h=768, max_w=768):
        """
        Translates an oriented bounding box to an axis-aligned bounding box
        """
        centerx, centery, sizeh, sizew, angle = int(centerx), int(centery), int(sizeh), int(sizew), float(angle)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        w_half = sizew / 2
        h_half = sizeh / 2
        corners = [
            [-w_half, -h_half],
            [w_half, -h_half],
            [w_half, h_half],
            [-w_half, h_half]
        ]

        rotated_corners = []
        for corner in corners:
            x_rot = corner[0] * cos_a - corner[1] * sin_a + centerx
            y_rot = corner[0] * sin_a + corner[1] * cos_a + centery
            rotated_corners.append((x_rot, y_rot))

        x_coords = [corner[0] for corner in rotated_corners]
        y_coords = [corner[1] for corner in rotated_corners]
        xmin, ymin = min(x_coords), min(y_coords)
        xmax, ymax = max(x_coords), max(y_coords)

        # ensure the bbox is inside the image patch
        xmin = max(xmin, 0)
        ymin = max(ymin,0)
        xmax = min(xmax,max_h)
        ymax = min(ymax,max_w)

        return [xmin, ymin, xmax, ymax]


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