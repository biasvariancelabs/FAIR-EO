import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import json
import numpy as np
import xml.etree.ElementTree as ET

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset

'''
RarePlanes is a unique computer vision dataset combining synthetic and real satellite imagery of aircraft. 
It consists of 253 Maxar WorldView-3 satellite scenes, paired with 50000 synthetic satellite images generated 
using AI.Reverie's simulation platform.
'''

class RarePlanesDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "RarePlanes"
    url = "https://www.iqt.org/library/the-rareplanes-dataset"
    labels = ["None", "Airplane"]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.annotations, self.ids = self.load_dataset(
            self.data_dir, self.annotation_dir
        )


    def __getitem__(self, index):
        img_name = self.ids[index]
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.png"))/255
        img_h, img_w = image.shape[:2]

        #annotation file
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.geojson")

        boxes = []
        labels = []

        with open(annot_file_path, "r") as f:
            annot_json = json.load(f)

        for item in annot_json["features"]:
            points_array = np.array(item['geometry']['coordinates'][0][:-1])
            lon_coords = points_array[:, 0]  # All longitude values
            lat_coords = points_array[:, 1]  # All latitude values

            xmin, ymin, xmax, ymax = self.convert_coords(img_name, lon_coords, lat_coords, img_w, img_h)

            # Ensure the box is within image boundaries
            xmin, xmax = max(0, xmin), min(img_w, xmax)
            ymin, ymax = max(0, ymin), min(img_h, ymax)

            if xmax > xmin and ymax > ymin:
                labels.append(int(1))  # Airplane class
                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

        return self.apply_transformations(image, target)


    def load_dataset(self, data_dir, annotation_dir):
        annotations = []
        ids = list(os.path.splitext(f)[0] for f in os.listdir(annotation_dir) if f.endswith('.geojson'))
        for annot in ids:
            annot_path = os.path.join(annotation_dir, f'{annot}.geojson')
            with open(annot_path, "r") as f:
                js = json.load(f)
            for feat in js["features"]:
                annotations.append({"label": int(1)})

        return annotations, ids
    
    
    def convert_coords(self, img_name, lon_coords, lat_coords, img_w, img_h):
        """
        Convert geographical coordinates to pixel coordinates
        """
        # Parse XML file
        xml_path = os.path.join(self.data_dir, f'{img_name}.png.aux.xml')
        tree = ET.parse(xml_path)

        # Extract geotransform values
        gt_text = tree.find('.//GeoTransform').text
        gt = [float(x) for x in gt_text.replace(',', ' ').split()]

        # Convert coordinates to pixels
        pixel_coords = []
        for lon, lat in zip(lon_coords, lat_coords):
            px = int((lon - gt[0]) / gt[1])
            py = int((lat - gt[3]) / gt[5])
            pixel_coords.append((px, py))

        # Get bounding box
        x_coords = [p[0] for p in pixel_coords]
        y_coords = [p[1] for p in pixel_coords]

        # Clip to image bounds and return
        return [
            max(0, min(x_coords)),
            max(0, min(y_coords)),
            min(img_w-1, max(x_coords)),
            min(img_h-1, max(y_coords))
        ]


    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", "Airplane", True)
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