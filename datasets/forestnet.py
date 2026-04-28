import numpy as np
import os
import pandas as pd
import pickle
from collections import defaultdict
import cv2
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

'''
The dataset was developed to automate forest loss driver classification and aid the development and 
implementation of targeted forest conservation and management policies. The dataset consists of 2,756 
satellite images of forest loss events with driver annotations.
'''

class ForestNetDataset(SemanticSegmentationDataset):

    url = ["https://stanfordmlgroup.github.io/projects/forestnet/"]
    labels = ["No data","Plantation", "Smallholder agriculture", "Grassland shrubland", "Other"]
    color_mapping = [[0,0,0],[51,255,51], [255,51,51], [179,81,77],[153,153,153]]
    name = "ForestNet"
    schema = FAIREOSchema


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):

        df =  pd.read_csv(self.csv_file)
        if index >= len(df):
            index = index % len(df)

        image = image_loader(self.images[index])
        h, w = image.shape[:2]
        mask_path = self.masks[index]

        label_value = df.loc[index, 'merged_label']
        label_index = self.labels.index(label_value)

        with open(mask_path, 'rb') as f:
            poly = pickle.load(f)

        # rasterize using OpenCV
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        if poly.geom_type == 'Polygon':
            # Extract exterior coordinates and round to nearest integer
            coords = np.array(poly.exterior.coords).astype(np.int32)
            cv2.fillPoly(binary_mask, [coords], 1)
        elif poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                coords = np.array(p.exterior.coords).astype(np.int32)
                cv2.fillPoly(binary_mask, [coords], 1)

        single_band_mask = np.zeros((h, w), dtype=np.uint8)
        single_band_mask[binary_mask == 1] = label_index

        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)


    def load_dataset(self, data_dir, csv_file=None):
        data = [d for d in os.listdir(data_dir)]

        self.images = []
        self.masks = []

        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[5])
            for i in data:
                i_csv = 'examples/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            data = ids_final

        for d in data:
            image_path = os.path.join(data_dir, d, "images/visible/composite.png")
            self.images.append(image_path)

            mask_path = os.path.join(data_dir, d, "forest_loss_region.pkl")
            self.masks.append(mask_path)


    def data_distribution_table(self):
        df =  pd.read_csv(self.csv_file)
        label_pixel_counts = defaultdict(float)

        for  index, row in df.iterrows():
            label = row['merged_label']
            path = row['example_path']
            path =  os.path.basename(path)

            mask_path = os.path.join(self.data_dir, path, "forest_loss_region.pkl")

            with open(mask_path, 'rb') as f:
                    poly = pickle.load(f)

            area = poly.area

            label_pixel_counts[label] += area

        result_df = pd.DataFrame([
            {"Label": label, "Number of pixels": int(pixel_count)}
            for label, pixel_count in label_pixel_counts.items()
        ])

        return result_df
