import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
This dataset contains high-resolution aerial imagery from the USDA NAIP program, high-resolution 
land cover labels from the Chesapeake Conservancy, low-resolution land cover labels from the USGS 
NLCD 2011 dataset, low-resolution multi-spectral imagery from Landsat 8, and high-resolution building 
footprint masks from Microsoft Bing, formatted to accelerate machine learning research into land 
cover mapping.
"""

class ChesapeakeDataset(SemanticSegmentationDataset):
    url = "https://lila.science/datasets/chesapeakelandcover"

    labels = ["no data","water","tree canopy","low vegetation","barren land","impervious (other)","impervious (road)"]
    color_mapping = [[0,0,0],[0,0,255],[0,128,1],[128,128,128],[0,64,1],[127,128,0],[0,128,129]] 
    name = "Chesapeake"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j] == 1:
                    single_band_mask[i,j] = 1
                if mask[i,j] == 2:
                    single_band_mask[i,j] = 2
                if mask[i,j] == 3:
                    single_band_mask[i,j] = 3
                if mask[i,j] == 4:
                    single_band_mask[i,j] = 4
                if mask[i,j] == 5:
                    single_band_mask[i,j] = 5
                if mask[i,j] == 6:
                    single_band_mask[i,j] = 6
                if mask[i,j] == 15:
                    single_band_mask[i,j] = 0

        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.selection == "rgb":
            image = image_loader(self.images[index])
            image = image[:,:,0:3]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            return image, mask

        elif self.selection == "all":
            image = image_loader(self.images[index])

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
               mask = self.target_transform(mask)

        return image, mask

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))

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

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('naip-new.tif')]+'lc.tif') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):  # images have different sizes
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count