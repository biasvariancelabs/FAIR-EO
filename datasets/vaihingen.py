import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset was created to provide a benchmark for urban semantic segmentation. It was used in 
the 2D Semantic Labeling Contest. The data set contains 33 patches (of different sizes) acquired 
over Vaihingen (Germany), each consisting of a true orthophoto (TOP) extracted from a larger TOP 
mosaic and a DSM. The dataset is a subset of the data used for the test of digital aerial cameras 
carried out by the German Association of Photogrammetry and Remote Sensing (DGPF).
"""


class VaihingenDataset(SemanticSegmentationDataset):
    url = "https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx"

    labels = ["Impervious surface","Building","Low vegetation","Tree","Car","Clutter"]
    color_mapping = [[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[0,0,0]] 
    name = "Vaihingen"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 0
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 1
                if mask[i,j,0] == 0 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 2
                if mask[i,j,0] == 0 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 3
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 4
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 5
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

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
        self.masks = [os.path.join(data_dir, "labels", image_id) for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):  #because images have different sizes
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count