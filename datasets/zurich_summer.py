import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
The Zurich Summer dataset is a collection of 20 chips (crops), taken from a QuickBird 
acquisition of the city of Zurich (Switzerland) in August 2002. QuickBird images are 
composed by 4 channels (NIR-R-G-B) and were pansharpened to the PAN resolution of about 
0.62 cm GSD. We manually annotated 8 different urban and periurban classes : Roads, Buildings, 
Trees, Grass, Bare Soil, Water, Railways and Swimming pools.
"""


class ZurichSummerDataset(SemanticSegmentationDataset):
    url = "https://zenodo.org/records/5914759"

    labels = ["Roads","Buildings","Trees","Grass","Bare soil","Water","Railways","Swimming pools","Clutter"]
    color_mapping = [[0,0,0],[100,100,100],[0,125,0],[0,255,0],[150,80,0],[0,0,150],[255,255,0],[150,150,255],[255,255,255]] 
    name = "Zurich Summer"
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
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j,0] == 100 and mask[i,j,1] == 100 and mask[i,j,2] == 100:
                    single_band_mask[i,j] = 1
                if mask[i,j,0] == 0 and mask[i,j,1] == 125 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 2
                if mask[i,j,0] == 0 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 3
                if mask[i,j,0] == 150 and mask[i,j,1] == 80 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 4
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 150:
                    single_band_mask[i,j] = 5
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 6
                if mask[i,j,0] == 150 and mask[i,j,1] == 150 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 7
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 8
        
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

        ids = os.listdir(os.path.join(data_dir, "images_tif"))
        
        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'images_tif/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        self.images = [os.path.join(data_dir, "images_tif", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "groundtruth", image_id[: image_id.rfind('.tif')]+'_GT.tif') for image_id in ids]

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