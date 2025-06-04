import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
This collection contains 206 Pre-Collection Landsat 7 Enhanced Thematic Mapper Plus (ETM+) Level-1G scenes.
Manually generated cloud masks are used to train and validate cloud cover assessment algorithms, which in 
turn are intended to compute the percentage of cloud cover in each scene.
"""


class L7IrishDataset(SemanticSegmentationDataset):
    url = "https://huggingface.co/datasets/torchgeo/l7irish/tree/main"

    labels = ["Fill","Cloud shadow","Clear","Thin cloud","Cloud"]
    color_mapping = [[0,0,0],[125,125,125],[255,255,255],[0,173,196],[0,0,255]] 
    name = "L7Irish"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        R = image[:,:,2]
        G = image[:,:,1]
        B = image[:,:,0]
        image = np.dstack((R,G,B))
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j] == 64:
                    single_band_mask[i,j] = 1
                if mask[i,j] == 128:
                    single_band_mask[i,j] = 2
                if mask[i,j] == 192:
                    single_band_mask[i,j] = 3
                if mask[i,j] == 255:
                    single_band_mask[i,j] = 4

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
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('.TIF')]+'_mask.tif') for image_id in ids]

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