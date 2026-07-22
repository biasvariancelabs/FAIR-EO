import numpy as np
import os
import pandas as pd
import csv

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
S1S2-Water dataset is a global reference dataset for training, validation and testing of convolutional 
neural networks for semantic segmentation of surface water bodies. The dataset consists of 65 triplets 
of Sentinel-1 and Sentinel-2 images with quality checked binary water mask.
"""


class S1S2WaterDataset(SemanticSegmentationDataset):
    url = "https://zenodo.org/records/11278238"

    labels = ["no water", "water"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "S1S2-Water"
    schema = FAIREOSchema


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        if image.shape[2]==2: #Sentinel-1 images
            VV = -image[:,:,0]
            VH = -image[:,:,1]
            VV_VH = VV/VH
            image = np.dstack([VV,VH,VV_VH]) / 10000 #initial values are negative DN

        if image.shape[2]==6: #Sentinel-2 images
            image = image[:,:,0 : 3] /3000
            
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
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
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('_img')]+'_msk.tif') for image_id in ids]

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