import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
Field delineation is important in developing and managing agricultural-related policies. 
Given an input satellite image, the goal is to output the delineated boundaries between 
farm parcels, or the segmentation masks of farm parcels. The dataset consists of Sentinel-2 
satellite imagery in France in 2017. Each satellite image comes along with the corresponding 
binary masks of boundaries and areas of farm parcels, which are grey scale images. The dataset 
consists of 1572 training samples, 198 validation samples, and 196 test samples.
"""


class FieldDelineationFranceDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/1GDL1pvlDCcsyEwafe7N41WtS7doFlju3"

    labels = ["no field boundary","field boundary"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "Field_delineation_France"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j] == 255:
                    single_band_mask[i,j] = 1
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "imgs"))

        #split is in a csv file
        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'imgs/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        self.images = [os.path.join(data_dir, "imgs", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "masks", image_id[: image_id.rfind('.jpeg')]+'.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count