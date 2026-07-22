import numpy as np
import os
import pandas as pd
import csv

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
Hephaestus is a large scale multitask dataset created for InSAR (Interferometric SAR) development. 
It revolves around the 44 most active volcanoes globally and is focused around global volcano 
monitoring. This description focuses on the semantic segmentation task.
"""


class HephaestusSemanticSegmentationDataset(SemanticSegmentationDataset):
    url = "https://www.dropbox.com/scl/fi/qx99z7tlx6994r7zn4fgh/Hephaestus_Classification.zip?rlkey=lnnmakeu50au0r50yjbjonivx&e=1&dl=0"

    labels = ["no-data","sill","mogi","dyke","earthquake","spheroid","unidentified"]
    color_mapping = [[0,0,0],[140,86,75],[255,127,14],[214,39,40],[188,189,34],[227,119,194],[23,190,207]] 
    name = "Hephaestus Semantic Segmentation"
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
        self.masks = [os.path.join(data_dir, "masks", image_id) for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count