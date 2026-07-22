import numpy as np
import os
import pandas as pd

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
BH-Pools contains annotated swimming pools over Belo Horizonte in Brazil. The data was acquired 
through the Google Earth Pro tool. The images were exported from an eye altitude of 330 meters. 
The dataset consists of 200 images of 8 different neighbourhoods and contains 3980 annotated pools.
"""


class BHPoolsDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/file/d/1Ph1Ubtk2jAYHhEO6whTTvdBKRkhvH7bZ/edit"

    labels = ["background","pool"]
    color_mapping = [[0,0,0],[0,0,255]] 
    name = "BH-Pools"
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
                if mask[i,j] == 65535:
                    single_band_mask[i,j] = 1
                    
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count