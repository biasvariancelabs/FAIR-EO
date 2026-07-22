import numpy as np
import os
import pandas as pd

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
HD-Maps is a dataset used for proposing a method which enriches road maps by including 
spatially high-resolution features such as the number and width of lanes, sidewalks, and 
parking lanes. This method considers the already existing maps together with the aerial 
and ground photographs.
"""


class HDMapsDataset(SemanticSegmentationDataset):
    url = "https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/hd-maps"

    labels = ["Building","Road","Sidewalk","Parking"]
    color_mapping = [[255,0,0],[255,0,255],[0,0,255],[255,255,0]] 
    name = "HDMaps"
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
                if mask[i,j,0] == 255 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j,0] == 255 and mask[i,j,1] == 105 and mask[i,j,2] == 180:
                    single_band_mask[i,j] = 1
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 2
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 3
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "masks", image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]

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