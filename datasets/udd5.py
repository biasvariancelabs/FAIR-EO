import numpy as np
import os
import pandas as pd

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
This is a collection of drone images collected at Peking University, Huludao city, 
Henan University and Cangzhou city. Urban Drone Dataset encompasses UDD-5 and UDD-6. 
UDD-5 contains 120 images and ground truths.
"""


class UDD5Dataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/file/d/1BNL8HNFRiNjSzdcQJo-uXiejZJ6DgunY/view"

    labels = ["Vegetation","Building","Road","Vehicle","Other"]
    color_mapping = [[100,158,0],[149,125,255],[231,102,185],[0,0,255],[0,0,0]] 
    name = "UDD-5"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
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

        ids = os.listdir(os.path.join(data_dir, "src"))

        self.images = [os.path.join(data_dir, "src", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "gt", image_id[: image_id.rfind('.JPG')]+'.png') for image_id in ids]

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