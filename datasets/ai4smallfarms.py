import numpy as np
import os
import pandas as pd

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader


"""
AI4SmallFarms is a dataset for crop field delineation in Southeast Asian smallholder farms. This data set allows 
researchers to test and benchmark machine learning methods to delineate agricultural field boundaries in polygon 
format.
"""


class AI4SmallFarmsDataset(SemanticSegmentationDataset):
    url = "https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-xy6-ngg6"

    labels = ["no crop boundary","crop boundary"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "AI4SmallFarms"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)


    def __getitem__(self, index):
        image = image_loader(self.images[index])[:,:,:3] # rgb image
        # normalize image for better visualization
        img_min, img_max = image.min(), image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)

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

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "masks", image_id) for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count