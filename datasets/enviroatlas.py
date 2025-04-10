import numpy as np
import os
import pandas as pd

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema


"""
This dataset contains imagery, land cover, and object layers for four of the cities covered by the EPA’s 
EnviroAtlas dataset: Pittsburgh, PA, Phoenix, AZ, Durham, NC, and Austin, TX. The dataset consists of 10 
test tiles and 5 validation tiles in each city, with an additional 10 training tiles and 8 (total) validation 
tiles in Pittsburgh, subsetting the total area that the EnviroAtlas dataset covers in each city.
"""


class EnviroAtlasDataset(SemanticSegmentationDataset):
    url = "https://zenodo.org/records/6268150"

    labels = ["water","impervious surface","soil and barren","trees and forest","grass and herbaceous"]
    color_mapping = [[80,117,169],[248,1,0],[209,207,193],[32,117,0],[133,204,98]] 
    name = "EnviroAtlas"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection


    def __getitem__(self, index):
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]): #initial label values are 10,20,30,40,70
            for j in np.arange(mask.shape[1]):
                if mask[i,j] == 10:
                    single_band_mask[i,j] = 0
                if mask[i,j] == 20:
                    single_band_mask[i,j] = 1
                if mask[i,j] == 30:
                    single_band_mask[i,j] = 2
                if mask[i,j] == 40:
                    single_band_mask[i,j] = 3
                if mask[i,j] == 70:
                    single_band_mask[i,j] = 4
        
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

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id[: image_id.rfind('a_naip')]+'h_highres_labels.tif') for image_id in ids]

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