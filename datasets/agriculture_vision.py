import numpy as np
import os
import pandas as pd

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset was created for the CVPR 2020 challenge. It contains 21061 aerial farmland images. Each image also 
has a boundary map and a mask. The boundary map indicates the region of the farmland, and the mask indicates 
valid pixels in the image.
"""


class AgricultureVisionDataset(SemanticSegmentationDataset):
    url = "https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021"

    labels = ["no-data","double_plant","drydown","endrow","nutrient_deficiency","planter_skip","storm_damage","water","waterway","weed_cluster"]
    color_mapping = [[0,0,0],[255,0,0],[255,255,0],[192,192,0],[0,255,0],[125,125,125],[0,0,255],[192,0,0],[0,0,192],[255,255,255]] 
    name = "Agriculture Vision"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        initial_mask = image_loader(self.masks['double_plant'][index])
        mask = {}
        masks = []
        masks.append(np.zeros([initial_mask.shape[0], initial_mask.shape[1]], np.uint8))
        
        for label in self.labels[1 :]:
            mask[label] = image_loader(self.masks[label][index])
            single_band_mask = np.zeros([mask[label].shape[0], mask[label].shape[1]], np.uint8)
            for i in np.arange(mask[label].shape[0]):
                for j in np.arange(mask[label].shape[1]):
                    if mask[label][i,j] == 255:
                        single_band_mask[i,j] = 1
                    
            masks.append(single_band_mask)
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images/rgb"))

        self.images = [os.path.join(data_dir, "images/rgb", image_id) for image_id in ids]
        self.masks = {}
        for label in self.labels[1 :]:
            self.masks[label] = [os.path.join(data_dir, "labels", label, image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count