import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch

from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

"""
This dataset contains Harmonized Landsat and Sentinel-2 imagery of burn scars and the associated masks for the 
years 2018-2021 over the contiguous United States. There are 804 512x512 scenes. Its primary purpose is for 
training geospatial machine learning models.
"""


class HLSBurnScarsDataset(SemanticSegmentationDataset):
    url = "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/tree/main"

    labels = ["not burned","burn scar"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "HLSBurnScars"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        if self.selection == "all": # 6 bands images
            image = image_loader(self.images[index])
        elif self.selection == "rgb":
            image = image_loader(self.images[index])[:,:,:3]
            
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir))

        self.images = [os.path.join(data_dir, image_id) for image_id in ids if image_id.endswith("_merged.tif")]
        self.masks = [os.path.join(data_dir, image_id) for image_id in ids if image_id.endswith("mask.tif")]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)): 
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    
    def show_image(self, index, show_title=False):
        image, mask = self[index]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig