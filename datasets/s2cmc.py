import numpy as np
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from aitlas.base import BaseDataset
from .schemas import FAIREOSchema

'''
This dataset comprises cloud masks for 513 1022x1022 pixels subscenes at 20m resolution.
The subscenes were sampled randomly from the 2018 Level-1C Sentinel-2 archive.
'''

class S2CloudMaskCatalogueDataset(BaseDataset):

    url = "https://zenodo.org/records/4172871#.YIvbA44zbIV"
    labels = ["CLEAR", "CLOUD", "CLOUD_SHADOW"]
    color_mapping = [[0, 255, 0], [255, 255, 255],  [0, 0, 255]]
    name = "Sentinel-2 Cloud Mask Catalogue"
    schema = FAIREOSchema


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.csv_file =  self.config.csv_file
        self.data = self.load_dataset(
            self.data_dir, self.mode, self.csv_file
        )


    def __getitem__(self, index):

        img_name = self.images[index]
        mask_dir = os.path.join(self.data_dir, "masks", img_name + ".npy")
        img_dir = os.path.join(self.data_dir, "subscenes", img_name + ".npy")

        image = np.load(img_dir)[:, :, :3]
        mask = np.load(mask_dir)

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, mode, csv_file=None):
        self.images = []
        self.masks = []
        subscenes_dir = os.path.join(data_dir, "subscenes")
        masks_dir = os.path.join(data_dir, "masks")

        for img_file in os.listdir(subscenes_dir):
            img_name = os.path.splitext(img_file)[0]

            mask_file = img_name + ".npy"
            mask_path = os.path.join(masks_dir, mask_file)

            if os.path.exists(mask_path):
                self.images.append(img_name)
                self.masks.append(img_name)

        if mode == "train":
            self.images = self.images[: 410]
            self.masks = self.masks[: 410]
        if mode == "val":
            self.images = self.images[410 : 461]
            self.masks = self.masks[410 : 461]
        if mode == "test":
            self.images = self.images[461 :]
            self.masks = self.masks[461 :]

    def __len__(self):
        return len(self.images)
    
    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum().item()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask
    
    def show_image(self, index, show_title=False):
        img, mask = self[index]
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
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig