import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import math
from matplotlib.patches import Patch

from aitlas.base import BaseDataset
from .schemas import MatDatasetSchema


"""
RIT-18 is a high-resolution multispectral dataset for semantic segmentation. It was collected with the 
Tetracam Micro-MCA6 multispectral imaging sensor flown on-board a DJI-1000 octocopter. The main contributions 
of this dataset include 1) very-high resolution multispectral imagery from a drone, 2) six-spectral VNIR bands, 
and 3) 18 object classes (plus background) with a severely unbalanced class distribution.
"""

LABELS = [
    "Other Class/Image Border", "Road Markings", "Tree", "Building", "Vehicle (Car, Truck, or Bus)",
    "Person", "Lifeguard Chair", "Picnic Table", "Black Wood Panel", "White Wood Panel", "Orange Landing Pad",
     "Water Buoy", "Rocks", "Other Vegetation", "Grass", "Sand", "Water (Lake)", "Water (Pond)", 
     "Asphalt (Parking Lot/Walkway)"
]
color_mapping = [
    [0, 0, 0],          # Other Class/Image Border
    [51, 255, 51],      # Road Markings
    [255, 51, 51],      # Tree
    [179, 81, 77],      # Building
    [153, 153, 153],    # Vehicle
    [255, 255, 0],      # Person
    [255, 165, 0],      # Lifeguard Chair
    [0, 128, 255],      # Picnic Table
    [102, 51, 0],       # Black Wood Panel
    [255, 255, 255],    # White Wood Panel
    [255, 128, 0],      # Orange Landing Pad
    [0, 255, 255],      # Water Buoy
    [128, 128, 128],    # Rocks
    [0, 255, 128],      # Other Vegetation
    [0, 200, 0],        # Grass
    [255, 204, 153],    # Sand
    [0, 102, 204],      # Water (Lake)
    [102, 178, 255],    # Water (Pond)
    [128, 0, 128]       # Asphalt (Parking Lot/Walkway)
]

IMG_W, IMG_H = 256, 256

class RIT18Dataset(BaseDataset):
    schema = MatDatasetSchema

    url = "https://github.com/rmkemker/RIT-18"
    labels = LABELS
    name = "RIT-18 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the data
        self.mat_file = self.config.mat_file
        self.mode = self.config.mode
        self.data = self.load_dataset(self.mat_file)

    def __getitem__(self, index):
        mat = scipy.io.loadmat(self.mat_file)
        scene = mat[f'{self.mode}_data']
        label = mat[f'{self.mode}_labels']

        x, y = self.data[index]

        image = scene[:3, y:y+IMG_H, x:x+IMG_W]
        image = np.transpose(image, (1, 2, 0))

        mask = label[y:y+IMG_H, x:x+IMG_W]
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)


    def load_dataset(self, mat_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        mat = scipy.io.loadmat(self.mat_file)
        label = mat[f'{self.mode}_labels']

        height, width = label.shape
        data = []
        stride = IMG_H

        for y in range(0, height - IMG_H + 1, stride):
            for x in range(0, width - IMG_W + 1, stride):
                if y + IMG_H <= height and x + IMG_W <= width:
                    data.append((x,y))

        return data

    def show_image(self, index, show_title=False):
        img, mask = self[index]

        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []

        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = color_mapping[i]

        fig = plt.figure(figsize=(15, 10))
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 5:
            height_factor = 0.68
        elif height_factor == 4:
            height_factor = 0.70
        elif height_factor == 3:
            height_factor = 0.75
        else:
            height_factor = 0.78
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, height_factor, 0.8, 0.25), ncol=3, mode='expand',
                loc='lower left', prop={'size': 10})
        plt.subplot(1, 2, 1)
        img_normalized = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img
        plt.imshow(img_normalized)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):

        mat = scipy.io.loadmat(self.mat_file)
        label = mat[f'{self.mode}_labels']

        label_dist = {key: 0 for key in self.labels}
        for index, label_name in enumerate(self.labels):
            label_dist[label_name] = (label == index).sum()

        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.name), pad=20, fontsize=18
            )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask  