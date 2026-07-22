import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import h5py
import math
import os
from matplotlib.patches import Patch


from aitlas.base import BaseDataset
from .schemas import MatDatasetSchema


"""
The multimodal C2Seg dataset contains satellite imagery from Sentinel-1 SAR (VV, VH), 
Sentinel-2 multispectral (RGB, NIR), and EnMAP hyperspectral (330 bands), designed for 
urban semantic segmentation across the cities of Berlin–Augsburg and Beijing–Wuhan.
"""

LABELS = [
    "Background",
    "Surface Waters",
    "Street Network",
    "Urban Fabric",
    "Industrial, Commercial, and Transport",
    "Mine, Dump, and Construction Sites",
    "Artificial Vegetated Areas",
    "Arable Land",
    "Permanent Crops",
    "Pastures",
    "Forests",
    "Shrub",
    "Open Spaces with No Vegetation",
    "Inland Wetlands"
]

color_mapping = [
    [0, 0, 0]  ,       # Background
    [0, 0, 255],       # Surface Waters
    [255, 255, 255],   # Street Network
    [255, 0, 0],       # Urban Fabric
    [255, 192, 203],   # Industrial, Commercial, and Transport
    [128, 0, 128],     # Mine, Dump, and Construction Sites
    [255, 20, 147],    # Artificial Vegetated Areas
    [255, 255, 0],     # Arable Land
    [165, 42, 42],     # Permanent Crops
    [210, 180, 140],   # Pastures
    [124, 252, 0],     # Forests
    [107, 142, 35],    # Shrub
    [139, 69, 19],     # Open Spaces with No Vegetation
    [221, 160, 221]    # Inland Wetlands
]

# Define sliding window size which can be 128 x 128 or 256 x 256
IMG_H, IMG_W = 256, 256


class C2SegDataset(BaseDataset):

    schema = MatDatasetSchema
    url = "https://github.com/danfenghong/RSE_Cross-city?tab=readme-ov-file"
    labels = LABELS
    name = "C2Seg"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)
        self.mat_file = self.config.mat_file
        self.data = self.load_dataset(self.config.mat_file)


    def __getitem__(self, index):
        try:
            mat = scipy.io.loadmat(self.mat_file)
            scene = mat['MSI']
            label_file = self.mat_file.replace('.mat', '_label.mat')
            if label_file != self.mat_file and os.path.exists(label_file):
                label_mat = scipy.io.loadmat(label_file)
                label = label_mat['label']
            else:
                label = mat['label']
        except NotImplementedError:
            with h5py.File(self.mat_file, 'r') as mat:
                scene = np.array(mat['MSI'])
                label_file = self.mat_file.replace('.mat', '_label.mat')
                if label_file != self.mat_file and os.path.exists(label_file):
                    with h5py.File(label_file, 'r') as label_mat:
                        label = np.array(label_mat['label'])
                else:
                    label = np.array(mat['label'])

        x, y = self.data[index]

        if scene.shape[0] == 4:  # Channels first format (Beijing/Wuhan)
            image = scene[:3, y:y+IMG_H, x:x+IMG_W]
            image = np.transpose(image, (1, 2, 0))
        else:  # Standard format (Berlin/Augsburg)
            image = scene[y:y+IMG_H, x:x+IMG_W, :3]

        mask = label[y:y+IMG_H, x:x+IMG_W]
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)


    def load_dataset(self, mat_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        try:
            # Use *_label.mat files for consistent labeling if they exist
            label_file = mat_file.replace('.mat', '_label.mat')
            if label_file != mat_file and os.path.exists(label_file):
                label_mat = scipy.io.loadmat(label_file)
                label = label_mat['label']
            else:
                mat = scipy.io.loadmat(mat_file)
                label = mat['label']
        except NotImplementedError:
            label_file = mat_file.replace('.mat', '_label.mat')
            if label_file != mat_file and os.path.exists(label_file):
                with h5py.File(label_file, 'r') as label_mat:
                    label = np.array(label_mat['label'])
            else:
                with h5py.File(mat_file, 'r') as mat:
                    label = np.array(mat['label'])

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
        try:
            # Use *_label.mat files for consistent labeling if they exist
            label_file = self.config.mat_file.replace('.mat', '_label.mat')
            if label_file != self.config.mat_file and os.path.exists(label_file):
                label_mat = scipy.io.loadmat(label_file)
                label = label_mat['label']
            else:
                mat = scipy.io.loadmat(self.config.mat_file)
                label = mat['label']
        except NotImplementedError:
            label_file = self.config.mat_file.replace('.mat', '_label.mat')
            if label_file != self.config.mat_file and os.path.exists(label_file):
                with h5py.File(label_file, 'r') as label_mat:
                    label = np.array(label_mat['label'])
            else:
                with h5py.File(self.config.mat_file, 'r') as mat:
                    label = np.array(mat['label'])

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