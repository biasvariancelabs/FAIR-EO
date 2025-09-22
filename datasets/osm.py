import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from ..base import BaseDataset
from .schemas import OSMDatasetSchema
from ..utils import image_loader


"""
OpenSentinelMap dataset consists of more than 1 million multispectral Sentinel-2 images captured over a 4 year 
time period and a set of corresponding per-pixel semantic labels derived from OpenStreetMap data. Each mask 
contains 3 different classification schemas (land use, water and roads, buildings), each in a different band.
"""

class OpenSentinelMapDataset(BaseDataset):

    url = ["https://github.com/VisionSystemsInc/open-sentinel-map?tab=readme-ov-file"]
    labels_land_use = [
        "wooded",
        "agricultural",
        "residential",
        "industrial",
        "commercial",
        "recreation",
        "airport",
        "quarry",
        "military",
        "desert_sand",
        "mountain_rock",
        "other_natural",
        "none",
        "unlabeled"
    ]
    labels_water_and_roads = [
        "water",
        "road",
        "none",
        "unlabeled"
    ]
    labels_buildings = [
        "building",
        "none",
        "unlabeled"
    ]
    color_mapping_land_use = [
        [0, 100, 0],
        [255, 255, 0],
        [152, 223, 138],
        [214, 39, 40],
        [148, 103, 189],
        [153, 26, 77],
        [247, 182, 210],
        [213, 166, 100],
        [134, 115, 47],
        [230, 102, 0],
        [183, 105, 137],
        [85, 184, 144],
        [179, 179, 179],
        [255, 255, 255]
    ]
    color_mapping_water_and_roads = [
        [0, 0, 255],
        [187, 115, 26],
        [179, 179, 179],
        [255, 255, 255]
    ]
    color_mapping_buildings = [
        [255, 0, 0],
        [179, 179, 179],
        [255, 255, 255]
    ]
    name = "Open Sentinel Map"
    schema = OSMDatasetSchema


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.classification = self.config.classification
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.data = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        img_name = self.data[index][3]
        mask_name = self.data[index][2]

        mask_path = os.path.join(self.data_dir, 'labels', f'{mask_name}.png')
        mask = image_loader(mask_path)

        # the mask has 3 bands, one per classification schema

        if self.classification == 'land_use':
            mask = mask[:, :, 0]
            mask = mask.copy()
            mask[mask == 254] = 12 # none class
            mask[mask == 255] = 13 # unlabeled class

        if self.classification == 'water_and_roads':
            mask = mask[:, :, 1]
            mask = mask.copy()
            mask[mask == 12] = 0 # water class
            mask[mask == 13] = 1 # road class
            mask[mask == 254] = 2
            mask[mask == 255] = 3

        if self.classification == 'buildings':
            mask = mask[:, :, 2]
            mask = mask.copy()
            mask[mask == 14] = 0 # building
            mask[mask == 254] = 1
            mask[mask == 255] = 2

        masks = [(mask == v) for v, label in enumerate(getattr(self, f"labels_{self.classification}"))]
        mask = np.stack(masks, axis=-1).astype("float32")

        img_dir = os.path.join(self.data_dir, 'osm_sentinel_imagery', self.data[index][0], self.data[index][1], mask_name, img_name)

        with np.load(img_dir, allow_pickle=True) as image_npz:
            image = image_npz['gsd_10'][:, :, :3]*3

        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        return self.apply_transformations(image, mask)


    def load_dataset(self, data_dir, mode, csv_file=None):
        self.data = []
        img_dir = os.path.join(data_dir, 'osm_sentinel_imagery')

        for year in os.listdir(img_dir):
            for tile in os.listdir(os.path.join(img_dir, year)):
                for scene in os.listdir(os.path.join(img_dir, year, tile)):
                    scene_path = os.path.join(img_dir, year, tile, scene)
                    if not os.listdir(scene_path):
                        continue

                    for npz in os.listdir(os.path.join(img_dir, year, tile, scene)):
                        if os.path.exists(os.path.join(data_dir, 'labels', f'{scene}.png')): # the image has a mask
                            self.data.append([year,tile,scene,npz])

        train = round((len(self.data)/100)*80)
        val = round((len(self.data)/100)*10)
        if mode == 'train':
            self.data = self.data[: train]
        if mode == 'val':
            self.data = self.data[train : train+val]
        if mode == 'test':
            self.data = self.data[train+val :]

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return getattr(self, f"labels_{self.classification}")

    def data_distribution_table(self):
        label_dist = {key: 0 for key in getattr(self, f"labels_{self.classification}")}

        for i in range(len(self.data)):
            _, mask = self.__getitem__(i)
            for index, label in enumerate(getattr(self, f"labels_{self.classification}")):
                label_dist[label] += mask[:, :, index].sum()

        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    def show_image(self, index, show_title=False):
        # plot the 3 classification schemas
        if self.classification == 'all': 
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            classification = ['land_use','water_and_roads','buildings']
            for i, j in enumerate(classification):
                self.classification = j
                
                img, mask = self[index]
                axes[0].imshow(img)
                axes[0].axis("off")
                legend_elements = []
                img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
                labels = getattr(self, f"labels_{self.classification}")
                colors = getattr(self, f"color_mapping_{self.classification}")

                for k, label in enumerate(labels):
                    legend_elements.append(
                        Patch(
                            facecolor=tuple([x / 255 for x in colors[k]]),
                            label=labels[k],
                        )
                    )
                    img_mask[np.where(mask[:, :, k] == 1)] = colors[k]

                axes[i + 1].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.2, 0.95, 0.6, 0.2), fontsize=10, ncol=3)
                axes[i + 1].imshow(img_mask)
                axes[i + 1].axis("off")

            self.classification = 'all' # to plot several examples

            plt.tight_layout()
            plt.show()

        # plot one selected classification schema
        else: 
            img, mask = self[index]
            img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
            legend_elements = []

            labels = getattr(self, f"labels_{self.classification}")
            colors = getattr(self, f"color_mapping_{self.classification}")
            for i, label in enumerate(labels):
                legend_elements.append(
                    Patch(
                        facecolor=tuple([x / 255 for x in colors[i]]),
                        label=labels[i],
                    )
                )
                img_mask[np.where(mask[:, :, i] == 1)] = colors[i]

            fig = plt.figure(figsize=(10, 8))
            height_factor = math.ceil(len(labels)/3)
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
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask