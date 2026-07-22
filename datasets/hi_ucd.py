import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from aitlas.base import BaseDataset
from .schemas import MultilabelChangeDetectionSchema
from aitlas.utils import image_loader


"""
Hi-UCD (ultra-High Urban Change Detection) series datasets are designed for deep learning based 
urban semantic change detection. Both semantic labels and change labels are available.
"""

class HiUCDDataset(BaseDataset):

    url = "https://github.com/Daisy-7/Hi-UCD-S"
    name = "Hi-UCD"
    schema = MultilabelChangeDetectionSchema
    labels = ["unlabeled",
              "unchanged water",
              "unchanged grass",
              "unchanged building", 
              "unchanged green house",
              "unchanged road",
              "unchanged bridge",
              "unchanged other",
              "unchanged bare land",
              "unchanged woodland",
              "changed water",
              "changed grass",
              "changed building", 
              "changed green house",
              "changed road",
              "changed bridge",
              "changed other",
              "changed bare land",
              "changed woodland"]
    color_mapping = [[255, 255, 255],
                    [0, 0, 220],
                    [220, 0, 0],
                    [0, 180, 0],
                    [255, 200, 0],
                    [0, 200, 200],
                    [200, 0, 200],
                    [120, 60, 0],
                    [0, 120, 60],
                    [60, 0, 120],
                    [255, 120, 120],
                    [120, 255, 120],
                    [120, 120, 255],
                    [255, 160, 60],
                    [60, 255, 160],
                    [160, 60, 255],
                    [90, 90, 90],
                    [120, 120, 120],
                    [20, 20, 20]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.task_type = self.config.task_type
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        img_id = self.data[index]

        pre_image = image_loader(os.path.join(self.data_dir, 'image/2018', f'{img_id}.png'))
        #normalize image for better visualization
        img_min, img_max = pre_image.min(), pre_image.max()
        pre_image = (pre_image - img_min) / (img_max - img_min + 1e-8)

        post_image = image_loader(os.path.join(self.data_dir, 'image/2019', f'{img_id}.png'))
        #normalize image 
        img_min, img_max = post_image.min(), post_image.max()
        post_image = (post_image - img_min) / (img_max - img_min + 1e-8)

        # multilabel mask
        mask = image_loader(os.path.join(self.data_dir, 'mask/2018_2019', f'{img_id}.png'))
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in np.arange(mask.shape[1]):
                if mask[i,j,0] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 0
                elif mask[i,j,0] == 1 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 1
                elif mask[i,j,0] == 2 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 2
                elif mask[i,j,0] == 3 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 3
                elif mask[i,j,0] == 4 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 4
                elif mask[i,j,0] == 5 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 5
                elif mask[i,j,0] == 6 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 6
                elif mask[i,j,0] == 7 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 7
                elif mask[i,j,0] == 8 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 8
                elif mask[i,j,0] == 9 and mask[i,j,2] == 1:
                    single_band_mask[i,j] = 9
                elif mask[i,j,0] == 1 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 10
                elif mask[i,j,0] == 2 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 11
                elif mask[i,j,0] == 3 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 12
                elif mask[i,j,0] == 4 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 13
                elif mask[i,j,0] == 5 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 14
                elif mask[i,j,0] == 6 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 15
                elif mask[i,j,0] == 7 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 16
                elif mask[i,j,0] == 8 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 17
                elif mask[i,j,0] == 9 and mask[i,j,2] == 2:
                    single_band_mask[i,j] = 18
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        # binary mask
        if self.task_type == "binary": 
            self.labels = ["non-change","change"]
            self.color_mapping =  [[0,0,0],[255,255,255]]
            mask = image_loader(os.path.join(self.data_dir, 'mask/2018_2019', f'{img_id}.png'))
            single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
            for i in np.arange(mask.shape[0]):
                for j in np.arange(mask.shape[1]):
                    if mask[i,j,0] == 0 and mask[i,j,2] == 0:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 1 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 2 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 3 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 4 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 5 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 6 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 7 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 8 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 9 and mask[i,j,2] == 1:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 1 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 2 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 3 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 4 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 5 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 6 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 7 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 8 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
                    elif mask[i,j,0] == 9 and mask[i,j,2] == 2:
                        single_band_mask[i,j] = 1
            
            masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
            mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        self.data = []
        for id in os.listdir(os.path.join(data_dir, 'image/2018')):
            self.data.append(id[: -4])

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (pre_img, post_img), mask = data

        # Create mask visualization
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

        fig = plt.figure(figsize=(15, 6))

        # Create subplot with space for legend on top
        plt.subplots_adjust(top=0.85, bottom=0.1)

        # Add legend at the top
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.1, 0.8, 0.05), ncol=4,
                    loc='center', prop={'size': 12})

        plt.subplot(1, 3, 1)
        plt.imshow(pre_img)
        plt.title("Pre-event")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(post_img)
        plt.title("Post-event")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img_mask)
        plt.title("Mask")
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig