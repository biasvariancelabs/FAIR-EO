import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.patches import Patch

from aitlas.base import BaseDataset
from .schemas import FAIREOSchema
from aitlas.utils import image_loader


"""
High Resolution Semantic Change Detection Dataset (HRSCD) contains 291 coregistered image pairs of RGB aerial images from IGS's 
BD ORTHO database. Pixel-level change and land cover annotations are provided.
"""

class HRSCDDataset(BaseDataset):

    url = "https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset"
    name = "HRSCD"
    schema = FAIREOSchema
    labels = ["no change","change"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        aoi, img_id = self.data[index]
        aoi_nb = aoi[1:]

        if aoi == 'D14':
            pre_image = image_loader(os.path.join(self.data_dir, 'images_2006/2006', aoi, f'{aoi_nb}-2005-{img_id}.tif'))
        elif aoi == 'D35':
            pre_image = image_loader(os.path.join(self.data_dir, 'images_2006/2006', aoi, f'{aoi_nb}-2006-{img_id}.tif'))
        #normalize image for better visualization
        img_min, img_max = pre_image.min(), pre_image.max()
        pre_image = (pre_image - img_min) / (img_max - img_min + 1e-8)

        post_image = image_loader(os.path.join(self.data_dir, 'images_2012/2012',  aoi, f'{aoi_nb}-2012-{img_id}-0M50-E080.tif'))
        #normalize image 
        img_min, img_max = post_image.min(), post_image.max()
        post_image = (post_image - img_min) / (img_max - img_min + 1e-8)

        mask = image_loader(os.path.join(self.data_dir, 'labels_change/change', aoi, f'{aoi_nb}-2012-{img_id}-0M50-E080.tif'))
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        self.data = []
        for aoi in os.listdir(os.path.join(data_dir, 'images_2006/2006')):
            for img_id in os.listdir(os.path.join(data_dir, 'images_2006/2006', aoi)):
                self.data.append((aoi, img_id[8 : -4]))

        if self.csv_file: # train/val/test splits created by BVLabs
            self.data = []
            with open(self.csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    aoi = lines[0].split('/')[0]
                    img_id = lines[0].split('/')[1][8 : -4]
                    self.data.append((aoi, img_id))

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
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
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig