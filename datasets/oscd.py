import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from .semantic_segmentation import BaseDataset
from .schemas import FAIREOSchema
from ..utils import image_loader


"""
The Onera Satellite Change Detection dataset addresses the issue of detecting changes between 
satellite images from different dates. It comprises 24 pairs of multispectral images taken from 
the Sentinel-2 satellites between 2015 and 2018.
"""

class OSCDDataset(BaseDataset):

    url = "https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection"
    name = "OSCD"
    schema = FAIREOSchema
    labels = ["no change","change"]
    color_mapping = [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        aoi = self.data[index]
        post_images = {}
        pre_images = {}

        if self.selection == 'all':
            bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
        elif self.selection == 'rgb':
            bands = ["B02","B03","B04"]
        for band in bands:
            pre_filepath = os.path.join(self.data_dir, 'Onera Satellite Change Detection dataset - Images', aoi, 'imgs_1_rect', f'{band}.tif')
            pre_images[band] = image_loader(pre_filepath)

            post_filepath = os.path.join(self.data_dir, 'Onera Satellite Change Detection dataset - Images', aoi, 'imgs_2_rect', f'{band}.tif')
            post_images[band] = image_loader(post_filepath)
            
        pre_image = np.stack([pre_images[band] for band in bands], axis=0).transpose(1,2,0)
        #normalize image for better visualization
        img_min, img_max = pre_image.min(), pre_image.max()
        pre_image = (pre_image - img_min) / (img_max - img_min + 1e-8)

        post_image = np.stack([post_images[band] for band in bands], axis=0).transpose(1,2,0)
        #normalize image 
        img_min, img_max = post_image.min(), post_image.max()
        post_image = (post_image - img_min) / (img_max - img_min + 1e-8)

        if os.path.exists(os.path.join(self.data_dir, 'Onera Satellite Change Detection dataset - Train Labels', aoi, 'cm', f'{aoi}-cm.tif')):
            label_path = os.path.join(self.data_dir, 'Onera Satellite Change Detection dataset - Train Labels', aoi, 'cm', f'{aoi}-cm.tif')
        else:
            label_path = os.path.join(self.data_dir, 'Onera Satellite Change Detection dataset - Test Labels', aoi, 'cm', f'{aoi}-cm.tif')
        
        mask = image_loader(label_path)
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
    
        for i in np.arange(mask.shape[0]):
            for j in np.arange(mask.shape[1]):
                if mask[i,j] == 1:
                    single_band_mask[i,j] = 0
                elif mask[i,j] == 2:
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

        # all images
        images_path = os.path.join(data_dir, 'Onera Satellite Change Detection dataset - Images')
        for aoi in os.listdir(images_path):
            if os.path.isdir(os.path.join(images_path, aoi)):
                self.data.append(aoi)

        # training images
        if self.mode == 'train':
            self.data = []
            train_path = os.path.join(data_dir, 'Onera Satellite Change Detection dataset - Train Labels')
            for aoi in os.listdir(train_path):
                if os.path.isdir(os.path.join(train_path, aoi)):
                    self.data.append(aoi)
                
        # testing images
        elif self.mode == 'test':
            test_path = os.path.join(data_dir, 'Onera Satellite Change Detection dataset - Test Labels')
            for aoi in os.listdir(test_path):
                if os.path.isdir(os.path.join(test_path, aoi)):
                    self.data.append(aoi)

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (pre_img, post_img), mask = data

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
        plt.subplots_adjust(top=0.85, bottom=0.1)

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