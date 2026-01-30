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
This Benchmark set contains 13 aerial image pairs and binary change masks. During the generation of 
the change mask, the following differences were considered as relevant changes: new built-up regions, 
building operations, planting of large group of trees, fresh plough-land and groundwork before building 
over.
"""

class SZTAKIAirChangeDataset(BaseDataset):

    url = "http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html"
    name = "SZTAKI AirChange"
    schema = FAIREOSchema
    labels = ["no change","change"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        pre_image = image_loader(self.images_pre[index])
        #normalize image for better visualization
        img_min, img_max = pre_image.min(), pre_image.max()
        pre_image = (pre_image - img_min) / (img_max - img_min + 1e-8)

        post_image = image_loader(self.images_post[index])
        #normalize image 
        img_min, img_max = post_image.min(), post_image.max()
        post_image = (post_image - img_min) / (img_max - img_min + 1e-8)

        mask = image_loader(self.annot[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
    
        for i in np.arange(mask.shape[0]):
            for j in np.arange(mask.shape[1]):
                if mask[i,j] == 0:
                    single_band_mask[i,j] = 0
                elif mask[i,j] == 255:
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
        self.images_pre = []
        self.images_post = []
        self.annot = []

        for aoi in os.listdir(data_dir):
            for tile in os.listdir(os.path.join(data_dir, aoi)):
                if os.path.isdir(os.path.join(data_dir, aoi, tile)):
                    if os.path.exists(os.path.join(data_dir, aoi, tile, 'im1.bmp')):
                        self.images_pre.append(os.path.join(data_dir, aoi, tile, 'im1.bmp'))
                    else:
                        self.images_pre.append(os.path.join(data_dir, aoi, tile, 'im1.BMP'))
                    if os.path.exists(os.path.join(data_dir, aoi, tile, 'im2.bmp')):
                        self.images_post.append(os.path.join(data_dir, aoi, tile, 'im2.bmp'))
                    else:
                        self.images_post.append(os.path.join(data_dir, aoi, tile, 'im2.BMP'))
                    if os.path.exists(os.path.join(data_dir, aoi, tile, 'gt.bmp')):
                        self.annot.append(os.path.join(data_dir, aoi, tile, 'gt.bmp'))
                    else:
                        self.annot.append(os.path.join(data_dir, aoi, tile, 'gt.BMP'))
                else:
                    if tile == 'im1.bmp':
                        self.images_pre.append(os.path.join(data_dir, aoi, tile))
                    elif tile == 'im2.bmp':
                        self.images_post.append(os.path.join(data_dir, aoi, tile))
                    else:
                        self.annot.append(os.path.join(data_dir, aoi, tile))

        if self.mode == 'train':
            self.images_pre = self.images_pre[:8]
            self.images_post = self.images_post[:8]
            self.annot = self.annot[:8]
        elif self.mode == 'val':
            self.images_pre = self.images_pre[8:10]
            self.images_post = self.images_post[8:10]
            self.annot = self.annot[8:10]
        elif self.mode == 'test':
            self.images_pre = self.images_pre[10:12]
            self.images_post = self.images_post[10:12]
            self.annot = self.annot[10:12]

        return self.images_pre, self.images_post, self.annot

    def __len__(self):
        return len(self.annot)
    
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

        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.95, 0.8, 0.05), ncol=3,
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