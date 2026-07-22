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
SEmantic Change detectiON Dataset (SECOND) is a well-annotated semantic change detection dataset. It contains 4662 pairs of aerial 
images from several platforms and sensors. Each image is annotated at the pixel level. The change category in the SECOND dataset 
focuses on 6 main land-cover classes, i.e., non-vegetated ground surface, tree, low vegetation, water, buildings and playgrounds, 
that are frequently involved in natural and man-made geographical changes. These 6 selected land-cover categories result in 30 
common change categories (including non-change ).
"""

class SECONDDataset(BaseDataset):

    url = "https://captain-whu.github.io/SCD/"
    name = "SECOND"
    schema = MultilabelChangeDetectionSchema
    labels = ["no-data","non-change","low vegetation","non-vegetated ground surface","tree","water","building","playground"]
    color_mapping =  [[0,0,0],[255,255,255],[0,128,0],[128,128,128],[0,255,0],[0,0,255],[128,0,0],[255,0,0]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.task_type = self.config.task_type
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        img_id = self.data[index]

        # pre event image (time 1)

        pre_image = image_loader(os.path.join(self.data_dir, 'im1', f'{img_id}.png'))
        # normalize image for better visualization
        img_min, img_max = pre_image.min(), pre_image.max()
        pre_image = (pre_image - img_min) / (img_max - img_min + 1e-8)

        # post event image (time 2)

        post_image = image_loader(os.path.join(self.data_dir, 'im2', f'{img_id}.png'))
        # normalize image
        img_min, img_max = post_image.min(), post_image.max()
        post_image = (post_image - img_min) / (img_max - img_min + 1e-8)

        # pre event mask (time 1) multilabel

        pre_mask = image_loader(os.path.join(self.data_dir, 'label1', f'{img_id}.png'))
        single_band_mask = np.zeros([pre_mask.shape[0], pre_mask.shape[1]], np.uint8)
        # match the RGB code to the corresponding index
        for idx, color in enumerate(self.color_mapping):
            single_band_mask[(pre_mask == color).all(axis=-1)] = idx
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        pre_mask = np.stack(masks, axis=-1).astype("float32")

        # post event mask (time 2) multilabel

        post_mask = image_loader(os.path.join(self.data_dir, 'label2', f'{img_id}.png'))
        single_band_mask = np.zeros([post_mask.shape[0], post_mask.shape[1]], np.uint8)
        # match the RGB code to the corresponding index
        for idx, color in enumerate(self.color_mapping):
            single_band_mask[(post_mask == color).all(axis=-1)] = idx
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        post_mask = np.stack(masks, axis=-1).astype("float32")

        mask = (pre_mask, post_mask)

        # binary mask
        if self.task_type == "binary": 
            self.labels = ["non-change","change"]
            self.color_mapping =  [[0,0,0],[255,255,255]]
            mask = image_loader(os.path.join(self.data_dir, 'label1', f'{img_id}.png'))
            single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        
            for i in np.arange(mask.shape[0]):
                for j in np.arange(mask.shape[1]):
                    if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                        single_band_mask[i,j] = 0
                    else:
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
        for id in os.listdir(os.path.join(data_dir, 'im1')):
            self.data.append(id[: -4])

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (pre_img, post_img), mask = data

        if self.task_type == 'multilabel':
            pre_mask, post_mask = mask
            # Create pre mask visualization
            pre_img_mask = np.zeros([pre_mask.shape[0], pre_mask.shape[1], 3], np.uint8)
            legend_elements = []
            for i, label in enumerate(self.labels):
                legend_elements.append(
                    Patch(
                        facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                        label=self.labels[i],
                    )
                )
                pre_img_mask[np.where(pre_mask[:, :, i] == 1)] = self.color_mapping[i]

            # Create post mask visualization
            post_img_mask = np.zeros([post_mask.shape[0], post_mask.shape[1], 3], np.uint8)
            legend_elements = []
            for i, label in enumerate(self.labels):
                legend_elements.append(
                    Patch(
                        facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                        label=self.labels[i],
                    )
                )
                post_img_mask[np.where(post_mask[:, :, i] == 1)] = self.color_mapping[i]

            fig = plt.figure(figsize=(10, 6))

            # Add legend at the top
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
                        loc='center', prop={'size': 12})

            plt.subplot(2, 2, 1)
            plt.imshow(pre_img)
            plt.title("Pre-event")
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.imshow(pre_img_mask)
            plt.title("Pre-event Mask")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(post_img)
            plt.title("Post-event")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            plt.imshow(post_img_mask)
            plt.title("Post-event Mask")
            plt.axis("off")
        
        elif self.task_type == "binary":
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

            fig = plt.figure(figsize=(10, 6))

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
            image, (pre_mask, post_mask) = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += pre_mask[:, :, index].sum()
                label_dist[self.labels[index]] += post_mask[:, :, index].sum()
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