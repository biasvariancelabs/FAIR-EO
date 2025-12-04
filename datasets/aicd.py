import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from .semantic_segmentation import BaseDataset
from ..utils import image_loader
from .schemas import FAIREOSchema


"""
The dataset consists of before and after scenes from Virtual Battle Station 2 with one major difference 
(new building, road, bush or car) and lots of smaller minor differences (lighting, weather, …). The goal 
was to automatically detect the major differences among the noisy background of other smaller differences.
"""

class AICDDataset(BaseDataset):

    url = "https://www.kaggle.com/datasets/kmader/aerial-change-detection-in-video-games"
    name = "AICD"
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
        img_id = self.data[index]

        if self.selection == 'shadows':
            pre_image = image_loader(os.path.join(self.data_dir, 'Images_Shadows', f'{img_id}_target.png'))
            post_image = image_loader(os.path.join(self.data_dir, 'Images_Shadows', f'{img_id}_moving.png'))
        elif self.selection == 'no shadows':
            pre_image = image_loader(os.path.join(self.data_dir, 'Images_NoShadows', f'{img_id}_target.png'))
            post_image = image_loader(os.path.join(self.data_dir, 'Images_NoShadows', f'{img_id}_moving.png'))

        #normalize image for better visualization
        img_min, img_max = pre_image.min(), pre_image.max()
        pre_image = (pre_image - img_min) / (img_max - img_min + 1e-8)

        img_min, img_max = post_image.min(), post_image.max()
        post_image = (post_image - img_min) / (img_max - img_min + 1e-8)

        mask = image_loader(os.path.join(self.data_dir, 'GroundTruth', f'{img_id}_gtmask.png'))
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)

        if len(mask.shape) == 2:
            for i in np.arange(mask.shape[0]):
                for j in np.arange(mask.shape[1]):
                    if mask[i,j] == 0:
                        single_band_mask[i,j] = 0
                    elif mask[i,j] == 255:
                        single_band_mask[i,j] = 1
        else:
            for i in np.arange(mask.shape[0]):
                for j in np.arange(mask.shape[1]):
                    if mask[i,j,0] == 0:
                        single_band_mask[i,j] = 0
                    elif mask[i,j,0] == 255:
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
        for id in os.listdir(os.path.join(data_dir, 'GroundTruth')):
            self.data.append(id[: -11])

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
            img_mask[np.where(mask[:, :, i] == i)] = self.color_mapping[i]

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