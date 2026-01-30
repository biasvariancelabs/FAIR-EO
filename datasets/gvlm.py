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
The Global Very-High-Resolution Landslide Mapping (GVLM) dataset is the first large-scale and 
open-source VHR landslide mapping dataset. It includes 17 bitemporal very-high-resolution imagery 
pairs. Each sub-dataset contains a pair of bitemporal images and the corresponding ground-truth 
map.
"""

class GVLMDataset(BaseDataset):

    url = "https://github.com/zxk688/GVLM"
    name = "GVLM"
    schema = FAIREOSchema
    labels = ["no change","change"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.im1, self.im2, self.ref = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        pre_image = image_loader(self.im1[index])
        post_image = image_loader(self.im2[index])

        mask = image_loader(self.ref[index])
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
        self.data = []
        im1 = [os.path.join(data_dir, aoi, 'im1.png') for aoi in os.listdir(data_dir)]
        im2 = [os.path.join(data_dir, aoi, 'im2.png') for aoi in os.listdir(data_dir)]
        ref = [os.path.join(data_dir, aoi, 'ref.png') for aoi in os.listdir(data_dir)]

        if self.mode == "train":
            im1 = im1[:10]
            im2 = im2[:10]
            ref = ref[:10]

        elif self.mode == "val":
            im1 = im1[10:14]
            im2 = im2[10:14]
            ref = ref[10:14]

        elif self.mode == "test":
            im1 = im1[14:]
            im2 = im2[14:]
            ref = ref[14:]

        return im1, im2, ref

    def __len__(self):
        return len(self.im1)
    
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
        for i in np.arange(len(self.im1)):
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