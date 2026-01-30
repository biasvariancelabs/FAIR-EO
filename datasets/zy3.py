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
The Zi-Yuan 3 (ZY3) data set is from the region of Wuhan, China. It was designed for the change detection task.
"""

PATCH_H, PATCH_W = 128, 128

class ZY3Dataset(BaseDataset):

    url = "https://drive.google.com/file/d/1yoVP5xc4dPA2sDwYIgVkMEjwCz7q9AdC/view"
    name = "ZY3"
    schema = FAIREOSchema
    labels = ["no change","change"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.t1_image, self.t2_image, self.mask = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        if self.mode == None: # original image
            pre_image = image_loader(self.t1_image[index])
            post_image = image_loader(self.t2_image[index])
            mask = image_loader(self.mask[index])

        elif self.mode: # patches of the original image
            pre_image = self.t1_image[index]
            post_image = self.t2_image[index]
            mask = self.mask[index]

        #normalize images 
        pre_image = (pre_image - pre_image.min()) / (pre_image.max() - pre_image.min() + 1e-8)
        post_image = (post_image - post_image.min()) / (post_image.max() - post_image.min() + 1e-8)

        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in np.arange(mask.shape[1]):
                if mask[i,j] == 1:
                    single_band_mask[i,j] = 1
                elif mask[i,j] == 255:
                    single_band_mask[i,j] = 0
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        
        self.t1_image = [os.path.join(data_dir, 't1.bmp')]
        self.t2_image = [os.path.join(data_dir, 't2.bmp')]
        self.mask = [os.path.join(data_dir, 'gt.bmp')]

        if self.mode: # create patches of the original image to increase the amount of training data
            self.t1_image = []
            self.t2_image = []
            self.mask = []
            stride = PATCH_H
            im1 = image_loader(os.path.join(data_dir, 't1.bmp'))
            im2 = image_loader(os.path.join(data_dir, 't2.bmp'))
            mask = image_loader(os.path.join(data_dir, 'gt.bmp'))
            # (height, width, bands) format
            img_h, img_w = im1.shape[:2]
            for y in range(0, img_h - PATCH_H + 1, stride):
                for x in range(0, img_w - PATCH_W + 1, stride):
                    if y + PATCH_H <= img_h and x + PATCH_W <= img_w:
                        self.t1_image.append(im1[y:y+PATCH_H, x:x+PATCH_W])
                        self.t2_image.append(im2[y:y+PATCH_H, x:x+PATCH_W])
                        self.mask.append(mask[y:y+PATCH_H, x:x+PATCH_W])

            train = round((len(self.t1_image)/100)*80)
            val = round((len(self.t1_image)/100)*10)

            if self.mode == "train":
                self.t1_image = self.t1_image[: train]
                self.t2_image = self.t2_image[: train]
                self.mask = self.mask[: train]
            elif self.mode == "val":
                self.t1_image = self.t1_image[train : (train+val)]
                self.t2_image = self.t2_image[train : (train+val)]
                self.mask = self.mask[train : (train+val)]
            elif self.mode == "test":
                self.t1_image = self.t1_image[train+val :]
                self.t2_image = self.t2_image[train+val :]
                self.mask = self.mask[train+val :]

        return self.t1_image, self.t2_image, self.mask

    def __len__(self):
        return len(self.t1_image)
    
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
        for i in np.arange(len(self.t1_image)):
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