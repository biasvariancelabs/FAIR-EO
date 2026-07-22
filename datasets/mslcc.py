import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import os
from matplotlib.patches import Patch

from aitlas.utils import image_loader
from aitlas.base import BaseDataset
from .schemas import FAIREOSchema


"""
The MSLCC dataset has been created for multi-sensor land-cover classification. It includes images acquired 
by Sentinel-1B and Sentinel-2A for two cities (Munich and Berlin, Germany) and their surrounding areas.
"""

LABELS = ["no-data","Agriculture field","Forest","Built-up","Water body"]

# Define sliding window size
PATCH_H, PATCH_W = 256, 256


class MSLCCDataset(BaseDataset):

    schema = FAIREOSchema
    url = "https://www.dlr.de/de/eoc/ueber-uns/institut-fuer-methodik-der-fernerkundung/photogrammetrie-und-bildanalyse/offentliche-datensatze/mslcc"
    labels = LABELS
    color_mapping = [[0, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 255]]
    name = "MSLCC"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.selection = self.config.selection
        self.mode = self.config.mode
        self.patches, self.masks = self.load_dataset(self.data_dir,self.csv_file)

    def __getitem__(self, index):

        image = self.patches[index]/2550
        mask = self.masks[index]
        masks_stacked = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks_stacked, axis=-1).astype("float32")

        image,mask = self.apply_transformations(image, mask)

        return image,mask
    

    def load_dataset(self, data_dir, csv_file = None):

        self.patches = []
        self.masks = []
        stride = PATCH_H

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        ids = os.listdir(os.path.join(data_dir, "annotations"))
        
        if self.selection == "s2":
            self.images_path = [os.path.join(data_dir, "s2", image_id[: image_id.rfind("anno")]+"s2.tif") for image_id in ids]
            self.masks_path = [os.path.join(data_dir, "annotations", image_id) for image_id in ids]

        if self.selection == "s1":
            self.images_path = [os.path.join(data_dir, "s1", image_id[: image_id.rfind("anno")]+"s1.tif") for image_id in ids]
            self.masks_path = [os.path.join(data_dir, "annotations", image_id) for image_id in ids]

        # create patches of the original 2 images to increase the amount of training data
        for i in np.arange(len(self.images_path)):
            image = image_loader(self.images_path[i]) 
            mask = image_loader(self.masks_path[i])

            if np.shape(image)[2]>4: # (bands, height, width) format
                img_h, img_w = image.shape[1:3]
                for y in range(0, img_h - PATCH_H + 1, stride):
                    for x in range(0, img_w - PATCH_W + 1, stride):
                        if y + PATCH_H <= img_h and x + PATCH_W <= img_w:
                            self.patches.append(image[:3, y:y+PATCH_H, x:x+PATCH_W].transpose(2,1,0))
                            self.masks.append(mask[y:y+PATCH_H, x:x+PATCH_W])
            else: # (height, width, bands) format
                img_h, img_w = image.shape[:2]
                for y in range(0, img_h - PATCH_H + 1, stride):
                    for x in range(0, img_w - PATCH_W + 1, stride):
                        if y + PATCH_H <= img_h and x + PATCH_W <= img_w:
                            self.patches.append(image[y:y+PATCH_H, x:x+PATCH_W, :3])
                            self.masks.append(mask[y:y+PATCH_H, x:x+PATCH_W])

        train = round((len(self.patches)/100)*80)
        val = round((len(self.patches)/100)*10)

        if self.mode == "train":
            self.patches = self.patches[: train]
        if self.mode == "val":
            self.patches = self.patches[train : (train+val)]
        if self.mode == "test":
            self.patches = self.patches[train+val :]

        return self.patches, self.masks
    
    def __len__(self):
        return len(self.patches)
    
    def get_labels(self):
        return self.labels
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def show_image(self, index, show_title=False):
        img, mask = self[index]
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

        fig = plt.figure(figsize=(10, 8))
        height_factor = math.ceil(len(self.labels)/3)
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
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self)):
            img, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:,:,index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
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