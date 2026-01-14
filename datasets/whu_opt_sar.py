import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from matplotlib.patches import Patch

from .schemas import WHUOPTSARSchema
from ..utils import image_loader
from ..base import BaseDataset

"""
WHU-OPT-SAR is the first and largest land use classification dataset that has fused high resolution optical and SAR 
images with sufficient annotation.
"""

class WHUOPTSARDataset(BaseDataset):
    url = "https://drive.google.com/drive/folders/1W3iMpkehng7ADXmhz9pPvmgFQBayq22t"

    labels = ["background","farmland","city","village","water","forest","road","others"]
    color_mapping = [[0,0,0],[204,102,0],[255,0,0],[255,255,0],[0,0,255],[85,167,0],[0,255,255],[153,102,153]] 
    name = "WHU-OPT-SAR"
    schema = WHUOPTSARSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.csv_file = self.config.csv_file
        self.s1images, self.s2images, self.masks = self.load_dataset(self.data_dir, self.csv_file)


    def __getitem__(self, index):
        # Sentinel-2 image
        if self.imagery == 's2':
            s2image = image_loader(self.s2images[index])

            if self.selection == 'rgb':
                s2image = s2image[:,:,:3]
            elif self.selection == 'all': # 4 bands
                s2image = s2image[:,:,:]
            if self.transform:
                s2image = self.transform(s2image)

            image = s2image
        
        # Sentinel-1 image
        elif self.imagery == 's1':
            s1image = image_loader(self.s1images[index])
            if self.transform:
                s1image = self.transform(s1image)

            image = s1image

        # Sentinel-1 and Sentinel-2 image
        elif self.imagery == 's1 and s2':
            s2image = image_loader(self.s2images[index])
            if self.selection == 'rgb':
                s2image = s2image[:,:,:3]
            elif self.selection == 'all': # 4 bands
                s2image = s2image[:,:,:]
            if self.transform:
                s2image = self.transform(s2image)

            s1image = image_loader(self.s1images[index])
            if self.transform:
                s1image = self.transform(s1image)

            image = np.dstack((s1image, s2image))

        mask = image_loader(self.masks[index])

        color_mapping = np.array(self.color_mapping)
        color_to_label = {tuple(c): i for i, c in enumerate(color_mapping)}

        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for color, label in color_to_label.items():
            single_band_mask[np.all(mask == color, axis=-1)] = label

        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        mask = self.target_transform(mask)

        return image, mask

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        self.images = []
        self.masks = []

        ids = os.listdir(os.path.join(data_dir, "optical"))

        # new 60/20/20 splits are provided by BVLabs
        if csv_file:
            ids = []
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    ids.append(lines[0].split('/')[1])

        self.s2images = [os.path.join(data_dir, "optical", image_id) for image_id in ids]
        self.s1images = [os.path.join(data_dir, "sar", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "predict", image_id) for image_id in ids]

        return self.s1images, self.s2images, self.masks

    def __len__(self):
        return len(self.masks)
    
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

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.masks)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, 0.7, 0.6, 0.5), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})

        if self.imagery == 's1' or self.imagery =='s2':
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")

        elif self.imagery == 's1 and s2':
            # s1 subplot
            plt.subplot(1, 3, 1)
            plt.imshow(img[:,:,0])
            plt.title("S1")
            plt.axis("off")

            # s2 subplot
            plt.subplot(1, 3, 2)
            plt.imshow(img[:,:,1:4])
            plt.title("S2")
            plt.axis("off")

            # mask subplot
            plt.subplot(1, 3, 3)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig