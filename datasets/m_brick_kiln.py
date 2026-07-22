from aitlas.base import BaseDataset
from aitlas.utils import image_loader
from .schemas import FAIREOSchema

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import json
import h5py
import re
import seaborn as sns

'''
M-brick-kiln dataset is a modified version of the brick-kiln dataset. It was adapted to Geo-Bench. It is used to 
automatically identifying brick kilns from satellite imagery.
'''

LABELS = ['no kiln', 'kiln']


class mBrickKilnDataset(BaseDataset):

    url = "https://zenodo.org/records/8276815"
    labels = LABELS
    name = "m-brick-kiln"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir, self.mode, self.csv_file)

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """
        img_id = self.data[index]
        with h5py.File(os.path.join(self.data_dir, img_id+'.hdf5'), 'r') as h5_file:    
            # extract image        
            image_b01 = h5_file["01 - Coastal aerosol"][:]
            image_b02 = h5_file["02 - Blue"][:]
            image_b03 = h5_file["03 - Green"][:]
            image_b04 = h5_file["04 - Red"][:]
            image_b05 = h5_file["05 - Vegetation Red Edge"][:]
            image_b06 = h5_file["06 - Vegetation Red Edge"][:]
            image_b07 = h5_file["07 - Vegetation Red Edge"][:]
            image_b08 = h5_file["08 - NIR"][:]
            image_b08A = h5_file["08A - Vegetation Red Edge"][:]
            image_b09 = h5_file["09 - Water vapour"][:]
            image_b10 = h5_file["10 - SWIR - Cirrus"][:]
            image_b11 = h5_file["11 - SWIR"][:]
            image_b12 = h5_file["12 - SWIR"][:]

            # extract label
            raw = h5_file.attrs["pickle"]

        if self.selection == "all":
            img = np.dstack((image_b01, image_b02, image_b03, image_b04, image_b05, image_b06, image_b07, image_b08, image_b08A, image_b09, image_b10, image_b11, image_b12))
        elif self.selection == "rgb":
            img = np.dstack((image_b04, image_b03, image_b02))
        img = (img - img.min())/(img.max() - img.min() + 1e-8)

        # Convert the string representation back to bytes
        raw_bytes = eval(raw)
        # The label is stored as a small integer right after the 'label' key
        match = re.search(rb"label\x94(K\x00|K\x01)", raw_bytes)
        if match:
            # K\x00 = 0, K\x01 = 1 in pickle's encoding
            target = 0 if match.group(1) == b"K\x00" else 1

        # apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def load_dataset(self, data_dir, mode, csv_file):
        with open(csv_file, "r") as f:
            data = json.load(f)
            self.data = data[mode] # mode can be train, valid or test

        return self.data
    
    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            image, label = self.__getitem__(i)
            label_dist[self.labels[label]] += 1
        label_count = pd.DataFrame(label_dist.items(), columns=["Label", "Count"])
        return label_count
    
    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
    
    def show_image(self, index):
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure