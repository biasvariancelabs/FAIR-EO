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
import seaborn as sns

'''
M-pv4ger is a modified version of the pv4ger dataset. This original dataset was modified in order 
to be compatible with Geo-Bench. It can be used for image classification tasks to detect rooftop-mounted 
solar photovoltaic systems.
'''

LABELS = ['no solar pv', 'solar pv']


class mPv4gerDataset(BaseDataset):

    url = "https://github.com/allenai/satlas/blob/main/SatlasPretrain.md"
    labels = LABELS
    name = "m-pv4ger"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.data, self.no_pv_ids, self.pv_ids = self.load_dataset(self.data_dir, self.mode, self.csv_file)

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """
        # load image
        img_id = self.data[index]
        with h5py.File(os.path.join(self.data_dir, img_id+'.hdf5'), 'r') as h5_file:
            img = np.dstack((h5_file["Red"][:], h5_file["Green"][:], h5_file["Blue"][:]))

        if img_id in self.no_pv_ids:
            target = self.labels.index("no solar pv")
        elif img_id in self.pv_ids:
            target = self.labels.index("solar pv")

        # apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def load_dataset(self, data_dir, mode, csv_file):
        self.data = []
        for img_id in os.listdir(os.path.join(data_dir)):
            self.data.append(img_id[: -5])

        # label_map.json contains the label of each image
        with open(csv_file, "r") as f:
            data = json.load(f)
            self.no_pv_ids = data["0"]
            self.pv_ids = data["1"]

        train = int(len(self.data)/100*60)
        val = int(len(self.data)/100*20)
        if self.mode == "train":
            self.data = self.data[: train]
        elif self.mode == "val":
            self.data = self.data[train : train+val]
        elif self.mode == "test":
            self.data = self.data[train+val :]

        return self.data, self.no_pv_ids, self.pv_ids
    
    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            image, label = self.__getitem__(i)
            label_dist[self.labels[label]] += 1
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of samples"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of samples', ax=ax)
        if show_title:
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