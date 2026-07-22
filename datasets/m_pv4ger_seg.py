import numpy as np
import pandas as pd
import h5py
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.patches import Patch

from aitlas.base import BaseDataset
from .schemas import FAIREOSchema

"""
M-pv4ger-seg dataset is a modified version of pv4ger-seg, adapted to Geo-Bench. It consists of 
the semantic segmentation equivalent of m-pv4seg. It is designed for rooftop-mounted photovoltaic 
systems detection.
"""

class mPv4gerSegDataset(BaseDataset):

    url = "https://zenodo.org/records/8277054"
    labels = ["no solar pv", "solar pv"]
    color_mapping = [[0, 0, 0], [255, 255, 255]]
    name = "m-pv4ger-seg"
    schema = FAIREOSchema


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset(self.data_dir, self.mode, self.csv_file)

    def __getitem__(self, index):
        img_id = self.data[index]
        with h5py.File(os.path.join(self.data_dir, img_id+'.hdf5'), 'r') as h5_file:
            image = np.dstack((h5_file["Red"][:], h5_file["Green"][:], h5_file["Blue"][:]))
            mask = h5_file["label"][:]
            masks = [(mask == v) for v in range(len(self.labels))]
            mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, mode, csv_file):
        with open(csv_file, "r") as f:
            data = json.load(f)
            self.data = data[mode] # mode can be train, valid or test

        return self.data

    def __len__(self):
        return len(self.data)

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