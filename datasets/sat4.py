import random
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns

from aitlas.base import BaseDataset
from .schemas import MatDatasetSchema


"""
Images were extracted from the National Agriculture Imagery Program (NAIP) dataset. SAT-4 
consists of a total of 500,000 image patches covering four broad land cover classes.
"""

LABELS = ["barren_land", "trees", "grassland", "none"]


class SAT4Dataset(BaseDataset):
    schema = MatDatasetSchema

    url = "https://www.kaggle.com/datasets/crawford/deepsat-sat4/data"
    labels = LABELS
    name = "SAT-4"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        self.mat_file = self.config.mat_file
        self.mode = self.config.mode
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset(self.config.mat_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = self.data[index][0]
        target = self.data[index][1]

        # apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def load_dataset(self, mat_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        indices = None
        if self.csv_file:
            with open(self.csv_file) as infile:
                data = csv.reader(infile)
                indices = [int(row[0]) for row in data]
                indices = sorted(indices, reverse=True)

        data = []
        mat_data = scipy.io.loadmat(mat_file)

        images = mat_data[f"{self.mode}_x"].transpose(3, 0, 1, 2)
        labels = mat_data[f"{self.mode}_y"].transpose()

        if indices:
            valid_indices = [idx for idx in indices if idx < len(images)]
            images = images[valid_indices]
            labels = labels[valid_indices]

        images = images[:, :, :, 0:3]
        class_indices = np.argmax(labels, axis=1)
        data = list(zip(images, class_indices))

        return data

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        mat_data = scipy.io.loadmat(self.config.mat_file)
        img_labels = mat_data[f"{self.mode}_y"].transpose()

        indices = None
        if self.csv_file:
            with open(self.csv_file) as infile:
                data = csv.reader(infile)
                indices = [int(row[0]) for row in data]
                indices = sorted(indices, reverse=True)
        
            valid_indices = [idx for idx in indices if idx < len(img_labels)]
            img_labels = img_labels[valid_indices]

        data = list(np.where(img_labels == 1)[1])
        res_list = [[i, self.labels[index]] for i, index in enumerate(data)]
        df = pd.DataFrame(res_list, columns=["id", "Label"])
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ["Label", "Count"]
        return label_count


    def show_samples(self):
        num_samples = len(self.data)
        sample_list = []

        for i in range(num_samples):
            _, label_idx = self.data[i]
            label_name = self.labels[label_idx]
            sample_list.append([i, label_name])

        df = pd.DataFrame(sample_list, columns=["Image ID", "Label"])
        return df.head(20)

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
        figure, ax = plt.subplots(int(size / 5), 5, figsize=(13.75, 2.8*int(size/5)))
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