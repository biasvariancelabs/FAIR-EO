from ..base import BaseDataset
from ..utils import image_loader
from .schemas import BrazilDAMSchema
import pandas as pd
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

'''
BrazilDAM is a multi sensor and multitemporal dataset that consists of multispectral images 
of ore tailings dams throughout Brazil. Images are from Sentinel-2 and Landsat 8 satellites.
'''

LABELS = ["nao_barragem","barragem"]


class BrazilDAMDataset(BaseDataset):

    url = "https://drive.google.com/drive/folders/1v1F4faAD8zCm_vocGxILiIUncaRz1pZB"
    labels = LABELS
    name = "BrazilDAM dataset"
    schema = BrazilDAMSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.imagery = self.config.imagery
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """
        # load image
        img = image_loader(self.data[index][0])
        img = img[:,:,1:4]
        # normalize the image for better visualization
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)

        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target
    
    def load_dataset(self):
        data = []
        if self.csv_file:
            if self.imagery == 'sentinel':
                with open(self.csv_file, "r") as f:
                    csv_reader = csv.reader(f)
                    raw_data = list(csv_reader)

                    for index, row in enumerate(raw_data):
                        file_name = row[0]

                        item = (
                            os.path.join(self.data_dir, row[1], file_name),
                            self.labels.index(row[0].split('/')[0]),
                        )
                        data.append(item)

            elif self.imagery == 'sentinel and landsat':
                with open(self.csv_file, "r") as f:
                    csv_reader = csv.reader(f)
                    raw_data = list(csv_reader)

                    for row in raw_data[1:]:
                        file_name = os.path.join(row[1], row[3], row[0]) # satellite, year, file name

                        item = (
                            os.path.join(self.data_dir, file_name),
                            self.labels.index(row[2]),
                        )
                        data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels
    
    def show_samples(self):
        file_names = [os.path.basename(item[0]) for item in self.data]
        labels = [item[1] for item in self.data]
        df = pd.DataFrame({'File_name': file_names, 'Label': labels})
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.DataFrame(self.data, columns=['File_name', 'Label'])
        label_count = df.groupby("Label").count().reset_index()
        label_count['Label'] = ["nao_barragem","barragem"]
        label_count.columns = ["Label", "Count"]
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