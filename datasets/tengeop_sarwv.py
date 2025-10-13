from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader
import os 
import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

'''
The TenGeoP-SARwv dataset contains images of Sentinel-1A wave mode (WV) in VV polarization. 
This dataset consists of more than 37000 SAR vignettes divided into ten defined geophysical 
categories, including both oceanic and meteorologic features. It was created to foster the 
development of approaches for ocean SAR image analysis.
'''

LABELS = ["pure ocean waves", "wind streaks","micro convective cells","rain cells","biological slicks",
          "sea ice","iceberg","low wind area","atmospheric front","oceanic front"]


class TenGeoPSARwvDataset(MultiClassClassificationDataset):

    url = "https://www.seanoe.org/data/00456/56796/"
    labels = LABELS
    name = "TenGeoP-SARwv dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """
        # load image
        img = image_loader(self.data[index][0])/100
        img = np.asarray(Image.fromarray(img).convert('RGB'))

        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target


    def load_dataset(self):
        data = []
        if self.csv_file:
            with open(self.csv_file, "r") as f:
                csv_reader = csv.reader(f)
                raw_data = list(csv_reader)

                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    if row[1] == 'F':
                        label = self.labels[0]
                    if row[1] == 'G':
                        label = self.labels[1]
                    if row[1] == 'H':
                        label = self.labels[2]
                    if row[1] == 'I':
                        label = self.labels[3]
                    if row[1] == 'J':
                        label = self.labels[4]
                    if row[1] == 'K':
                        label = self.labels[5]
                    if row[1] == 'L':
                        label = self.labels[6]
                    if row[1] == 'M':
                        label = self.labels[7]
                    if row[1] == 'N':
                        label = self.labels[8]
                    if row[1] == 'O':
                        label = self.labels[9]
                    item = (
                        os.path.join(self.data_dir, file_name),
                        self.labels.index(label),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data
    
    def show_image(self, index):
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0][:,:,0])
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
            axes.imshow(self[image_index][0][:,:,0])
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure
    
    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            if df.loc[i][1] == 'F':
                df.iloc[i][1] = self.labels[0]
            if df.iloc[i][1] == 'G':
                df.iloc[i][1] = self.labels[1]
            if df.iloc[i][1] == 'H':
                df.iloc[i][1] = self.labels[2]
            if df.iloc[i][1] == 'I':
                df.iloc[i][1] = self.labels[3]
            if df.iloc[i][1] == 'J':
                df.iloc[i][1] = self.labels[4]
            if df.iloc[i][1] == 'K':
                df.iloc[i][1] = self.labels[5]
            if df.iloc[i][1] == 'L':
                df.iloc[i][1] = self.labels[6]
            if df.iloc[i][1] == 'M':
                df.iloc[i][1] = self.labels[7]
            if df.iloc[i][1] == 'N':
                df.iloc[i][1] = self.labels[8]
            if df.iloc[i][1] == 'O':
                df.iloc[i][1] = self.labels[9]
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            if df.loc[i][1] == 'F':
                df.iloc[i][1] = self.labels[0]
            if df.iloc[i][1] == 'G':
                df.iloc[i][1] = self.labels[1]
            if df.iloc[i][1] == 'H':
                df.iloc[i][1] = self.labels[2]
            if df.iloc[i][1] == 'I':
                df.iloc[i][1] = self.labels[3]
            if df.iloc[i][1] == 'J':
                df.iloc[i][1] = self.labels[4]
            if df.iloc[i][1] == 'K':
                df.iloc[i][1] = self.labels[5]
            if df.iloc[i][1] == 'L':
                df.iloc[i][1] = self.labels[6]
            if df.iloc[i][1] == 'M':
                df.iloc[i][1] = self.labels[7]
            if df.iloc[i][1] == 'N':
                df.iloc[i][1] = self.labels[8]
            if df.iloc[i][1] == 'O':
                df.iloc[i][1] = self.labels[9]
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ["Label", "Count"]
        return label_count