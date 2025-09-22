import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import seaborn as sns
import numpy as np
import fnmatch
from PIL import Image

from ..utils import image_loader
from .multiclass_classification import MultiClassClassificationDataset


'''
The dataset was created for Sentinel-1 ship interpretation. The version 1 of the dataset contains 11346 
SAR ship chips for 41 Sentinel-1 SAR images. Version 2 adds 34528 SAR ship chips for 46 Sentinel-1 SAR 
images.
'''

LABELS = ["Cargo","Tanker","Passenger","Law Enforcement","Anti-pollution equipment","Port Tender","Search",
            "Search and Rescue vessel","Pilot Vessel","High speed craft","Diving ops","Dredging",
            "Dredging or underwater ops","Towing","Fishing","Wing in ground","Tug","Other Type","Reserved",
            "Pleasure Craft"]


class OpenSARShipDataset(MultiClassClassificationDataset):

    url = "https://opensar.sjtu.edu.cn/index.html"
    labels = LABELS
    name = "OpenSARShip dataset"

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
        img = image_loader(self.data[index][0])
        img = img.astype(np.uint8)
        img = np.asarray(Image.fromarray(img).convert('RGB'))

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
            with open(self.csv_file, "r") as f:
                csv_reader = csv.reader(f)
                raw_data = list(csv_reader)[1 :]

                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    annot = file_name[75 :]
                    label = annot[: annot.rfind('_x')]
                    if fnmatch.fnmatch(label,"1)_*"):
                        label = label[3 :]

                    item = (
                        os.path.join(self.data_dir, file_name),
                        self.labels.index(label),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name","Label"])
        for i in np.arange(len(df)):
            annot = df['File name'][i][75 :]
            label = annot[: annot.rfind('_x')]
            if fnmatch.fnmatch(label,"1)_*"):
                label = label[3 :]
            df['Label'][i]= label
        label_count = df.groupby("Label").count().reset_index()
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

    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            annot = df['File name'][i][75 :]
            label = annot[: annot.rfind('_x')]
            if fnmatch.fnmatch(label,"1)_*"):
                label = label[3 :]
            df['Label'][i]= label
        return df.head(20)    