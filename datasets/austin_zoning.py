from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
The dataset was created to replicate the project shown in the City Class project by Roman Kuchokov, 
on a different dataset. The objective is to classify a satellite picture of a small part of a city 
into one of its zones (such as residential, industrial, etc). It contains 3666 small satellite images 
of the whole city of Austin, each being labeled with the corresponding zoning tag.
'''

class AustinZoningDataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/franchenstein/austin-zoning-satellite-images"
    labels = ["AV","CS","CS-CO","CS-CO-NP","CS-MU-CO-NP","CS-NP","DR","ERC","GO","GO-CO","GR",
              "GR-CO","GR-MU-CO","GR-NP","I-RR","I-SF-2","I-SF-4A","LA","LI","LI-CO","LI-CO-NP",
              "LI-NP","LI-PDA","LI-PDA-NP","LO","MF-2","MF-2-CO","MF-3","MF-3-CO","MF-3-NP","NBG-NP",
              "P","P-CO","P-CO-NP","P-NP","PUD","PUD-NP","R&D-PDA","RR","RR-NP","SF-1","SF-2","SF-2-CO",
              "SF-2-NP","SF-3","SF-3-NP","SF-4A","SF-6","SF-6-CO","UNZ","other"] # 50 majority classes
    name = "Austing Zoning dataset"

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
                raw_data = list(csv_reader)

                with open(self.data_dir + 'austin.csv', "r") as g:
                    csv_reader = csv.reader(g)
                    label_list = list(csv_reader)

                    # to consider all the 153 classes
                    if not self.labels:
                        self.labels = []
                        for i in label_list:
                            self.labels.append(i[3])
                        self.labels = list(sorted(set(self.labels)))

                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    for i in label_list:
                        name = i[1]
                        if file_name == name:
                            label = i[3]
                    try:
                        item = (
                            os.path.join(self.data_dir, file_name),
                            self.labels.index(label),
                        )
                    except:
                        item = (
                            os.path.join(self.data_dir, file_name),
                            self.labels.index("other"),
                        )

                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name","Label"])
        with open(self.data_dir + 'austin.csv', "r") as g:
            csv_reader = csv.reader(g)
            label_list = list(csv_reader)
            for i in label_list:
                for j in np.arange(len(df)):
                    if df["File name"][j] == i[1]:
                        df["Label"][j] = i[3]
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ["Label", "Count"]
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        for i in np.arange(len(label_count)):
            if label_count["Count"][i]<10:
                label_count = label_count.drop([i])

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
    
    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        with open(self.data_dir + 'austin.csv', "r") as g:
            csv_reader = csv.reader(g)
            label_list = list(csv_reader)
            for i in label_list:
                for j in np.arange(len(df)):
                    if df["File name"][j] == i[1]:
                        df["Label"][j] = i[3]
        return df.head(20)