from aitlas.datasets.multiclass_classification import MultiClassClassificationDataset
from aitlas.utils import image_loader
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


'''
The aim of this dataset is to automate the difficult task of detecting the location of large ships in satellite images. 
The dataset consists of images extracted from Planet satellite imagery collected over the San Francisco Bay and San Pedro 
Bay areas of California. It includes 4000 80x80 RGB images labeled with either a 'ship' or 'no-ship' classification. 
Images were derived from PlanetScope full-frame visual scene products, which are orthorectified to a 3-meter pixel size.
'''

LABELS = ["no-ship","ship"]

class ShipsNetDataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery"
    labels = LABELS
    name = "ShipsNet dataset"

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

                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    label = row[0].split('/')[1].split('_')[0]
                    if label == '0':
                        label_name = 'no-ship'
                    else:
                        label_name = 'ship'
                    item = (
                        os.path.join(self.data_dir, file_name),
                        self.labels.index(label_name),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name","Label"])
        for i in np.arange(len(df)):
            if df['File name'][i].split('/')[1].split('_')[0] == '0':
                df['Label'][i] = 'no-ship'
            else:
                df['Label'][i] = 'ship'
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
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Dataset", "Label"])
        for i in np.arange(len(df)):
            if df['File name'][i].split('/')[1].split('_')[0] == '0':
                df['Label'][i] = 'no-ship'
            else:
                df['Label'][i] = 'ship'
        return df.head(20)