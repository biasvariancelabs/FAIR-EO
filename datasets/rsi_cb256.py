from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
RSI-CB stands for Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data. 
RSI-CB256 containss more than 24000 images of size 256 x 256 pixel, and 35 categories.
'''

LABELS = ["airplane","airport_runway","artificial_grassland","avenue","bare_land","bridge","city_building",
          "coastline","container","crossroads","dam","desert","dry_farm","forest","green_farmland","highway",
          "hirst","lakeshore","mangrove","marina","mountain","parkinglot","pipeline","residents","river",
          "river_protection_forest","sandbeach","sapling","sea","shrubwood","snow_mountain","sparse_forest",
          "storage_room","stream","town"]

class RSICB256Dataset(MultiClassClassificationDataset):

    url = "https://figshare.com/articles/dataset/RSI-CB256/22139921?file=39360821"
    labels = LABELS
    name = "RSI-CB256 dataset"

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

                # If not provided initialize the labels from the csv file
                if not self.labels:
                    self.labels = []
                    for index, row in enumerate(raw_data):
                        self.labels.append(row[1])
                    self.labels = list(sorted(set(self.labels)))

                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    item = (
                        os.path.join(self.data_dir, row[1], file_name),
                        self.labels.index(row[0].split('/')[0]),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep="/", names=["Label", "File name"])
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