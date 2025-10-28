import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from ..utils import image_loader
from .multiclass_classification import MultiClassClassificationDataset

'''
The Cars Overhead With Context (COWC) dataset is a large set of annotated cars from overhead. 
All cars in the annotated images have a dot placed at their center. Negative samples are also 
included in the dataset. It can be used for image classification or object detection/counting 
tasks. This dataloader is written for the the image classification subset of the dataset.
'''

class COWCImageClassificationDataset(MultiClassClassificationDataset):

    url = "https://gdo152.llnl.gov/cowc/download/"
    labels = ["negative","car"] 
    name = "COWC for image classification dataset"

    def __init__(self, config):
        super().__init__(config)
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
        img = image_loader(self.data[index][0])
        target = self.data[index][1]
        # apply transformations
        if self.transform:
            img = self.transform(img)

        return img, target

    def load_dataset(self):
        data = []
        raw_data = []
        if self.csv_file:
            with open(self.csv_file, "r",encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    raw_data.append(line.split(' '))
                       
                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    label = row[1]
                    
                    item = (
                        os.path.join(self.data_dir, file_name),
                        int(label),
                    )

                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        raw_data = []
        with open(self.csv_file, "r",encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                raw_data.append(line.split(' '))

        df = pd.DataFrame(raw_data, columns=['File_name', 'Label'])
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ["Label", "Count"]
        label_count["Label"] = ['Negative','Car']
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
    
    def show_samples(self):
        raw_data = []
        with open(self.csv_file, "r",encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    raw_data.append(line.split(' '))

        df = pd.DataFrame(raw_data, columns=['File_name', 'Label'])
        return df.head(20)