from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader
import pandas as pd
import csv
import os
import numpy as np

'''
BrazilDAM is a multi sensor and multitemporal dataset that consists of multispectral images 
of ore tailings dams throughout Brazil. Images are from Sentinel-2 and Landsat 8 satellites.
'''

LABELS = ["nao_barragem","barragem"]


class BrazilDAMDataset(MultiClassClassificationDataset):

    url = "https://drive.google.com/drive/folders/1v1F4faAD8zCm_vocGxILiIUncaRz1pZB"
    labels = LABELS
    name = "BrazilDAM dataset"

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
        img = img[:,:,1:4]*2
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

                    item = (
                        os.path.join(self.data_dir, row[1], file_name),
                        self.labels.index(row[0].split('/')[0]),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data
    
    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            df['Label'][i]=df['File name'][i].split('/')[0]
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            df['Label'][i]=df['File name'][i].split('/')[0]
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ["Label", "Count"]
        return label_count