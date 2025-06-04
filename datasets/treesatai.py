from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader

import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

'''
The dataset was created for forest applications in Europe. It is highly suitable for benchmarking 
as it represents the real-world data situation of many German forest management services.
'''

LABELS = ["Abies_alba","Acer_pseudoplatanus","Alnus_spec.","Betula_spec.","Cleared","Fagus_sylvatica",
          "Fraxinus_excelsior","Larix_decidua","Larix_kaempferi","Picea_abies","Pinus_nigra","Pinus_strobus",
          "Pinus_sylvestris","Populus_spec.","Prunus_spec.","Pseudotsuga_menziesii","Quercus_petraea",
          "Quercus_robur","Quercus_rubra","Tilia_spec."]


class TreeSatAIDataset(MultiClassClassificationDataset):

    url = "https://zenodo.org/records/6780578"
    labels = LABELS
    name = "TreeSatAI dataset"

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
        img = image_loader(self.data[index][0])[:,:,0:3]
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
                raw_data = f.readlines()

                for file_name in raw_data:
                    label = file_name.split('_')
                    if label[0] == 'Cleared':
                        label = label[0]
                    else:
                        label = label[0]+'_'+label[1]
                    item = (
                        os.path.join(self.data_dir, file_name[: -1]),
                        self.labels.index(label),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            label = df["File name"][i].split('_')
            if label[0] == 'Cleared':
                label = label[0]
            else:
                label = label[0]+'_'+label[1]
            df["Label"][i] = label
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