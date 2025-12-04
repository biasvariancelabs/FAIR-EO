from ..base import BaseDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import fnmatch
import json

'''
SatlasPretrain is a large-scale pre-training dataset for remote sensing image understanding. It consists of 302M labels 
under 137 categories and seven label types: points, polygons, polylines, properties, segmentation labels, regression labels, 
and classification labels. This description considers the image classification task.
'''

LABELS = ['wildfire','flood','snow','coast','vessel']


class SatlasPretrainImageClassificationDataset(BaseDataset):

    url = "https://github.com/allenai/satlas/blob/main/SatlasPretrain.md"
    labels = LABELS
    name = "SatlasPretrain Image Classification dataset"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """
        # load image
        img = image_loader(os.path.join(self.data_dir, self.data[index][0]))
        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def load_dataset(self):
        # the images are naip (high resolution) or s2 (low resolution)
        # json files list the tiles assigned to each split
        data = []
        split_final = []
        if self.mode:
            if self.mode == 'train': 
                csv_file = ["/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_highres_train.json",
                            "/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_lowres_train.json"]
            elif self.mode == 'val':
                csv_file = ["/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_highres_val.json",
                            "/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/train_lowres_val.json"]
            elif self.mode == 'test':
                csv_file = ["/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/test_highres.json",
                            "/home/dragik/data/FAIR_EO/satlaspretrain_image_classification/metadata/test_lowres.json"]

            for csv in csv_file:
                with open(csv, "r") as f:
                    split = json.load(f)
                    split_final.extend(split)

        for tile in os.listdir(os.path.join(self.data_dir, 'dynamic')):
            for label in os.listdir(os.path.join(self.data_dir, 'dynamic', tile)):
                label_name = label[: label.rfind('_')]
                for json_file in os.listdir(os.path.join(self.data_dir, 'dynamic', tile, label)):
                    if fnmatch.fnmatch(json_file,'*.json'):
                        with open(os.path.join(self.data_dir, 'dynamic', tile, label, json_file), 'r') as file:
                            vector_file = json.load(file)
                            try:
                                file_name = vector_file["metadata"]["ImageName"]
                                if fnmatch.fnmatch(file_name, 'S2A*'):
                                    tci = os.path.join('sentinel2a',file_name,'tci',tile+'.png')
                                elif fnmatch.fnmatch(file_name, 'S2B*'):
                                    tci = os.path.join('sentinel2b',file_name,'tci',tile+'.png')
                                elif fnmatch.fnmatch(file_name, 'm*'):
                                    tci = os.path.join('naip',file_name,'tci',tile+'.png')
                                else: # file_name == fake
                                    continue
                            except:
                                continue

                # The dataset being very large, only a subset is considered
                # Check if the image exists in our subset
                if os.path.exists(os.path.join(self.data_dir, tci)):
                    if self.mode:
                        tile_csv = [int(tile[: tile.rfind('_')]), int(tile[tile.rfind('_') :][1 :])]
                        if tile_csv in split_final:
                            item = (
                                os.path.join(tci),
                                int(self.labels.index(label_name)),
                            )
                            data.append(item)
                    else:
                        item = (
                                os.path.join(tci),
                                int(self.labels.index(label_name)),
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
        df = pd.DataFrame(self.data, columns=['File name', 'Label'])
        for i in np.arange(len(df)):
            df["Label"][i] = self.labels[int(df["Label"][i])]
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.DataFrame(self.data, columns=['File name', 'Label'])
        for i in np.arange(len(df)):
            df["Label"][i] = self.labels[int(df["Label"][i])]
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ["Label", "Count"]
        return label_count
    
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