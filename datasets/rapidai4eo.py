from aitlas.base import BaseDataset
from aitlas.utils import image_loader
from .schemas import RapidAI4EOSchema

import random
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import math
import os
import csv
import json 
import fnmatch
from sklearn.model_selection import train_test_split

'''
The RapidAI4EO corpus is a dataset of dense time series satellite imagery. At each location the corpus contains datacubes 
of two cloud-free, regular-cadence image products and corresponding land cover labels: Planet Fusion five-day cadence, 
Sentinel-2 L2A monthly image mosaics and CORINE Land Cover multiclass labels.
'''

LABEL_MAPPING = [
    "clc_111", "clc_112", "clc_121", "clc_122", "clc_123", "clc_124",
    "clc_131", "clc_132", "clc_133", "clc_141", "clc_142",
    "clc_211", "clc_212", "clc_213",
    "clc_221", "clc_222", "clc_223",
    "clc_231",
    "clc_241", "clc_242", "clc_243", "clc_244",
    "clc_311", "clc_312", "clc_313",
    "clc_321", "clc_322", "clc_323", "clc_324",
    "clc_331", "clc_332", "clc_333", "clc_334", "clc_335",
    "clc_411", "clc_412",
    "clc_421", "clc_422", "clc_423",
    "clc_511", "clc_512",
    "clc_521", "clc_522", "clc_523",
    "clc_999"
]

LABELS = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Non-irrigated arable land",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Olive groves",
    "Pastures",
    "Annual crops associated with permanent crops",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grasslands",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland-shrub",
    "Beaches dunes sands",
    "Bare rocks",
    "Sparsely vegetated areas",
    "Burnt areas",
    "Glaciers and perpetual snow",
    "Inland marshes",
    "Peat bogs",
    "Salt marshes",
    "Salines",
    "Intertidal flats",
    "Water courses",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean",
    "NODATA"
]


class RapidAI4EODataset(BaseDataset):
    url = "https://rapidai4eo.source.coop/"

    labels = LABELS
    name = "RapidAI4EO"
    schema = RapidAI4EOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)

        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.imagery = self.config.imagery
        self.date = self.config.date
        self.selection = self.config.selection
        self.task = self.config.task
        self.data = self.load_dataset(self.data_dir, self.csv_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a multi-hot vector.
        """
        path = self.data[index][2]
        tile_path = self.data[index][1]
        tile = self.data[index][0]
        sat = f"{self.imagery}-SR"
        dates = []

        # time series
        for date in os.listdir(os.path.join(path,sat)):
            dates.append(date)

        if self.date == None: # consider the whole time series
            if self.selection == "rgb":
                img = [image_loader(os.path.join(path,sat,date))[:,:,1:4] for date in dates]
            else:
                img = [image_loader(os.path.join(path,sat,date)) for date in dates]
            
        else: # consider a specific date
            if self.selection == "rgb":
                img = image_loader(os.path.join(path,sat,dates[int(self.date)]))[:,:,1:4]
            else:
                img = image_loader(os.path.join(path,sat,dates[int(self.date)]))

            #normalize image for better visualization
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-8)

        if self.transform:
            img = self.transform(img)

        # Load GeoJSON label file
        label_files = os.path.join(self.data_dir,"labels",tile_path)
        for label_file in os.listdir(label_files):
            if fnmatch.fnmatch(label_file, f"*{tile}*"):
                json_file = os.path.join(self.data_dir,"labels",tile_path,label_file)
        with open(json_file, "r") as f:
            data = json.load(f)

        properties = data["properties"]

        # Build ordered label vector
        label_vector = np.array(
            [1.0 if float(properties.get(label, 0.0)) > 0.0 else 0.0
            for label in LABEL_MAPPING]
        )

        target = label_vector

        return img, target

    def load_dataset(self, data_dir, csv_file):

        data = []
        for zone1 in os.listdir(os.path.join(data_dir,"images")):
            for zone2 in os.listdir(os.path.join(data_dir,"images",zone1)):
                for tile in os.listdir(os.path.join(data_dir,"images",zone1,zone2)):
                    path = os.path.join(data_dir,"images",zone1,zone2,tile)
                    tile_path = os.path.join(zone1,zone2)
                    data.append((tile, tile_path, path))

        # Create 60/20/20 train/validation/test splits
        # First split: 60% train vs 40% (temp)
        train, temp = train_test_split(
            data,
            test_size=0.40,
            random_state=42,
            shuffle=True
        )
        # Second split: 20% val, 20% test 
        val, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=42,
            shuffle=True
        )

        if self.task == "train":
            data = train
        elif self.task == "validation":
            data = val
        elif self.task == "test":
            data = test

        return data

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels


    def show_image(self, index):
        labels_list = list(compress(self.labels, self[index][1]))
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with labels:\n "
            f"{str(labels_list).strip('[]')}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def show_batch(self, size, show_title=True):
        if size % 4:
            raise ValueError("The provided size should be divided by 4!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(int(size / 4), 4, figsize=(13.75, 2.0*int(size/4)))
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            labels_list = list(compress(self.labels, self[image_index][1]))
            height, width, depth = self[image_index][0].shape
            white_image = np.zeros([height, width, 3], dtype=np.uint8)
            white_image.fill(255)
            text = '\n'. join(labels_list)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = width/600 * 2.5
            font_thickness = math.ceil(width/600 * 4)
            x = 30

            for i, line in enumerate(text.split('\n')):
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                gap = textsize[1] + 5
                y = textsize[1] + i * gap
                cv2.putText(white_image, line, (x, y), font,
                            font_size,
                            (0, 0, 0),
                            font_thickness,
                            lineType=cv2.LINE_AA)

            display_image = np.hstack((self[image_index][0], white_image))
            axes.imshow(display_image)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axis('off')
        figure.tight_layout()

        return figure

    def show_timeseries(self, index, longitude, latitude, bands=None, figsize=(12, 6), show_legend=True):
        """
        Plot Sentinel-2 or Planet Fusion time series for a given sample index.
        
        Args:
            index (int): Index of the sample in the dataset.
            bands (list or None): List of band indices to plot (0-based). 
                                If None, plots all bands.
            figsize (tuple): Figure size.
            longitude (int): longitude of the selected pixel.
            latitude (int): latitude of the selected pixel.
            show_legend (bool): Whether to show the legend.
        Returns:
            matplotlib.figure.Figure object.
        """

        # Load data and label for the index
        data, label_idx = self[index]

        # Select bands to plot
        if bands is None:
            bands = range(data[0].shape[2])  # All bands by default

        time_steps = len(data)
        time_range = np.arange(time_steps)

        indices = np.where(label_idx == 1)[0]
        label_name = " & ".join([self.labels[index] for index in indices])

        # Extract bands
        img_pixel = []
        for b in range(len(bands)): 
            img_band_series = [data[i][longitude, latitude, b] for i in time_range]
            img_pixel.append(img_band_series)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {label_name}\n")

        for i, img_band_series in enumerate(img_pixel):
            ax.plot(time_range, img_band_series, label=f"B{i+1}", alpha=0.5)

        ax.set_xlabel("Time Step (Day)")
        ax.set_ylabel("Reflectance Value")

        if show_legend:
            ax.legend(loc="upper right", fontsize='small', bbox_to_anchor=(1.10, 1))

        plt.tight_layout()
        plt.show()

        return fig
    
    
    def data_distribution_table(self):
        # Load GeoJSONs
        # Collect binary label vectors for all tiles
        binary_labels = []
        for zone1 in os.listdir(os.path.join(self.data_dir,"labels")):
            for zone2 in os.listdir(os.path.join(self.data_dir,"labels",zone1)):
                for tile in os.listdir(os.path.join(self.data_dir,"labels",zone1,zone2)):
                    tile_path = os.path.join(self.data_dir,"labels",zone1,zone2,tile)
                    with open(tile_path, "r") as f:
                        data = json.load(f)
                        props = data["properties"]
                        # Convert values > 0.0 to 1, else 0
                        label_vector = [1 if float(props.get(label, 0.0)) > 0.0 else 0 for label in LABEL_MAPPING]
                        binary_labels.append(label_vector)

        # Create DataFrame
        df = pd.DataFrame(binary_labels, columns=LABELS)

        # Sum each column to get number of tiles containing the label
        label_count = pd.DataFrame(df.sum(axis=0)).reset_index()
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