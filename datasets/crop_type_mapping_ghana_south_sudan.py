import numpy as np
import pandas as pd
import os
import csv
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from aitlas.base import BaseDataset
from .schemas import FAIREOSchema
from aitlas.utils import image_loader


"""
The dataset is a re-release of the dataset from Rustowicz et al. in Ghana and South Sudan in a format more familiar 
to the machine learning community. The inputs are growing season time series of imagery from three satellites 
(Sentinel-1, Sentinel-2, and Planet's PlanetScope) in 2016 and 2017, and the outputs are semantic segmentation of 
crop types.
"""

class CropTypeGhanaSoudanDataset(BaseDataset):

    url = ["https://drive.google.com/drive/folders/1WhVObtFOzYFiXBsbbrEGy1DUtv7ov7wF"]
    labels = ["Groundnut","Maize","Rice","Soya bean/Sorghum"]
    color_mapping = [[255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 0, 255]]
    name = "Crop Type Mapping (Ghana and South Sudan)"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.data = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        img_id = self.data[index]
        image = np.load(os.path.join(self.data_dir, img_id))

        # shape: (bands, height, width, time_step)
        planet = image['planet'] # shape: (4, 211, 211, 116)
        planet = planet.transpose(3,0,1,2)
        # normalize image for better visualization
        img_min, img_max = planet.min(), planet.max()
        planet = (planet - img_min) / (img_max - img_min + 1e-8)

        s1 = image['s1'] # shape: (3, 64, 64, 29)
        s1 = s1.transpose(3,0,1,2)
        # normalize image for better visualization
        img_min, img_max = s1.min(), s1.max()
        s1 = (s1 - img_min) / (img_max - img_min + 1e-8)

        s2 = image['s2'] # shape: (10, 64, 64, 91)
        s2 = s2.transpose(3,0,1,2)
        # normalize image for better visualization
        img_min, img_max = s2.min(), s2.max()
        s2 = (s2 - img_min) / (img_max - img_min + 1e-8)

        mask = np.load(os.path.join(self.data_dir, img_id.replace('npy','truth')))
        mask = mask['truth'] # shape: (64, 64)

        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        planet, mask = self.apply_transformations(planet, mask)

        return planet, s1, s2, mask


    def load_dataset(self, data_dir, mode, csv_file=None):
        self.data = []

        for aoi in os.listdir(data_dir):
            img_dir = os.path.join(data_dir, aoi, 'npy')
            img_ids = os.listdir(img_dir)
            csv_file = f'/home/dragik/data/FAIR_EO/crop_type_mapping/{aoi}/list_eval_partition.csv'
            if self.mode:
                split_list = []
                with open(csv_file, mode ='r')as file:
                    csvFile = csv.reader(file)
                    for lines in csvFile:
                        if self.mode == 'train':
                            if lines[2] == '0':
                                split_list.append(lines[1])
                        if self.mode == 'val':
                            if lines[2] == '1':
                                split_list.append(lines[1])
                        if self.mode == 'test':
                            if lines[2] == '2':
                                split_list.append(lines[1])

                    split_list = split_list[1 :]

            for img_id in img_ids:
                if self.mode:
                    if img_id[-10 : -4] in split_list:
                        self.data.append(os.path.join(aoi,'npy', img_id))
                elif self.mode == None:
                    self.data.append(os.path.join(aoi,'npy', img_id))

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    
    def show_image(self, index, show_title=False):
        planet, s1, s2, mask = self[index]
        planet = planet[0,:3,:,:].transpose(1,2,0) # first time step for visualization
        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(planet)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig
    
    def show_timeseries(self, index, longitude, latitude):
        planet, s1, s2, mask = self[index]

        crop = mask[longitude, latitude]
        crop_name = "Unknown"
        for j, k in enumerate(self.labels):
            if crop[j] == 1:
                crop_name = k
                break

        time_range_s1 = np.arange(s1.shape[0])
        time_range_s2 = np.arange(s2.shape[0])

        # extract S1 bands for a specific pixel
        s1_pixel = []
        for b in range(3):
            s1_band_series = [s1[i][b, longitude, latitude] for i in time_range_s1]
            s1_pixel.append(s1_band_series)

        # extract Sentinel-2 bands
        s2_pixel = []
        for b in range(10):
            s2_band_series = [s2[i][b, longitude, latitude] for i in time_range_s2]
            s2_pixel.append(s2_band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n")

        for i, s1_band_series in enumerate(s1_pixel):
            if i == 0:
                band = 'VV'
            elif i == 1:
                band = 'VH'
            else:
                band = 'VV/VH'
            ax.plot(time_range_s1, s1_band_series, label=f"S1 {band}")

        for i, s2_band_series in enumerate(s2_pixel):
            ax.plot(time_range_s2, s2_band_series, label=f"S2-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask