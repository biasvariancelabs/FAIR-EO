import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv 

from ..base import BaseDataset
from .schemas import FAIREOSchema

"""
The dataset contains Sentinel-2 time series of crop types in Kenya. Mapping crop types in smallholder 
regions faces challenges of small fields, sparse ground truth labels, intercropping, and highly 
heterogeneous landscapes.
"""


class CropTypeKenyaDataset(BaseDataset):
    url = ["https://drive.google.com/drive/folders/1Rq1F-ys-rkjftAUnbdGJ1T6udsZWnEyo"]
    name = "Crop Type Mapping in Kenya"
    schema = FAIREOSchema
    labels = ["Banana","Beans","Cassava","Groundnut","Maize","Non-Crop","Other","Sugar Cane","Sweet Potatoes"] 
    color_mapping = [[243, 94, 90],[198, 128, 6],[131, 157, 6],[24, 176, 45],[24, 183, 142],[21, 171, 220],[80, 134, 254],[207, 83, 248],[253, 66, 183]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.img_ids = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        for i in np.arange(10): # allows to skip images without a label
            img_id = self.img_ids[index] 

            # the image shape is [14, 365], each of the 14 bands contains a single value per observation (365 days of data)
            # the npy file is a masked array (contains data with a mask filtering invalid values)
            masked_arr = np.load(os.path.join(self.data_dir, f"timeseries/{img_id}"), allow_pickle=True)
            # fill the invalid values with nan
            image = masked_arr.filled(np.nan)
            # consider the 13 S2 bands
            image = image[:13,:]
            image = image.T.astype(np.float32) 

            # interpolate the nan values
            image = self.interpolate_nan_values(image)
            
            # reflectance values
            image = image/10000 

            labels = os.path.join(self.data_dir, "crop_type_kenya_2017_metadata.csv")
            with open(labels, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    if lines[0] == img_id:
                        try:
                            target = self.labels.index(lines[1]) # cropType value
                        except: # empty label
                            target = ''

            if self.transform:
                image = self.transform(image)

            if isinstance(target, int):
                break
            else:
                index = index + 1

        return image, target

    def load_dataset(self, data_dir, mode=None):
        img_ids = []
        labels = os.path.join(data_dir, "crop_type_kenya_2017_metadata.csv")
        with open(labels, mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                if mode == 'train':
                    if lines[4] == 'train': # fold_random value
                        img_ids.append(lines[0])
                elif mode == 'test':
                    if lines[4] == 'test':
                        img_ids.append(lines[0])
                else:
                    img_ids.append(lines[0])

        if img_ids[0] == 'fileName':
            img_ids = img_ids[1 :]

        return img_ids
    
    def interpolate_nan_values(self, data: np.ndarray):
        """
        Interpolates NaNs in a 2D array (time series), shape: [time, features].
        NaNs are filled using linear interpolation along time axis.
        """
        time_steps, num_features = data.shape
        interpolated = np.copy(data)

        for band in range(num_features):
            y = interpolated[:, band]
            nans = np.isnan(y)
            x = np.arange(time_steps)
            interpolated[:, band] = np.interp(x, x[~nans], y[~nans])

        return interpolated

    def __len__(self):
        return len(self.img_ids)
    
    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        csv_file = os.path.join(self.data_dir, "crop_type_kenya_2017_metadata.csv")
        df = pd.read_csv(csv_file, sep=",", names=["File name", "Label","id","region","fold_random","fold_busia_test","fold_bungoma_test","fold_siaya_test"])
        df = df[1 :]
        df = df.drop(columns=["id","region","fold_random","fold_busia_test","fold_bungoma_test","fold_siaya_test"])
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
    
    def show_timeseries(self, index, bands=None, figsize=(12, 6), show_legend=True):
        """
        Plot Sentinel-2 time series for a given sample index.
        
        Args:
            index (int): Index of the sample in the dataset.
            bands (list or None): List of band indices to plot. If None, plots all bands.
            figsize (tuple): Figure size.
            show_legend (bool): Whether to show the legend.
        Returns:
            matplotlib.figure.Figure object.
        """

        # Load data and label for the index
        data, label_idx = self[index]

        # Select bands to plot, all bands by default
        if bands is None:
            bands = range(data.shape[1])  

        time_steps = data.shape[0]
        time_range = np.arange(time_steps)

        label_name = self.labels[label_idx]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Time Series (Index {index}) - Label: {label_name}")

        # Plot each selected band
        for band_idx in bands:
            ax.plot(time_range, data[:, band_idx], label=f"Band {band_idx + 1}")

        ax.set_xlabel("Time Step (Day)")
        ax.set_ylabel("Reflectance Value")

        if show_legend:
            ax.legend(loc="upper right", fontsize='small', bbox_to_anchor=(1.10, 1))

        plt.tight_layout()
        plt.show()

        return fig