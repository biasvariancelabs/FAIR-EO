import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv 

from ..base import BaseDataset
from .schemas import FAIREOSchema

"""
EuroCropsML is a ready-to-use ML dataset combining EuroCrops reference data with Sentinel-2 reflectance 
data from 2021. It contains data from Latvia, Portugal, and Estonia and is intended for benchmarking 
few-shot crop type classification.
"""


class EuroCropsMLDataset(BaseDataset):
    url = ["https://zenodo.org/records/15095445"]
    name = "EuroCropsML"
    schema = FAIREOSchema
    
    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.data = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        time_series = self.data[index][0] # time series of shape (364, 13) = (time step, band)
        target = self.labels.index(self.data[index][1])

        # Interpolate NaN values
        time_series = self.interpolate_nan_values(time_series)
        time_series = np.asarray(time_series, dtype=np.float32)

        return time_series, target

    def load_dataset(self, data_dir, mode=None):
        self.labels = []
        main_dir = "/".join(data_dir.split('/')[:-2])
        with open(os.path.join(main_dir,'cropclasses.csv'), 'r') as file: # the file can be found in the github repository of the dataset
            csvFile = csv.reader(file)
            next(csvFile) # skip header
            for lines in csvFile:
                self.labels.append(lines[0])

        # Load parquet file into a DataFrame
        # the parquet file contains mean values for each Sentinel-2 band, each of the 13 band contains a single value per observation (364 observation points, one per day of the year)
        df = pd.read_parquet(data_dir)
        # the label values are in the EC_hcat_n column, spectral values (13 for the 13 bands) are in the columns with a date header
        data = []
        dates = df.columns[6:]
        for _, row in df.iterrows():
            time_series = []
            label = row['EC_hcat_n']
            for date in dates:
                bands_values = row[date]
                if bands_values is None: # no observation for that date
                    bands_values = [np.nan] * 13
                time_series.append(bands_values)
            data.append((np.array(time_series), label))
        
        return data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        df = pd.read_parquet(self.data_dir)
        label_count = df['EC_hcat_n'].value_counts()
        distribution_table = pd.DataFrame({'Label': label_count.index, 'Count': label_count.values})

        return distribution_table

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        # sort by count (descending)
        label_count = label_count.sort_values(by="Count", ascending=False)
        # keep the 50 majority classes for display
        label_count = label_count.head(50)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
    
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
            if np.all(nans):
                # If all values are NaN, leave as-is
                continue
            x = np.arange(time_steps)
            interpolated[:, band] = np.interp(x, x[~nans], y[~nans])

        return interpolated
    
    
    def show_timeseries(self, index, bands=None, figsize=(12, 6), show_legend=True):
        """
        Plot Sentinel-2 time series for a given sample index.
        
        Args:
            index (int): Index of the sample in the dataset.
            bands (list or None): List of band indices to plot (0-based). 
                                If None, plots all bands.
            figsize (tuple): Figure size.
            show_legend (bool): Whether to show the legend.
        Returns:
            matplotlib.figure.Figure object.
        """

        # Load data and label for the index
        time_series, label_idx = self[index]

        # Select bands to plot
        if bands is None:
            bands = range(time_series.shape[1])  # All bands by default

        time_steps = time_series.shape[0]
        time_range = np.arange(time_steps)

        label_name = self.labels[label_idx]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Time Series (Index {index}) - Label: {label_name}")

        # Plot each selected band
        for band_idx in bands:
            ax.plot(time_range, time_series[:, band_idx], label=f"Band {band_idx + 1}")

        ax.set_xlabel("Time Step (Day)")
        ax.set_ylabel("Reflectance Value")

        if show_legend:
            ax.legend(loc="upper right", fontsize='small', bbox_to_anchor=(1.10, 1))

        plt.tight_layout()
        plt.show()

        return fig