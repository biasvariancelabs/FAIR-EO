import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset contains data from 692 regions of size 10km x 10km around the globe; for each region, 500 latitude/longitude 
coordinates are sampled for their satellite time series and land cover type.
"""


class OODLandCoverDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/138EeHCXxYJZ_OdNqgcn4olY2-CCxEnVa"

    labels = ["deciduous_broadleaf_forests","deciduous_needleleaf_forests","evergreen_broadleaf_forests","evergreen_needleleaf_forests",
              "grasslands","mixed_forests","savannas","urban_and_built_up_lands","water_bodies","woody_savannas",
              "cropland_natural_vegetation_mosaics","croplands","barren","permanent_wetlands","permanent_snow_and_ice","closed_shrublands",
              "open_shrublands"]
    color_mapping = [[139,69,19],[255,0,0],[255,165,0],[255,255,0],[192,192,0],[0,255,0],[0,128,0],[0,255,255],[0,128,128],[0,0,255],
                     [75,0,130],[148,0,211],[255,0,255],[255,192,203],[125,125,125],[173,216,230],[128,0,0]] 
    name = "Out-of-Domain Land Cover"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.img_ids = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        # the image shape is [7, 46], each 7 band contain a single value per observation (46 days of data)
        masked_arr = np.load(os.path.join(self.data_dir, f"MODIS_timeseries_by_box/{img_id}"), allow_pickle=True)
        image = masked_arr.T.astype(np.float32) 
        
        # reflectance values
        image = image/10000 
        image = np.clip(image, 0.0, 1.0)

        # The training requires 10, 12 or 13 bands. Add 3 empty bands. 
        #num_extra_bands = 3
        #extra_bands = np.zeros((image.shape[0], num_extra_bands), dtype=np.float32)
        #image = np.concatenate([image, extra_bands], axis=1)

        target = self.labels.index(img_id.split('/')[1])

        return image, target

    def load_dataset(self, data_dir, mode=None):
        img_ids = []
        split = os.path.join(data_dir, "randomSplit_metadata_1shot_2way.csv")
        with open(split, mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                if mode == 'train':
                    if lines[4] == 'train': # fold value
                        img_ids.append(os.path.join(lines[0],lines[1],lines[3]))
                elif mode == 'test':
                    if lines[4] == 'test': # fold value
                        img_ids.append(os.path.join(lines[0],lines[1],lines[3]))
                else:
                    img_ids.append(os.path.join(lines[0],lines[1],lines[3]))

        if img_ids[0] == 'box/landcover/filename':
            img_ids = img_ids[1 :]

        return img_ids

    def __len__(self):
        return len(self.img_ids)
    
    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        csv_file = os.path.join(self.data_dir, "randomSplit_metadata_1shot_2way.csv")
        df = pd.read_csv(csv_file, sep=",", names=["box","landcover","n_samples","filename","fold","lat","lon","metafold"])
        df = df[1 :]
        df = df.drop(columns=["box","n_samples","fold","lat","lon","metafold"])
        label_count = df.groupby("landcover").count().reset_index()
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
            bands (list or None): List of band indices to plot (0-based). 
                                If None, plots all bands.
            figsize (tuple): Figure size.
            show_legend (bool): Whether to show the legend.
        Returns:
            matplotlib.figure.Figure object.
        """

        # Load data and label for the index
        data, label_idx = self[index]

        # Select bands to plot
        if bands is None:
            bands = range(data.shape[1])  # All bands by default

        time_steps = data.shape[0]
        time_range = np.arange(time_steps)

        label_name = self.labels[label_idx]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Time Series (Index {index}) - Label: {label_name}")

        # Plot each selected band
        for band_idx in bands:
            ax.plot(time_range, data[:, band_idx], label=f"Band {band_idx + 1}")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Reflectance Value")
        
        if show_legend:
            ax.legend(loc="upper right", fontsize='small', bbox_to_anchor=(1.10, 1))

        plt.tight_layout()
        plt.show()

        return fig