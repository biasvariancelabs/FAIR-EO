from json.encoder import INFINITY
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aitlas.utils import image_loader
from .schemas import BioMasstersSchema
from aitlas.base import BaseDataset

"""
BioMassters is a deep learning ready dataset to predict yearly Above Ground Biomass (AGB) for Finnish forests 
using multi-temporal satellite imagery from the European Space Agency and European Commission's joint Sentinel-1 
and Sentinel-2 satellite missions.
"""


class BioMasstersDataset(BaseDataset):

    url = "https://nascetti-a.github.io/BioMasster/"
    name = "BioMassters"
    schema = BioMasstersSchema
    labels = ["low density","medium density","high density"]
    color_mapping = [[0,0,255],[0,255,0],[255,0,0]]

    def __init__(self, config):
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.annotation_dir = self.config.annotation_dir
        self.imagery = self.config.imagery
        self.selection = self.config.selection
        self.timeseries = self.config.timeseries
        self.data = self.load_dataset(self.data_dir, self.annotation_dir)

    def __getitem__(self, index):
        img_id = self.data[index]
        months = ["00","01","02","03","04","05","06","07","08","09","10","11"]
        image_ts = []
        for month in months:
            if self.imagery == "S1":
                try: # some patches do not have an image for each month
                    image = image_loader(os.path.join(self.data_dir, f'{img_id}_S1_{month}.tif')) # 4 bands images
                except:
                    continue
            elif self.imagery == "S2":
                try: # some patches do not have an image for each month
                    image = image_loader(os.path.join(self.data_dir, f'{img_id}_S2_{month}.tif')) # 11 bands images
                except:
                    continue
            elif self.imagery == "all":
                try:
                    image_s1 = image_loader(os.path.join(self.data_dir, f'{img_id}_S1_{month}.tif'))
                    image_s2 = image_loader(os.path.join(self.data_dir, f'{img_id}_S2_{month}.tif'))
                except:
                    continue
                image = np.dstack((image_s1, image_s2))

            image_ts.append(image)

        image = np.stack(image_ts, axis=0) # shape = (time stamp, height, width, bands)

        if self.timeseries == False: # we keep only the first time stamp
            image = image[0,:,:,:]
            if self.imagery == "S1":
                if self.selection == "rgb":
                    image = image[:,:,0:3]
                    # normalize image
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                elif self.selection == "all":
                    image = image[:,:,0:4]    
            elif self.imagery == "S2":
                if self.selection == "rgb":
                    image = image[:,:,[2,1,0]]
                    # normalize image
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                elif self.selection == "all":
                    image = image[:,:,0:11]
            elif self.imagery == "all":
                image = image[:,:,0:15]

        mask = image_loader(os.path.join(self.annotation_dir, f'{img_id}_agbm.tif'))
        # map the values to the categorical labels using the classification from the paper https://nascetti-a.github.io/BioMasster/ 
        mask[(mask >= 0) & (mask <= 68.56)] = 0
        mask[(mask > 68.56) & (mask <= 112.92)] = 1
        mask[(mask > 112.92)] = 2
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, annotation_dir):
        self.data = []
        for img_id in os.listdir(annotation_dir):
            if self.imagery == "S2":
                try: # some labels do not have an S2 image
                    img_id = img_id[: -9]
                    image = image_loader(os.path.join(data_dir, f'{img_id}_S2_00.tif'))
                    self.data.append(img_id)
                except:
                    continue
            elif self.imagery == "S1":
                try: # some images are corrupted
                    img_id = img_id[: -9]
                    image = image_loader(os.path.join(data_dir, f'{img_id}_S1_00.tif'))
                    self.data.append(img_id)
                except:
                    continue
            elif self.imagery == "all": 
                try: # keep only the patches that have both S1 and S2
                    img_id = img_id[: -9]
                    image_s1 = image_loader(os.path.join(data_dir, f'{img_id}_S1_00.tif'))
                    image_s2 = image_loader(os.path.join(data_dir, f'{img_id}_S2_00.tif'))
                    self.data.append(img_id)
                except:
                    continue

        return self.data
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

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
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig

    def show_image(self, index, show_title=False):
        img, mask = self[index]
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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, 0.81, 0.6, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        if self.imagery == "S1" or self.imagery == "S2":
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")
        elif self.imagery == "all":
            plt.subplot(1, 3, 1)
            plt.imshow((img[:,:,0:3] - img[:,:,0:3].min()) / (img[:,:,0:3].max() - img[:,:,0:3].min() + 1e-8))
            plt.title("S1 image")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow((img[:,:,4:7] - img[:,:,4:7].min()) / (img[:,:,4:7].max() - img[:,:,4:7].min() + 1e-8))
            plt.title("S2 image")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig

    def __len__(self):
        return len(self.data)

    def show_timeseries(self, index, latitude, longitude):
        image, mask = self.__getitem__(index)

        label = mask[longitude, latitude]
        label_name = "Unknown"
        for j, k in enumerate(self.labels):
            if label[j] == 1:
                label_name = k
                break

        if image.ndim == 3: 
            raise ValueError("No time dimension available; set 'timeseries=True' to get timeseries data.")
        if self.selection != "all":
            print("Set 'selection=all' to get timeseries data.")

        if self.imagery == "S2":
            bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","CLP"]
        elif self.imagery == "S1":
            bands = ["ASC VV", "ASC VH", "DSC VV", "DSC VH"]

        timeseries = []
        for band_idx in range(image.shape[3]): 
            ts = image[:, latitude, longitude, band_idx] # time series for this pixel and band
            timeseries.append(ts)

        timeseries = np.array(timeseries)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Timeseries with index {index} from the region with pixel coordinates ({latitude}, {longitude}), with label {label_name}")

        for band_idx in range(timeseries.shape[0]):
            ax.plot(timeseries[band_idx], label=bands[band_idx])

        ax.set_ylabel("ρ")
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return fig