from ..base import BaseDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import fnmatch
from matplotlib.patches import Patch
import math

'''
This dataset was produced as part of the Crop Type Detection competition at the Computer Vision for Agriculture (CV4A) 
Workshop at the ICLR 2020 conference. The objective of the competition was to create a machine learning model to 
classify fields by crop type from images collected during the growing season by the Sentinel-2 satellites. The dataset 
is multitemporal.
'''

LABELS = ["no-data","Maize","Cassava","Common Bean","Maize & Common Bean (intercropping)","Maize & Cassava (intercropping)","Maize & Soybean (intercropping)","Cassava & Common Bean (intercropping)"]


class CV4AKenyaCropTypeDataset(BaseDataset):

    url = "https://source.coop/radiantearth/african-crops-kenya-02"
    labels = LABELS
    color_mapping = [[0,0,0],[255,0,255],[100,100,100],[255,0,0],[255,255,0],[0,0,255],[159,129,183],[0,255,0]] 
    name = "CV4A Kenya Crop Type dataset"
    schema = FAIREOSchema

    def __init__(self, config, date=None):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.images, self.annotations, self.tiles, self.dates = self.load_dataset()


    def __getitem__(self, index):
        tile = self.tiles[index]
        
        # filter the images for the selected tile
        images = [img for img in self.images if f'{self.data_dir}/{tile}/' in img]

        # load Sentinel-2 timeseries
        s2_series = []
        for date in self.dates:
            bands = [img for img in images if date in img]
            bands_image = [image_loader(os.path.join(self.data_dir, band)) for band in bands] 
            s2 = np.stack(bands_image)
            s2_series.append(s2)

        s2 = np.stack(s2_series, axis=0)  # shape: (13, 13, H, W) = (time step, bands, H, W)

        mask = image_loader(self.annotations[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.target_transform:
            mask = self.target_transform(mask)
            
        return s2, mask

    def load_dataset(self):
        images = []
        annotations = []
        tiles = ['0','1','2','3']
        dates = []
        for tile in os.listdir(self.data_dir):
            for date in os.listdir(os.path.join(self.data_dir, tile)):
                if os.path.isdir(os.path.join(self.data_dir, tile, date)):
                    if date not in dates:
                        dates.append(date)
                    for band in os.listdir(os.path.join(self.data_dir, tile, date)):
                        images.append(os.path.join(self.data_dir, tile, date, band))
                elif fnmatch.fnmatch(date,'*label.tif'):
                    annotations.append(os.path.join(self.data_dir,tile,tile+'_label.tif'))

        return images, annotations, tiles, dates
    
    def __len__(self):
        return len(self.images)

    def get_labels(self):
        return self.labels
        
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def show_image(self, index, show_title=False):
        # show the rgb image for the first timestep
        img, mask = self[index]
        img = img[0][1:4,:,:].transpose(1,2,0) 
        #normalize image for better visualization
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, height_factor, 0.8, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 10})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig

    def show_timeseries(self, index, longitude, latitude):
        s2, mask = self[index]
        for i in np.arange(3000): # loop until we find a pixel with a crop type label
            crop = mask[longitude, latitude]
            crop_name = "Unknown"
            for j, k in enumerate(self.labels):
                if crop[j] == 1:
                    crop_name = k
                    break
            # skip no-data pixels
            if crop_name == 'no-data':
                longitude, latitude = longitude+1, latitude+1
            else:
                break

        time_range = np.arange(13)

        # extract Sentinel-2 bands
        s2_pixel = []
        for b in range(12): # ignore the 13th cloud mask band
            s2_band_series = [s2[i][b, longitude, latitude] for i in time_range]
            s2_pixel.append(s2_band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n")

        for i, s2_band_series in enumerate(s2_pixel):
            ax.plot(time_range, s2_band_series, label=f"S2-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.annotations)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        label_count = label_count.drop('no-data')
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig