import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import os
from matplotlib.patches import Patch
import fnmatch

from aitlas.base import BaseDataset
from .schemas import DFC21DSESchema
from aitlas.utils import image_loader


"""
The DFC21-DSE dataset was created for the detection of settlements without electricity challenge track (Track DSE) of the 2021 IEEE GRSS Data Fusion Contest. 
It contains multitemporal and multimodal data.
"""



class DFC21DSEDataset(BaseDataset):

    schema = DFC21DSESchema
    url = "https://ieee-dataport.org/competitions/2021-ieee-grss-data-fusion-contest-track-dse"
    labels = ["no-data","Human settlements without electricity","No human settlements without electricity","Human settlements with electricity","No human settlements with electricity"]
    color_mapping = [[0,0,0],[255,0,0],[0,0,255],[255,255,0],[190,0,255]]
    name = "DFC21-DSE"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.selection = self.config.selection
        self.time_series = self.config.time_series
        self.imagery = self.config.imagery
        self.data, self.tiles = self.load_dataset(self.data_dir,self.csv_file)

    def __getitem__(self, index):
        tile = self.tiles[index]

        '''Sentinel-2'''

        if self.selection == 'all':
            bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
        elif self.selection == 'rgb':
            bands = ['B04','B03','B02']

        # get a time series of dimension (time, bands, height, width)
        band_images = []

        for band in bands:
            band_files = [img for img in self.data[tile]['s2'] if fnmatch.fnmatch(img, f'*{band}.tif')]
            band_stack = np.stack([image_loader(img) for img in band_files], axis=0) # stack along time dimension
            band_images.append(band_stack)

        s2_images = np.stack(band_images, axis=1)  # (time, bands, H, W)

        # normalize image for better visualization
        p1 = np.percentile(s2_images, 1, axis=(1, 2, 3), keepdims=True)
        p99 = np.percentile(s2_images, 99.5, axis=(1, 2, 3), keepdims=True)

        s2_images = np.clip(s2_images, p1, p99)
        s2_images = (s2_images - p1) / (p99 - p1 + 1e-8)

        if self.time_series == False:
            s2_images = s2_images[0,:,:,:].transpose(1,2,0)


        '''Sentinel-1'''

        bands = ['VV','VH']

        # get a time series of dimension (time, bands, height, width)
        band_images = []

        for band in bands:
            band_files = [img for img in self.data[tile]['s1'] if fnmatch.fnmatch(img, f'*{band}.tif')]
            band_stack = np.stack([image_loader(img) for img in band_files], axis=0)
            band_images.append(band_stack)

        s1_images = np.stack(band_images, axis=1)  # (time, bands, H, W)
        vv = s1_images[:, 0, :, :]
        vh = s1_images[:, 1, :, :]
        vv_minus_vh = vv - vh  # (time, H, W)

        s1_images = np.concatenate([s1_images, vv_minus_vh[:, None, :, :]], axis=1)

        # normalize image for better visualization
        p2 = np.percentile(s1_images, 2, axis=(2, 3), keepdims=True)
        p98 = np.percentile(s1_images, 98, axis=(2, 3), keepdims=True)

        s1_images = np.clip(s1_images, p2, p98)
        s1_images = (s1_images - p2) / (p98 - p2 + 1e-8)

        if self.time_series == False:
            s1_images = s1_images[0,:,:,:].transpose(1,2,0)

        
        '''Landsat 8'''

        if self.selection == 'all':
            bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11']
        elif self.selection == 'rgb':
            bands = ['B4','B3','B2']

        # get a time series of dimension (time, bands, height, width)
        band_images = []

        for band in bands:
            band_files = [img for img in self.data[tile]['l8'] if fnmatch.fnmatch(img, f'*{band}.tif')]
            band_stack = np.stack([image_loader(img) for img in band_files], axis=0)
            band_images.append(band_stack)

        l8_images = np.stack(band_images, axis=1)  # (time, bands, H, W)
        
        # normalize image for better visualization
        p1 = np.percentile(l8_images, 1, axis=(1, 2, 3), keepdims=True)
        p99 = np.percentile(l8_images, 99.5, axis=(1, 2, 3), keepdims=True)

        l8_images = np.clip(l8_images, p1, p99)
        l8_images = (l8_images - p1) / (p99 - p1 + 1e-8)

        if self.time_series == False:
            l8_images = l8_images[0,:,:,:].transpose(1,2,0)


        '''VIIRS'''

        dnb_images = np.stack([image_loader(img) for img in self.data[tile]['dnb']], axis=0)
        
        # normalize image for better visualization
        img_min, img_max = dnb_images.min(), dnb_images.max()
        dnb_images = (dnb_images - img_min) / (img_max - img_min + 1e-8)

        if self.time_series == False:
            dnb_images = dnb_images[0,:,:]


        if self.imagery == 's2':
            images = s2_images
        elif self.imagery == 's1':
            images = s1_images
        elif self.imagery == 'l8':
            images = l8_images
        elif self.imagery == 'viirs':
            images = dnb_images
        elif self.imagery == 'all':
            images = (s2_images, s1_images, l8_images, dnb_images)


        mask = image_loader(self.data[tile]['mask'])
        
        masks_stacked = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks_stacked, axis=-1).astype("float32")

        images, mask = self.apply_transformations(images, mask)

        return images, mask
    

    def load_dataset(self, data_dir, csv_file = None):

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        data = {}
        tiles = []

        # note: image names are identical for each tile

        for tile in os.listdir(data_dir):
            tiles.append(tile)
            tile_dir = os.path.join(data_dir, tile)
            ids = os.listdir(tile_dir)
            data[tile] = {
                "dnb": [os.path.join(tile_dir, i) for i in ids if fnmatch.fnmatch(i, 'DNB*')],
                "s2":  [os.path.join(tile_dir, i) for i in ids if fnmatch.fnmatch(i, 'L2A*')],
                "s1":  [os.path.join(tile_dir, i) for i in ids if fnmatch.fnmatch(i, 'S1*')],
                "l8":  [os.path.join(tile_dir, i) for i in ids if fnmatch.fnmatch(i, 'LC08*')],
                "mask": os.path.join(tile_dir, 'groundTruth.tif')
            }

        return data, tiles
    
    def __len__(self):
        return len(self.data)
    
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
        if self.imagery != 'all':
            print('Imagery must be set to all for visualization')

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
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), ncol=1, mode='expand',
                   loc='lower left', prop={'size': 12})

        plt.subplot(1, 5, 1)
        plt.imshow(img[0])
        plt.title('S2 image')
        plt.axis("off")

        plt.subplot(1, 5, 2)
        plt.imshow(img[1])
        plt.title('S1 image')
        plt.axis("off")

        plt.subplot(1, 5, 3)
        plt.imshow(img[2])
        plt.title('L8 image')
        plt.axis("off")

        plt.subplot(1, 5, 4)
        plt.imshow(img[3])
        plt.title('VIIRS image')
        plt.axis("off")

        plt.subplot(1, 5, 5)
        plt.imshow(img_mask)
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self)):
            img, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:,:,index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig

    def show_timeseries(self, index, longitude, latitude):
        if self.imagery != 'all' or self.time_series == False or self.selection != 'all':
            print('Imagery and selection must be set to all, and time_series to True')
        
        image, mask = self[index]

        label = mask[longitude, latitude]
        label_name = "Unknown"
        for j, k in enumerate(self.labels):
            if label[j] == 1:
                label_name = k
                break

        s2 = image[0]
        s1 = image[1]
        l8 = image[2]
        dnb = image[3]

        # extract S1 bands for a specific pixel
        s1_pixel = []
        for b in range(3):
            s1_band_series = [s1[i][b, longitude, latitude] for i in np.arange(4)] # 4 time steps for s1 and s2
            s1_pixel.append(s1_band_series)

        # extract Sentinel-2 bands for a specific pixel
        s2_pixel = []
        for b in range(12):
            s2_band_series = [s2[i][b, longitude, latitude] for i in np.arange(4)]
            s2_pixel.append(s2_band_series)

        # extract L8 bands for a specific pixel
        l8_pixel = []
        for b in range(11):
            l8_band_series = [l8[i][b, longitude, latitude] for i in np.arange(3)] # 3 time steps for L8 data
            l8_pixel.append(l8_band_series)

        # extract VIIRS band for a specific pixel
        dnb_pixel = []
        dnb_band_series = [dnb[i][longitude, latitude] for i in np.arange(9)] # 9 time steps for VIIRS data
        dnb_pixel.append(dnb_band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {label_name}\n")

        for i, s1_band_series in enumerate(s1_pixel):
            if i == 0:
                band = 'VV'
            elif i == 1:
                band = 'VH'
            else:
                band = 'VV-VH'
            ax.plot(np.arange(4), s1_band_series, label=f"S1 {band}")

        for i, s2_band_series in enumerate(s2_pixel):
            ax.plot(np.arange(4), s2_band_series, label=f"S2-B{i+1}", alpha=0.5)

        for i, l8_band_series in enumerate(l8_pixel):
            ax.plot(np.arange(3), l8_band_series, label=f"L8-B{i+1}", alpha=0.5)
        
        for i, dnb_band_series in enumerate(dnb_pixel):
            ax.plot(np.arange(9), dnb_band_series, label=f"VIIRS-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig