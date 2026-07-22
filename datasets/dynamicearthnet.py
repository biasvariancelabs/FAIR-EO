import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from datetime import timedelta
import datetime
import calendar

from aitlas.base import BaseDataset
from .schemas import DynamicEarthNetSchema
from aitlas.utils import image_loader


"""
The DynamicEarthNet dataset that contains daily, cloud-free satellite data acquired from January 2018 to December 2019. 
For each of the 75 regions, the data contains a sequence of images with daily revisits. Pixel-wise semantic labels are 
available for the first day of each month. These serve as ground-truth to define land cover changes over the span of 
two observed years.
"""

class DynamicEarthNetDataset(BaseDataset):

    url = ["https://mediatum.ub.tum.de/1650201"]
    labels = ["impervious surface","agriculture","forest and other vegetation","wetlands","soil","water","snow and ice"]
    color_mapping = [[92,92,92],[0,255,0],[0,255,255],[255,0,255],[255,255,0],[0,0,255],[255,255,255]]
    name = "DynamicEarthNet"
    schema = DynamicEarthNetSchema


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.imagery = self.config.imagery
        self.timeseries = self.config.timeseries
        self.data, self.planetscope_data = self.load_dataset(self.data_dir, self.mode)

    def __getitem__(self, index):
        aoi = self.data[index][0]
        tile = self.data[index][1]
        img_id = self.data[index][2]
        year = img_id[-14:-10]
        month = img_id[-9:-7]

        if self.imagery == "s2":
            image = image_loader(os.path.join(self.data_dir, "sentinel2", aoi[:-4], f'{year}_{month}.tif'))
            if self.selection == "rgb":
                image = image[:,:,[2,1,0]]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif self.imagery == "s1":
            image = image_loader(os.path.join(self.data_dir, "sentinel1", aoi[:-4], f'{aoi[:-4]}_{year}_{month}.tif')) # 8 bands image
            if self.selection == "rgb": # keep a grayscale image
                image = image[:,:,0]
        elif self.imagery == "planet":
            region = self.planetscope_data[index][0]
            aoi_planet = self.planetscope_data[index][1]
            tile_planet = self.planetscope_data[index][2]
            img_id_planet = self.planetscope_data[index][3]
            if self.timeseries == True: # consider the time series
                timeseries = []
                year = int(img_id_planet[:4])
                month = int(img_id_planet[5:7])
                # create a list of dates corresponding to the file names
                last_day = calendar.monthrange(year, month)[1]
                start = datetime.date(year, month, 1)
                end = datetime.date(year, month, last_day)
                dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
                date_strings = [d.strftime("%Y-%m-%d") for d in dates]
                for date in date_strings:
                    image = image_loader(os.path.join(self.data_dir, "planet", region, aoi_planet, tile_planet, "PF-SR", f"{date}.tif"))
                    timeseries.append(image)
                image = np.stack(timeseries, axis=0)

            elif self.timeseries == False: # we keep only the first image of the month since those are associated to a mask
                image = image_loader(os.path.join(self.data_dir, "planet", region, aoi_planet, tile_planet, "PF-SR", f"{img_id_planet}-01.tif"))
                if self.selection == "rgb":
                    image = image[:,:,[2,1,0]]
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        elif self.imagery == "all":
            region = self.planetscope_data[index][0]
            aoi_planet = self.planetscope_data[index][1]
            tile_planet = self.planetscope_data[index][2]
            img_id_planet = self.planetscope_data[index][3]
            # we consider the first time step, S1 and S2 are monthly, PlanetScope is daily
            s2_image = image_loader(os.path.join(self.data_dir, "sentinel2", aoi[:-4], f'{year}_{month}.tif'))
            s1_image = image_loader(os.path.join(self.data_dir, "sentinel1", aoi[:-4], f'{aoi[:-4]}_{year}_{month}.tif'))
            planet_image = image_loader(os.path.join(self.data_dir, "planet", region, aoi_planet, tile_planet, "PF-SR", f"{img_id_planet}-01.tif"))
            if self.selection == "rgb":
                s2_image = s2_image[:,:,[2,1,0]]
                s2_image = (s2_image - s2_image.min()) / (s2_image.max() - s2_image.min() + 1e-8)
                s1_image = s1_image[:,:,0]
                s1_image = (s1_image - s1_image.min()) / (s1_image.max() - s1_image.min() + 1e-8)
                planet_image = planet_image[:,:,[2,1,0]]
                planet_image = (planet_image - planet_image.min()) / (planet_image.max() - planet_image.min() + 1e-8)
            image = np.dstack((s1_image, s2_image, planet_image))
        
        mask = image_loader(os.path.join(self.data_dir, "labels", aoi, "Labels", "Raster", tile, img_id))
        masks = [(mask[:, :, v] == 255) for v in range(len(self.labels))]
        mask = np.stack(masks, axis=-1).astype("float32")

        image, mask = self.apply_transformations(image, mask)

        return image, mask


    def load_dataset(self, data_dir, mode, csv_file=None):
        self.data = []
        label_aois = []
        self.planetscope_data = []

        # labels ids
        for aoi in os.listdir(os.path.join(data_dir, "labels")):
            label_aois.append(aoi)
            tiles = os.listdir(os.path.join(data_dir, "labels", aoi, "Labels", "Raster"))
            for tile in tiles:
                img_ids = os.listdir(os.path.join(data_dir, "labels", aoi, "Labels", "Raster", tile))
                for img_id in img_ids:
                    self.data.append((aoi, tile, img_id))

        # not all labels have a corresponding s1 image
        s1_lookup = {}
        for aoi in os.listdir(os.path.join(data_dir, "sentinel1")):
            for img_id in os.listdir(os.path.join(data_dir, "sentinel1", aoi)):
                s1_lookup[(aoi, img_id)] = (aoi, img_id)

        # not all planet images have a label
        planet_lookup = {}
        planet_lookup_full = {}
        for region in os.listdir(os.path.join(data_dir, "planet")):
            for aoi in os.listdir(os.path.join(data_dir, "planet", region)):
                for tile in os.listdir(os.path.join(data_dir, "planet", region, aoi)):
                    for fname in os.listdir(os.path.join(data_dir, "planet", region, aoi, tile, "PF-SR")):
                        date_prefix = fname[:-7]
                        planet_lookup[(tile, date_prefix)] = (tile, date_prefix)
                        planet_lookup_full[(tile, date_prefix)] = (region, aoi, tile, date_prefix)

        # Align self.data (obtained from the label files) with the available images 
        aligned_data = []
        
        for (aoi, tile, img_id) in self.data:
            year = img_id[-14:-10]
            month = img_id[-9:-7]
            aoi_stripped = aoi[:-4] # match label AOI to image tile
            if self.imagery == "planet":
                key = (aoi_stripped, f'{year}-{month}')  
                if key in planet_lookup:
                    aligned_data.append((aoi, tile, img_id))
                    self.planetscope_data.append(planet_lookup_full[key])
            elif self.imagery == "s1":
                key = (aoi_stripped, f'{aoi_stripped}_{year}_{month}.tif')
                if key in s1_lookup:
                    aligned_data.append((aoi, tile, img_id))
            elif self.imagery == "s2":
                aligned_data = self.data
            elif self.imagery == "all":
                key_planet = (aoi_stripped, f'{year}-{month}') 
                key_s1 = (aoi_stripped, f'{aoi_stripped}_{year}_{month}.tif')
                if key_planet in planet_lookup and key_s1 in s1_lookup:
                    aligned_data.append((aoi, tile, img_id))
                    self.planetscope_data.append(planet_lookup_full[key_planet])

        self.data = aligned_data

        train = int(len(self.data) * 0.60)
        val = int(len(self.data) * 0.20)
        if self.mode == "train":
            self.data = self.data[: train]
        elif self.mode == "val":
            self.data = self.data[train : train+val]
        elif self.mode == "test":
            self.data = self.data[train+val :]
        
        return self.data, self.planetscope_data

    def __len__(self):
        return len(self.data)

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
        if self.selection != "rgb":
            print("The selection parameter must be set to rgb for visualization")

        image, mask = self[index]
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
        
        if self.imagery == "s1" or self.imagery == "s2" or self.imagery == "planet":
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")
        
        elif self.imagery == "all":
            plt.subplot(1, 4, 1)
            plt.imshow(image[:,:,0])
            plt.title("S1")
            plt.axis("off")
            plt.subplot(1, 4, 2)
            plt.imshow(image[:,:,1:4])
            plt.title("S2")
            plt.axis("off")
            plt.subplot(1, 4, 3)
            plt.imshow(image[:,:,4:])
            plt.title("Planet")
            plt.axis("off")
            plt.subplot(1, 4, 4)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")
        
        fig.tight_layout()
        plt.show()
        return fig
    
    def show_timeseries(self, index, longitude, latitude):
        if self.imagery != "planet":
            print("The imagery parameter must be set to planet for time series visualization.")
        
        planet, mask = self[index]

        crop = mask[longitude, latitude]
        crop_name = "Unknown"
        for j, k in enumerate(self.labels):
            if crop[j] == 1:
                crop_name = k
                break

        time_range = np.arange(planet.shape[0])

        # extract bands for a specific pixel
        pixel = []
        for b in range(4):
            band_series = [planet[i][longitude, latitude, b] for i in time_range]
            pixel.append(band_series)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n")

        for i, band_series in enumerate(pixel):
            ax.plot(time_range, band_series, label=f"Planet-B{i+1}", alpha=0.5)

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

    def get_labels(self):
        return self.labels