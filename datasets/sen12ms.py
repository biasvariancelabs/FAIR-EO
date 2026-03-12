import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv
import math

from ..utils import image_loader
from .schemas import SEN12MSSchema
from ..base import BaseDataset


"""
The SEN12MS dataset contains 180662 patch triplets of corresponding Sentinel-1 dual-pol SAR data, 
Sentinel-2 multi-spectral images, and MODIS-derived land cover maps.
"""

class SEN12MSDataset(BaseDataset):
    url = "https://dataserv.ub.tum.de/s/m1474000"

    labels = ["no-data","Evergreen needleleaf forest","Evergreen broadleaf forest","Deciduous needleleaf forest",
              "Deciduous broadleaf forest","Mixed forest","Closed shrublands","Open shrublands","Woody savannas",
              "Savannas","Grasslands","Permanent wetlands","Croplands","Urban and built-up","Cropland/Natural vegetation mosaic",
              "Snow and ice","Barren or sparsely vegetated","Water"]
    color_mapping = [[255,255,255],[252,132,255],[253,0,1],[0,255,4],[2,0,255],[253,255,0],[1,255,255],[253,0,254],[123,2,1],
                     [122,255,255],[0,0,128],[255,255,126],[128,0,251],[253,128,0],[1,130,252],[126,255,0],[133,1,128],
                     [128,254,128]] 
    name = "SEN12MS"
    schema = SEN12MSSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.csv_file = self.config.csv_file
        self.bands_s2 = self.config.bands_s2
        self.bands_s1 = self.config.bands_s1
        self.s1images, self.s2images, self.masks = self.load_dataset(self.data_dir, self.csv_file)

    def __getitem__(self, index):
        # Sentinel-2 image
        if self.imagery == 's2':
            s2image = image_loader(self.s2images[index])

            if self.selection == 'rgb':
                s2image = s2image[:,:,1:4]
            elif self.selection == 'all': # 13 bands
                s2image = s2image[:,:,:]
            elif self.selection == "bands": # the config must contain a self.bands field with a list of chosen bands
                idx = []
                s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
                for b in self.bands_s2:
                    if b not in s2_bands:
                        print("The bands must be valid Sentinel-2 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12")
                        break
                    else:
                        idx.append(s2_bands.index(b))
                s2image = s2image[:,:,idx]

            # Normalize image
            img_min, img_max = s2image.min(), s2image.max()
            s2image = (s2image - img_min) / (img_max - img_min + 1e-8)
            if self.transform:
                s2image = self.transform(s2image)

            image = s2image
        
        # Sentinel-1 image
        elif self.imagery == 's1':
            s1image = image_loader(self.s1images[index])
        
            if self.selection == 'rgb': # (VV,VH,VV/VH)
                s1image = np.dstack((s1image[:,:,0], s1image[:,:,1], s1image[:,:,0]/s1image[:,:,1])) 
            elif self.selection == 'all': # original 2-band (VV,VH)
                s1image = s1image[:,:,:]
            elif self.selection == 'bands': # the config must contain a self.bands field with a list of chosen bands
                idx = []
                s1_bands = ["VV","VH"]
                for b in self.bands_s1:
                    if b not in s1_bands:
                        print("The bands must be valid Sentinel-1 bands, i.e. one of VV, VH")
                        break
                    else:
                        idx.append(s1_bands.index(b))
                s1image = s1image[:,:,idx]
                
            # Normalize image
            img_min, img_max = s1image.min(), s1image.max()
            s1image = (s1image - img_min) / (img_max - img_min + 1e-8)
            if self.transform:
                s1image = self.transform(s1image)

            image = s1image

        # Sentinel-1 and Sentinel-2 image
        elif self.imagery == 's1 and s2':
            s2image = image_loader(self.s2images[index])
            if self.selection == 'rgb':
                s2image = s2image[:,:,1:4]
            elif self.selection == 'all': # 13 bands
                s2image = s2image[:,:,:]
            elif self.selection == "bands": # the config must contain a self.bands field with a list of chosen bands
                idx = []
                s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
                for b in self.bands_s2:
                    if b not in s2_bands:
                        print("The bands must be valid Sentinel-2 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12")
                        break
                    else:
                        idx.append(s2_bands.index(b))
                s2image = s2image[:,:,idx]
            # Normalize image
            img_min, img_max = s2image.min(), s2image.max()
            s2image = (s2image - img_min) / (img_max - img_min + 1e-8)
            if self.transform:
                s2image = self.transform(s2image)

            s1image = image_loader(self.s1images[index])
            if self.selection == 'rgb': # (VV,VH,VV/VH)
                s1image = np.dstack((s1image[:,:,0], s1image[:,:,1], s1image[:,:,0]/s1image[:,:,1])) 
            elif self.selection == 'all': # original 2-band (VV,VH)
                s1image = s1image[:,:,:]
            elif self.selection == 'bands': # the config must contain a self.bands field with a list of chosen bands
                idx = []
                s1_bands = ["VV","VH"]
                for b in self.bands_s1:
                    if b not in s1_bands:
                        print("The bands must be valid Sentinel-1 bands, i.e. one of VV, VH")
                        break
                    else:
                        idx.append(s1_bands.index(b))
                s1image = s1image[:,:,idx]
            # Normalize image
            img_min, img_max = s1image.min(), s1image.max()
            s1image = (s1image - img_min) / (img_max - img_min + 1e-8)
            if self.transform:
                s1image = self.transform(s1image)

            image = np.dstack((s1image, s2image))

        # the mask contains 4 possible classification schemas
        # this selects the first schema (IGBP)
        mask = image_loader(self.masks[index])[:,:,0] 
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        ids = []
        # csv_files created by BVLabs provide a 60/20/20 split stratified based on season
        # the csv_files contain the file names of the masks, to which corresponds a Sentinel-1 and a Sentinel-2 image
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                img_path = row[0]
                season = row[1]
                ids.append(img_path)

        s1images = [os.path.join(data_dir, "images/s1", data[: data.rfind('lc')]+'s1'+data[data.rfind('lc') :][2 :]) for data in ids]
        s2images = [os.path.join(data_dir, "images/s2", data[: data.rfind('lc')]+'s2'+data[data.rfind('lc') :][2 :]) for data in ids]
        masks = [os.path.join(data_dir, "labels", data) for data in ids]

        # not all masks have both a corresponding s1 and s2 image
        # filter out non-existing files 
        if self.imagery == 's1 and s2':
            valid_s1, valid_s2, valid_masks = [], [], []

            for s1, s2, m in zip(s1images, s2images, masks):
                if os.path.exists(s1) and os.path.exists(s2) and os.path.exists(m):
                    valid_s1.append(s1)
                    valid_s2.append(s2)
                    valid_masks.append(m)

            self.s1images = valid_s1
            self.s2images = valid_s2
            self.masks = valid_masks

        elif self.imagery == 's1':
            valid_s1, valid_masks = [], []

            for s1, m in zip(s1images, masks):
                if os.path.exists(s1) and os.path.exists(m):
                    valid_s1.append(s1)
                    valid_masks.append(m)

            self.s1images = valid_s1
            self.s2images = s2images
            self.masks = valid_masks

        elif self.imagery == 's2':
            valid_s2, valid_masks = [], []

            for s2, m in zip(s2images, masks):
                if os.path.exists(s2) and os.path.exists(m):
                    valid_s2.append(s2)
                    valid_masks.append(m)

            self.s1images = s1images
            self.s2images = valid_s2
            self.masks = valid_masks

        return self.s1images, self.s2images, self.masks

    def __len__(self):
        return len(self.masks)
    
    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
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
        if self.selection != 'rgb':
            print("The selection parameter must be set to rgb for image visualization")
            
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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.7, 0.5), ncol=2, mode='expand',
                   loc='lower left', prop={'size': 12})

        if self.imagery == 's1' or self.imagery =='s2':
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")

        elif self.imagery == 's1 and s2':
            # s1 subplot
            plt.subplot(1, 3, 1)
            plt.imshow(img[:,:,:3])
            plt.title("S1")
            plt.axis("off")

            # s2 subplot
            plt.subplot(1, 3, 2)
            plt.imshow(img[:,:,3:6])
            plt.title("S2")
            plt.axis("off")

            # mask subplot
            plt.subplot(1, 3, 3)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig