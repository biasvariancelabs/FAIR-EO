import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import fnmatch
from matplotlib.patches import Patch
import math

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import SACropTypeSchema

"""
This dataset was produced as part of the Radiant Earth Spot the Crop Challenge 
(https://zindi.africa/hackathons/radiant-earth-spot-the-crop-hackathon). The objective 
of the competition was to create a machine learning model to classify fields by crop 
type from images collected during the growing season by the Sentinel-2 and Sentinel-1 
satellites.
"""


class SACropTypeDataset(SemanticSegmentationDataset):
    url = "https://source.coop/repositories/radiantearth/south-africa-crops-competition/access"

    labels = ["no-data","lucerne/medics","planted pastures","fallow","wine grapes","weeds","small grain grazing","wheat","canola","rooibos"]
    color_mapping = [[0,0,0],[255,0,0],[255,255,0],[192,192,0],[0,255,0],[125,125,125],[0,0,255],[0,255,255],[198,196,255],[0,0,192]] 
    name = "SA Crop Type"
    schema = SACropTypeSchema
    
    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.date = self.config.date

    def dataloader(self):
        """Create and return a dataloader for the dataset"""
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # drop_last=True,
            collate_fn=lambda x: x,
        )

    def __getitem__(self, index):
        if self.date == None: #considers the whole time series
            image_list=[]
            if fnmatch.fnmatch(self.data_dir, '*/s1/*'): #Sentinel-1 imagery
                for i in np.arange(len(self.images_VH[index])):
                    image_VH = image_loader(self.images_VH[index][i])
                    image_VV = image_loader(self.images_VV[index][i])
                    image = np.dstack((image_VH,image_VV))
                    image_list.append(image)
                
            if fnmatch.fnmatch(self.data_dir, '*/s2/*'): #Sentinel-2 imagery
                image_list=[]
                for i in np.arange(len(self.images['images_B01'][index])):
                    images = {}
                    for x in ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']:
                        images[f'images_{x}'] = image_loader(self.images[f'images_{x}'][index][i])
                    image = np.dstack((images[f'images_{x}'] for x in ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']))
                    image_list.append(image)
            image = np.array(image_list)

        else: #considers the imagery for a specific date
            date = self.date
            if fnmatch.fnmatch(self.data_dir, '*/s1/*'): #Sentinel-1 imagery
                image_VV = image_loader(self.images_VV[index][date])
                image_VH = image_loader(self.images_VH[index][date])
                image = np.dstack((image_VH,image_VV))

            if fnmatch.fnmatch(self.data_dir, '*/s2/*'): #Sentinel-2 imagery         
                    images = {}
                    for x in ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']:
                        images[f'images_{x}'] = image_loader(self.images[f'images_{x}'][index][date])
                    image = np.dstack((images[f'images_{x}'] for x in ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']))

        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        image, mask = self.apply_transformations(image, mask)
        return image, mask
    
    def __len__(self):
        return len(self.masks)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        regions = os.listdir(os.path.join(data_dir, "imagery")) #each region contains a time series

        if fnmatch.fnmatch(data_dir, '*/s1/*'): #Sentinel-1 imagery 
            self.images_VH=[]
            self.images_VV=[]
            for i in regions:
                images_per_region_VH = []
                images_per_region_VV = []
                dates = os.listdir(os.path.join(data_dir,"imagery",i))
                for j in dates:
                    for h in os.listdir(os.path.join(data_dir,"imagery",i,j)):
                        if fnmatch.fnmatch(h,'*_VH_10m.tif'):
                            images_per_region_VH.append(os.path.join(data_dir,"imagery",i,j,h))
                        if fnmatch.fnmatch(h,'*_VV_10m.tif'):
                            images_per_region_VV.append(os.path.join(data_dir,"imagery",i,j,h))

                self.images_VH.append(images_per_region_VH)
                self.images_VV.append(images_per_region_VV)

        if fnmatch.fnmatch(data_dir, '*/s2/*'): #Sentinel-2 imagery
            self.images = {}
            for x in ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']:
                self.images[f'images_{x}'] = []
        
            for i in regions:
                images_per_region = {}
                for x in ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']:
                    images_per_region[f'images_per_region_{x}'] = []
                dates = os.listdir(os.path.join(data_dir,"imagery",i))
                for j in dates:
                    for h in os.listdir(os.path.join(data_dir,"imagery",i,j)):
                        for x in ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']:
                            if fnmatch.fnmatch(h,f'*_{x}_10m.tif'):
                                images_per_region[f'images_per_region_{x}'].append(os.path.join(data_dir,"imagery",i,j,h))

                for x in ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']:
                    self.images[f'images_{x}'].append(images_per_region[f'images_per_region_{x}'])

        self.masks = [os.path.join(data_dir, "labels", image_id +'.tif') for image_id in regions]

    def show_image(self, index, show_title=False):
        img, mask = self.__getitem__(index)
        img = img[:,:,1] #plots the second band
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
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig
    
    def show_timeseries(self, index, longitude, latitude):
        image, label = self.__getitem__(index)
        location = []
        for i in np.arange(len(image)):
            location.append(image[i][longitude,latitude])
        crop = label[longitude,latitude]
        for j, k in enumerate(self.labels):
            if crop[j]==1:
                crop_name = k   
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(
            f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n",
            fontsize=14,
        )
        ax.plot(location)
        if fnmatch.fnmatch(self.data_dir, '*/s1/*'):     
            ax.legend(['VH','VV'])
        if fnmatch.fnmatch(self.data_dir, '*/s2/*'):     
            ax.legend(['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'])
        ax.set_ylabel("ρ") 

        return fig

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):  #because images have different sizes
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count