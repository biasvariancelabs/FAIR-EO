import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import fnmatch
from matplotlib.patches import Patch
import math

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset was created as part of a competition. It's motivation was to achieve 
automatic prediction of the list of species most likely to be observed at a given 
location.
"""


class GeoLifeCLEF2021Dataset(BaseDataset):
    url = "https://www.kaggle.com/competitions/geolifeclef-2021/data"

    labels = ["Missing Data",
    "Annual Summer Crops",
    "Annual Winter Crops",
    "Broad-leaved Forests",
    "Coniferous Forests",
    "Natural Grasslands",
    "Woody Moorlands",
    "Continuous Urban Fabric",
    "Discontinuous Urban Fabric",
    "Industrial and Commercial Units",
    "Road Surfaces",
    "Bare Rock",
    "Beaches, Dunes, and Sands",
    "Water Bodies",
    "Glacier and Perpetual Snow",
    "Intensive Grasslands",
    "Orchards",
    "Vineyards",
    "Open Water",
    "Perennial Ice/Snow",
    "Developed, Open Space",
    "Developed, Low Intensity",
    "Developed, Medium Intensity",
    "Developed, High Intensity",
    "Barren Land (Rock/Sand/Clay)",
    "Deciduous Forest",
    "Evergreen Forest",
    "Mixed Forest",
    "Shrub/Scrub",
    "Grassland/Herbaceous",
    "Pasture/Hay",
    "Cultivated Crops",
    "Woody Wetlands",
    "Emergent Herbaceous Wetlands",
    ]
    color_mapping = [
    [0, 0, 0],           # Missing Data
    [255, 255, 0],       # Annual Summer Crops
    [255, 165, 0],       # Annual Winter Crops
    [34, 139, 34],       # Broad-leaved Forests
    [0, 100, 0],         # Coniferous Forests
    [144, 238, 144],     # Natural Grasslands
    [139, 69, 19],       # Woody Moorlands
    [178, 34, 34],       # Continuous Urban Fabric
    [220, 20, 60],       # Discontinuous Urban Fabric
    [128, 0, 0],         # Industrial and Commercial Units
    [105, 105, 105],     # Road Surfaces
    [169, 169, 169],     # Bare Rock
    [255, 248, 220],     # Beaches, Dunes, and Sands
    [0, 191, 255],       # Water Bodies
    [240, 248, 255],     # Glacier and Perpetual Snow
    [124, 252, 0],       # Intensive Grasslands
    [255, 215, 0],       # Orchards
    [218, 165, 32],      # Vineyards
    [30, 144, 255],      # Open Water
    [224, 255, 255],     # Perennial Ice/Snow
    [192, 192, 192],     # Developed, Open Space
    [255, 160, 122],     # Developed, Low Intensity
    [255, 99, 71],       # Developed, Medium Intensity
    [139, 0, 0],         # Developed, High Intensity
    [222, 184, 135],     # Barren Land (Rock/Sand/Clay)
    [0, 128, 0],         # Deciduous Forest
    [0, 100, 0],         # Evergreen Forest
    [85, 107, 47],       # Mixed Forest
    [160, 82, 45],       # Shrub/Scrub
    [173, 255, 47],      # Grassland/Herbaceous
    [152, 251, 152],     # Pasture/Hay
    [255, 255, 102],     # Cultivated Crops
    [46, 139, 87],       # Woody Wetlands
    [102, 205, 170],     # Emergent Herbaceous Wetlands
    ]

    name = "GeoLifeCLEF 2021"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.images,self.masks = self.load_dataset(self.data_dir, self.csv_file)


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        img_ids = []
        split_list = []
        
        aois = os.listdir(os.path.join(data_dir, "patches_sample"))
        for aoi in aois:
            # load only train or test images
            if self.mode: 
                csv_file = f'/home/dragik/data/FAIR_EO/GeoLifeClef2021/data/observations/observations_{aoi}_{self.mode}.csv'
                with open(csv_file, mode ='r')as file:
                    csvFile = csv.reader(file)
                    for lines in csvFile:
                        split_list.append(lines[0].split(';')[0])
                    split_list = split_list[1 :]

                id = os.listdir(os.path.join(data_dir, "patches_sample", aoi,'00'))
                for i in id:
                    path = os.listdir(os.path.join(data_dir, "patches_sample", aoi,'00', i))
                    for filename in path:
                        if fnmatch.fnmatch(filename,'*rgb.jpg'):
                            img_id = filename[: filename.rfind('_rgb')]
                            if img_id in split_list:
                                img_ids.append(os.path.join(aoi, '00', i, img_id))

            # load all images
            elif self.mode == None: 
                id = os.listdir(os.path.join(data_dir, "patches_sample", aoi,'00'))
                for i in id:
                    path = os.listdir(os.path.join(data_dir, "patches_sample", aoi,'00', i))
                    for filename in path:
                        if fnmatch.fnmatch(filename,'*rgb.jpg'):
                            img_id = filename[: filename.rfind('_rgb')]
                            img_ids.append(os.path.join(aoi, '00', i, img_id))

        self.images = [os.path.join(data_dir, "patches_sample", image_id+'_rgb.jpg') for image_id in img_ids]
        self.masks = [os.path.join(data_dir, "patches_sample", image_id+'_landcover.tif') for image_id in img_ids]

        return self.images, self.masks

    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.images)

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0, height_factor, 1, 0), ncol=3, mode='expand',
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
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask