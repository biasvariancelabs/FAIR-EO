import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
import json
import tifffile
from matplotlib.patches import Patch
from shapely import wkt
from skimage.draw import polygon

from aitlas.base import BaseDataset
from .schemas import xView2Schema
from aitlas.utils import image_loader


"""
The xView2 dataset is designed for automated building damage assessment in the aftermath of natural disasters 
using satellite imagery. It was created as part of the xView2 Challenge organized by DARPA and the DIUx xView 
Team. The dataset contains pre- and post-disaster imagery along with detailed annotations marking buildings 
and classifying them into damage levels: no damage, minor, major, and destroyed.
"""

class xView2Dataset(BaseDataset):

    url = "https://xview2.org/dataset"
    name = "xView2"
    schema = xView2Schema
    labels = ["no-data","no-damage","minor-damage","major-damage","destroyed","un-classified"]
    color_mapping = [[0,0,0],[92,92,92],[0,255,0],[255,0,0],[0,0,255],[255,255,0]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.classification = self.config.classification
        self.data = self.load_dataset(self.data_dir)

        if self.classification == "binary":
            self.labels = ["no-damage","damage"]
            self.color_mapping = [[0,0,0],[255,255,255]]

    def __getitem__(self, index):
        img_id = self.data[index]

        pre_path = os.path.join(self.data_dir, 'images', f'{img_id}_pre_disaster.tif')
        post_path = os.path.join(self.data_dir, 'images', f'{img_id}_post_disaster.tif')

        pre_image = image_loader(pre_path)
        post_image = image_loader(post_path)

        mask_2d = np.zeros((1024, 1024), dtype=np.int16)

        # Get affine transform from GeoTIFF tags
        with tifffile.TiffFile(post_path) as tif:
            tags = tif.pages[0].tags
            model_pixelscale = tags['ModelPixelScaleTag'].value   
            model_tiepoint  = tags['ModelTiepointTag'].value    

        sx, sy = model_pixelscale[0], model_pixelscale[1]
        x0 = model_tiepoint[3]   # origin longitude
        y0 = model_tiepoint[4]   # origin latitude

        def geo_to_pixel(lon, lat):
            col = (lon - x0) / sx
            row = (y0 - lat) / sy  
            return col, row

        # Load GeoJSON label file
        label_file = os.path.join(self.data_dir, "labels", f'{img_id}_post_disaster.json')
        with open(label_file, "r") as f:
            data = json.load(f)

        annotations = data["features"]["lng_lat"]
        for annotation in annotations:
            label = annotation["properties"]["subtype"]
            if self.classification == "binary":
                if label in ["no-data","no-damage","un-classified"]:
                    label = "no-damage"
                elif label in ["minor-damage","major-damage","destroyed"]:
                    label = "damage"

            wkt_geom = annotation["wkt"]

            if pd.isna(wkt_geom) or 'EMPTY' in str(wkt_geom):
                continue

            geometry = wkt.loads(wkt_geom)
            pixel_coords = [geo_to_pixel(x, y) for x, y in geometry.exterior.coords]
            mask_2d = self.add_building(mask_2d, pixel_coords, label)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        self.data = []
        for id in os.listdir(os.path.join(data_dir, 'images')):
            if fnmatch.fnmatch(id, "*pre_disaster*"):
                self.data.append(id[: -17])

        return self.data

    def add_building(self, mask:np.ndarray, coords:list, label:str)->np.ndarray:
        pixel_coords = [(int(round(x)), int(round(y))) for x, y in coords]

        if len(pixel_coords) < 4:
            return mask

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = self.labels.index(label)
        return mask

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (pre_img, post_img), mask = data

        # Create mask visualization
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

        fig = plt.figure(figsize=(15, 6))
        plt.subplots_adjust(top=0.85, bottom=0.1)

        # Add legend at the top
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
                    loc='center', prop={'size': 12})

        plt.subplot(1, 3, 1)
        plt.imshow(pre_img)
        plt.title("Pre-event")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(post_img)
        plt.title("Post-event")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img_mask)
        plt.title("Mask")
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig


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