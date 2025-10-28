import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.draw import polygon
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from .semantic_segmentation import SemanticSegmentationDataset
from .schemas import FAIREOSchema
from ..utils import image_loader


"""
SpaceNet 7 is a large-scale dataset for multi-temporal analysis of urban development. It consists of 24 
consecutive monthly mosaic images, comprising more than 40000 km2 of imagery and over 500000 unique building 
footprint labels. The satellite imagery was collected by Planet Labs' Dove Satellites. It enables research on 
building detection, urban change detection, and foundation mapping in rapidly evolving areas.
"""


class SpaceNet7Dataset(SemanticSegmentationDataset):

    url = ["https://medium.com/the-downlinq/announcing-spacenet-7-the-multi-temporal-urban-development-challenge-53f68e24b9fa"]
    name = "SpaceNet 7 Multi-Temporal Urban Development"
    schema = FAIREOSchema
    labels = ["non-building","building"]
    color_mapping = [[0,0,0],[255,255,255]]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode

    def __getitem__(self, index):
        scene_id, img_id = self.data[index]

        image = image_loader(os.path.join(self.data_dir,scene_id, "images", f'{img_id}.tif'))[:,:,:3]
        image = image.astype(np.float32)

        # normalize if values are large
        if image.max() > 1.0:
            image = image / 65535.0  # Normalize uint16 range [0, 65535] to [0, 1]

        img_h, img_w = image.shape[:2]
        mask_2d = np.zeros((img_h, img_w), dtype = np.int64)

        # Load GeoJSON data 
        # it contains the polygons coordinates in pixels
        annotation = os.path.join(self.data_dir,scene_id, "labels_match_pix",f'{img_id}_Buildings.geojson')
        gdf = gpd.read_file(annotation)

        for index, row in gdf.iterrows():
            geometry = row['geometry']

            if isinstance(geometry, Polygon):
                polygons = [geometry]
            elif isinstance(geometry, MultiPolygon):
                polygons = list(geometry.geoms)

            for poly in polygons:
                coords = list(poly.exterior.coords)
                mask_2d = self.add_building(mask_2d, coords)

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[mask_2d]

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        img_ids = []
        pairs = []
        for scene_id in os.listdir(data_dir):
            img_list = []
            for filename in os.listdir(os.path.join(data_dir,scene_id, "images")):
                if filename.endswith('.tif'):
                    img_id = filename[:-4]
                    img_ids.append((scene_id,img_id))
                    img_list.append(img_id)
            pairs.append((scene_id, img_list))
        self.data = img_ids
        self.scenes = pairs

    def add_building(self, mask:np.ndarray, coords:list)->np.ndarray:
        pixel_coords = [(int(round(x)), int(round(y))) for x, y, *_ in coords]

        if len(pixel_coords) < 4:
            return mask

        # Extract x and y coordinates for polygon filling
        x_coords = [coord[0] for coord in pixel_coords[:-1]]
        y_coords = [coord[1] for coord in pixel_coords[:-1]]

        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
        mask[rr, cc] = 1
        return mask

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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.0, 0.80, 1.0, 0.05), ncol=len(self.labels),
                    loc='upper center', prop={'size': 12})
        plt.subplot(1, 2, 1)

        img_normalized = (img - img.min()) / (img.max() - img.min())
        if img.max() == img.min():
            img_normalized = np.zeros_like(img)

        plt.imshow(img_normalized)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig